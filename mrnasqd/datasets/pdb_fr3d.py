"""Experimental RNA secondary structures from PDB chains annotated by FR3D.

Sources (all public, fetched once and cached under ``raw/``):
  * BGSU RNA 3D Hub non-redundant list: one representative chain per
    equivalence class (``nrlist/download/<release>/<cutoff>/csv``).
  * RCSB GraphQL: experimental method, resolution, entity type and the
    canonical one-letter sequence (modified residues mapped to parent base).
  * PDBe residue listing: entity sequence index <-> author numbering, and
    which residues were observed.
  * BGSU FR3D base-pair annotations (``pdb/<ID>/interactions/fr3d/basepairs/csv``).

Reference structure = FR3D ``cWW`` pairs (exact family, not near ``ncWW``)
between AU, GC and GU, within the representative chain, in model 1, with no
symmetry operator.  Non-canonical pairs are counted but not scored, the usual
convention for secondary-structure benchmarks.  Every rejected chain gets a
reason code so the selection can be audited.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import requests

from mrnasqd import RAW_DIR, baseline_path

baseline_path()
from genus import compute_genus  # noqa: E402  (baseline module)


NRLIST_URL = 'http://rna.bgsu.edu/rna3dhub/nrlist/download/{release}/{cutoff}/csv'
FR3D_URL = 'http://rna.bgsu.edu/rna3dhub/pdb/{pdb}/interactions/fr3d/basepairs/csv'
PDBE_RESIDUES_URL = ('https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/'
                     '{pdb}/chain/{chain}')
RCSB_GRAPHQL_URL = 'https://data.rcsb.org/graphql'

CANONICAL = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'),
             ('G', 'U'), ('U', 'G')}
MIN_HAIRPIN_SPAN = 4          # j - i >= 4, same constraint as the candidates
ACCEPTED_METHODS = {'X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY',
                    'SOLUTION NMR', 'SOLID-STATE NMR'}

_RCSB_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    exptl { method }
    rcsb_entry_info { resolution_combined }
    rcsb_accession_info { initial_release_date }
    polymer_entities {
      entity_poly { type rcsb_sample_sequence_length pdbx_seq_one_letter_code_can }
      rcsb_polymer_entity_container_identifiers { entity_id auth_asym_ids }
      rcsb_polymer_entity { pdbx_description }
      rcsb_entity_source_organism { scientific_name }
    }
  }
}
"""


# ── HTTP with an on-disk cache ──────────────────────────────────────────────

_session = requests.Session()
_session.headers['User-Agent'] = 'mRNA_SQD dataset builder (academic use)'


def _cached_get(url: str, cache_path: str, retries: int = 4,
                timeout: float = 120.0) -> bytes:
    """GET ``url`` once; later calls read ``cache_path``."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return f.read()
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 404:
                body = b''
            else:
                r.raise_for_status()
                body = r.content
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = cache_path + '.part'
            with open(tmp, 'wb') as f:
                f.write(body)
            os.replace(tmp, cache_path)
            return body
        except Exception as err:  # network hiccup: back off and retry
            last_err = err
            time.sleep(2 ** attempt)
    raise RuntimeError(f'GET failed after {retries} tries: {url}: {last_err}')


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ── 1. Non-redundant list ───────────────────────────────────────────────────

@dataclass
class NRClass:
    class_id: str
    representative: str          # e.g. '437D|1|A' or '1ABC|1|A+1ABC|1|B'
    n_members: int

    @property
    def is_single_chain(self) -> bool:
        return '+' not in self.representative

    @property
    def pdb_model_chain(self) -> Tuple[str, str, str]:
        pdb, model, chain = self.representative.split('|')
        return pdb.upper(), model, chain


def load_nrlist(release: str = 'current', cutoff: str = '4.0A',
                raw_dir: str = RAW_DIR) -> Tuple[List[NRClass], str]:
    path = os.path.join(raw_dir, f'nrlist_{release}_{cutoff}.csv')
    _cached_get(NRLIST_URL.format(release=release, cutoff=cutoff), path)
    classes: List[NRClass] = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            classes.append(NRClass(class_id=row[0], representative=row[1],
                                   n_members=len(row[2].split(','))))
    return classes, path


# ── 2. RCSB entry metadata ──────────────────────────────────────────────────

def fetch_rcsb_entries(pdb_ids: List[str], raw_dir: str = RAW_DIR,
                       batch: int = 100) -> Dict[str, dict]:
    """Return {PDB_ID: entry_json}; batches cached as raw/rcsb/<hash>.json."""
    out: Dict[str, dict] = {}
    ids = sorted(set(p.upper() for p in pdb_ids))
    for k in range(0, len(ids), batch):
        chunk = ids[k:k + batch]
        key = hashlib.sha1(','.join(chunk).encode()).hexdigest()[:16]
        path = os.path.join(raw_dir, 'rcsb', f'{key}.json')
        if not os.path.exists(path):
            for attempt in range(4):
                try:
                    r = _session.post(RCSB_GRAPHQL_URL, timeout=180, json={
                        'query': _RCSB_QUERY, 'variables': {'ids': chunk}})
                    r.raise_for_status()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(r.text)
        with open(path) as f:
            data = json.load(f)
        for entry in (data.get('data') or {}).get('entries') or []:
            if entry:
                out[entry['rcsb_id'].upper()] = entry
    return out


def _entity_for_chain(entry: dict, auth_chain: str) -> Optional[dict]:
    for ent in entry.get('polymer_entities') or []:
        ids = ent['rcsb_polymer_entity_container_identifiers']
        if auth_chain in (ids.get('auth_asym_ids') or []):
            return ent
    return None


# ── 3. PDBe residue listing (index <-> author numbering) ────────────────────

def fetch_residue_map(pdb: str, chain: str, entity_id: str,
                      raw_dir: str = RAW_DIR
                      ) -> Tuple[Dict[Tuple[int, str], int], Dict[int, dict]]:
    """Return ({(auth_num, ins): seq_index0}, {seq_index0: residue_json}).

    PDBe lists ions, ligands and waters that share the author chain ID as
    separate molecules, each numbered from 1, so only the polymer entity
    ``entity_id`` is read.
    """
    path = os.path.join(raw_dir, 'pdbe_residues', f'{pdb}_{chain}.json')
    body = _cached_get(PDBE_RESIDUES_URL.format(pdb=pdb.lower(), chain=chain),
                       path)
    if not body:
        return {}, {}
    data = json.loads(body)
    by_auth: Dict[Tuple[int, str], int] = {}
    by_idx: Dict[int, dict] = {}
    for mol in data.get(pdb.lower(), {}).get('molecules', []):
        if str(mol.get('entity_id')) != str(entity_id):
            continue
        for ch in mol.get('chains', []):
            if ch.get('chain_id', chain) != chain:
                continue
            for res in ch.get('residues', []):
                idx = int(res['residue_number']) - 1
                ins = (res.get('author_insertion_code') or '').strip()
                by_auth[(int(res['author_residue_number']), ins)] = idx
                by_idx[idx] = res
    return by_auth, by_idx


# ── 4. FR3D base pairs ──────────────────────────────────────────────────────

@dataclass
class Unit:
    pdb: str
    model: str
    chain: str
    resname: str
    resnum: int
    ins: str
    symop: str

    @classmethod
    def parse(cls, uid: str) -> 'Unit':
        f = uid.split('|')
        f += [''] * (9 - len(f))
        return cls(pdb=f[0].upper(), model=f[1], chain=f[2], resname=f[3],
                   resnum=int(f[4]), ins=f[7].strip(), symop=f[8].strip())


def fetch_fr3d_pairs(pdb: str, raw_dir: str = RAW_DIR
                     ) -> List[Tuple[Unit, str, Unit]]:
    path = os.path.join(raw_dir, 'fr3d', f'{pdb}.csv')
    body = _cached_get(FR3D_URL.format(pdb=pdb), path, timeout=300)
    rows = []
    for rec in csv.reader(io.StringIO(body.decode('utf-8', 'replace'))):
        if len(rec) < 3:
            continue
        try:
            rows.append((Unit.parse(rec[0]), rec[1].strip(), Unit.parse(rec[2])))
        except (ValueError, IndexError):
            continue
    return rows


# ── 5. Build one chain record ───────────────────────────────────────────────

@dataclass
class ChainRecord:
    id: str
    nr_class: str
    nr_members: int
    pdb: str
    chain: str
    method: str
    resolution: Optional[float]
    release_date: str
    description: str
    organism: str
    sequence: str
    length: int
    pairs: List[Tuple[int, int]]
    dotbracket: str
    genus: int
    topology_class: str
    n_cww_total: int
    n_noncanonical_cww: int
    n_conflict_dropped: int
    n_short_span_dropped: int
    n_noncww_basepairs: int
    unobserved_fraction: float
    nonstandard_fraction: float
    resname_mismatches: int
    flags: List[str] = field(default_factory=list)


def pairs_to_dotbracket(pairs: List[Tuple[int, int]], n: int) -> str:
    """Multi-bracket dot-bracket: greedy page assignment over ()[]{}<>."""
    brackets = ['()', '[]', '{}', '<>', 'Aa', 'Bb', 'Cc', 'Dd']
    db = ['.'] * n
    pages: List[List[Tuple[int, int]]] = []
    for i, j in sorted(pairs):
        for p, page in enumerate(pages):
            if all(not (a < i < b < j or i < a < j < b) for a, b in page):
                page.append((i, j))
                break
        else:
            pages.append([(i, j)])
            p = len(pages) - 1
        o, c = brackets[min(p, len(brackets) - 1)]
        db[i], db[j] = o, c
    return ''.join(db)


class Rejected(Exception):
    def __init__(self, code: str, detail: str = ''):
        super().__init__(code)
        self.code, self.detail = code, detail


def build_chain_record(nr: NRClass, entry: dict, raw_dir: str = RAW_DIR,
                       min_len: int = 20, max_len: int = 150,
                       max_unobserved: float = 0.05,
                       max_nonstandard: float = 0.05,
                       max_resolution: float = 4.0) -> ChainRecord:
    pdb, model, chain = nr.pdb_model_chain
    if entry is None:
        raise Rejected('no_rcsb_entry')
    methods = [e['method'] for e in entry.get('exptl') or []]
    method = '+'.join(methods)
    if not any(m in ACCEPTED_METHODS for m in methods):
        raise Rejected('method', method)
    res_list = (entry.get('rcsb_entry_info') or {}).get('resolution_combined')
    resolution = res_list[0] if res_list else None
    is_nmr = any('NMR' in m for m in methods)
    if not is_nmr and (resolution is None or resolution > max_resolution):
        raise Rejected('resolution', str(resolution))
    ent = _entity_for_chain(entry, chain)
    if ent is None:
        raise Rejected('chain_not_in_entry', chain)
    poly = ent['entity_poly']
    if poly['type'] != 'polyribonucleotide':
        raise Rejected('not_rna', poly['type'])
    seq = (poly['pdbx_seq_one_letter_code_can'] or '').replace('\n', '').upper()
    seq = seq.replace('T', 'U')
    n = len(seq)
    if not (min_len <= n <= max_len):
        raise Rejected('length', str(n))
    nonstd = sum(c not in 'ACGU' for c in seq) / n
    if nonstd > max_nonstandard:
        raise Rejected('nonstandard_residues', f'{nonstd:.3f}')

    entity_id = ent['rcsb_polymer_entity_container_identifiers']['entity_id']
    by_auth, by_idx = fetch_residue_map(pdb, chain, entity_id, raw_dir)
    if not by_idx:
        raise Rejected('no_residue_map')
    if len(by_idx) != n:
        raise Rejected('residue_map_length', f'{len(by_idx)} vs {n}')
    unobs = sum(1 for r in by_idx.values()
                if float(r.get('observed_ratio') or 0) <= 0) / n
    if unobs > max_unobserved:
        raise Rejected('unobserved', f'{unobs:.3f}')

    rows = fetch_fr3d_pairs(pdb, raw_dir)
    if not rows:
        raise Rejected('no_fr3d_annotation')

    cww: Dict[Tuple[int, int], Tuple[str, str]] = {}
    n_noncww = 0
    mismatches = 0
    for u1, fam, u2 in rows:
        if (u1.model != model or u2.model != model or u1.chain != chain
                or u2.chain != chain or u1.symop or u2.symop):
            continue
        i = by_auth.get((u1.resnum, u1.ins))
        j = by_auth.get((u2.resnum, u2.ins))
        if i is None or j is None or i == j:
            continue
        for u, idx in ((u1, i), (u2, j)):
            listed = by_idx[idx].get('residue_name', '')
            if listed != u.resname:
                mismatches += 1
        a, b = min(i, j), max(i, j)
        if fam == 'cWW':
            cww[(a, b)] = (seq[a], seq[b])
        elif not fam.startswith('n'):
            n_noncww += 1
    if mismatches:
        raise Rejected('resname_mismatch', str(mismatches))

    canonical = [(a, b) for (a, b), bp in cww.items() if bp in CANONICAL]
    n_noncanon = len(cww) - len(canonical)
    short = [(a, b) for a, b in canonical if b - a < MIN_HAIRPIN_SPAN]
    canonical = [(a, b) for a, b in canonical if b - a >= MIN_HAIRPIN_SPAN]

    # A nucleotide may carry two cWW annotations (bifurcated / alt-loc
    # geometry).  Neither assignment is trustworthy, so both are dropped.
    count: Dict[int, int] = {}
    for a, b in canonical:
        count[a] = count.get(a, 0) + 1
        count[b] = count.get(b, 0) + 1
    clean = sorted((a, b) for a, b in canonical
                   if count[a] == 1 and count[b] == 1)
    n_conflict = len(canonical) - len(clean)
    if not clean:
        raise Rejected('no_canonical_pairs')

    genus = compute_genus(clean)
    flags: List[str] = []
    if n_conflict:
        flags.append('conflicting_cww_dropped')
    if unobs > 0:
        flags.append('has_unobserved_residues')

    org = ent.get('rcsb_entity_source_organism') or []
    return ChainRecord(
        id=f'{pdb}_{chain}', nr_class=nr.class_id, nr_members=nr.n_members,
        pdb=pdb, chain=chain, method=method,
        resolution=resolution,
        release_date=(entry.get('rcsb_accession_info') or {}).get(
            'initial_release_date', '')[:10],
        description=(ent.get('rcsb_polymer_entity') or {}).get(
            'pdbx_description', '') or '',
        organism=(org[0].get('scientific_name') if org else '') or '',
        sequence=seq, length=n, pairs=clean,
        dotbracket=pairs_to_dotbracket(clean, n), genus=genus,
        topology_class='pseudoknotted' if genus >= 1 else 'nested',
        n_cww_total=len(cww), n_noncanonical_cww=n_noncanon,
        n_conflict_dropped=n_conflict, n_short_span_dropped=len(short),
        n_noncww_basepairs=n_noncww, unobserved_fraction=round(unobs, 4),
        nonstandard_fraction=round(nonstd, 4), resname_mismatches=mismatches,
        flags=flags)


# ── 6. Whole dataset ────────────────────────────────────────────────────────

def build_dataset(release: str = 'current', cutoff: str = '4.0A',
                  raw_dir: str = RAW_DIR, workers: int = 8,
                  **filters) -> Tuple[List[ChainRecord], List[dict], dict]:
    classes, nr_path = load_nrlist(release, cutoff, raw_dir)
    exclusions: List[dict] = []
    single = []
    for c in classes:
        if c.is_single_chain:
            single.append(c)
        else:
            exclusions.append({'nr_class': c.class_id, 'id': c.representative,
                               'reason': 'multi_chain_representative',
                               'detail': ''})

    entries = fetch_rcsb_entries([c.pdb_model_chain[0] for c in single],
                                 raw_dir)

    # Cheap pre-filter from RCSB metadata so FR3D/PDBe are fetched only for
    # chains that can pass the length and type filters.
    lo, hi = filters.get('min_len', 20), filters.get('max_len', 150)
    todo = []
    for c in single:
        pdb, _, chain = c.pdb_model_chain
        entry = entries.get(pdb)
        ent = _entity_for_chain(entry, chain) if entry else None
        if ent is not None:
            poly = ent['entity_poly']
            n = poly.get('rcsb_sample_sequence_length') or 0
            if poly['type'] != 'polyribonucleotide':
                exclusions.append({'nr_class': c.class_id, 'id': f'{pdb}_{chain}',
                                   'reason': 'not_rna', 'detail': poly['type']})
                continue
            if not (lo <= n <= hi):
                exclusions.append({'nr_class': c.class_id, 'id': f'{pdb}_{chain}',
                                   'reason': 'length', 'detail': str(n)})
                continue
        todo.append(c)

    def work(c: NRClass):
        pdb, _, chain = c.pdb_model_chain
        try:
            return build_chain_record(c, entries.get(pdb), raw_dir, **filters), None
        except Rejected as rej:
            return None, {'nr_class': c.class_id, 'id': f'{pdb}_{chain}',
                          'reason': rej.code, 'detail': rej.detail}
        except Exception as err:
            return None, {'nr_class': c.class_id, 'id': f'{pdb}_{chain}',
                          'reason': 'error', 'detail': repr(err)[:200]}

    records: List[ChainRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rec, exc in pool.map(work, todo):
            if rec is not None:
                records.append(rec)
            else:
                exclusions.append(exc)

    records.sort(key=lambda r: r.id)
    manifest = {
        'nrlist_release': release, 'nrlist_cutoff': cutoff,
        'nrlist_file_sha256': sha256_file(nr_path),
        'n_classes': len(classes), 'n_single_chain': len(single),
        'n_fetched': len(todo), 'n_records': len(records),
        'n_nested': sum(r.genus == 0 for r in records),
        'n_pseudoknotted': sum(r.genus >= 1 for r in records),
        'filters': {'min_len': lo, 'max_len': hi,
                    'max_unobserved': filters.get('max_unobserved', 0.05),
                    'max_nonstandard': filters.get('max_nonstandard', 0.05),
                    'max_resolution': filters.get('max_resolution', 4.0),
                    'nmr': 'accepted (no resolution)',
                    'pair_rule': 'FR3D cWW, AU/GC/GU, j-i>=4, '
                                 'same chain, model 1, no symop, '
                                 'conflicting cWW dropped'},
        'built': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    return records, exclusions, manifest


def record_to_json(rec: ChainRecord) -> str:
    return json.dumps(asdict(rec))
