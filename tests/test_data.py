import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnasqd.datasets.pdb_fr3d import Unit, pairs_to_dotbracket  # noqa: E402
from mrnasqd.datasets.splits import (cap_clusters, family_keyword,  # noqa: E402
                                     pk_kind)
from mrnasqd.encoding import (expressible_coverage,  # noqa: E402
                              generate_substem_candidates)
from mrnasqd.metrics import structure_metrics  # noqa: E402


def test_unit_parse_altloc_and_symop():
    u = Unit.parse('437D|1|A|C|14||B')
    assert (u.pdb, u.model, u.chain, u.resname, u.resnum) == ('437D', '1', 'A', 'C', 14)
    assert u.ins == '' and u.symop == ''
    v = Unit.parse('1ABC|1|A|G|5|||A|7_555')
    assert v.ins == 'A' and v.symop == '7_555'


def test_dotbracket_pages():
    pairs = [(2, 17), (3, 16), (9, 25), (10, 24)]
    db = pairs_to_dotbracket(pairs, 28)
    assert db[2] == '(' and db[17] == ')' and db[9] == '[' and db[25] == ']'


def test_maximal_only_cannot_express_shorter_true_helix():
    # Baseline synth_hairpin_003: labelled 3-bp stem inside a 4-bp maximal stem.
    seq = 'GCAAUAUAUUGC'
    truth = [(0, 11), (1, 10), (2, 9)]
    maximal = generate_substem_candidates(seq, 2, 'maximal')
    trunc = generate_substem_candidates(seq, 2, 'truncations')
    assert expressible_coverage(maximal, truth) == 0
    assert expressible_coverage(trunc, truth) == 3
    assert {s.pairs for s in maximal} <= {s.pairs for s in trunc}


def test_substem_modes_nest():
    seq = 'GGGGAAAACCCCAAAGGGAAACCC'
    for m in (2, 3):
        a = {s.pairs for s in generate_substem_candidates(seq, m, 'maximal')}
        b = {s.pairs for s in generate_substem_candidates(seq, m, 'truncations')}
        c = {s.pairs for s in generate_substem_candidates(seq, m, 'all')}
        assert a <= b <= c
        assert all(len(p) >= m for p in c)


def test_pk_kind():
    nested = {'genus': 0, 'pairs': [(0, 10), (1, 9)]}
    htype = {'genus': 1, 'pairs': [(0, 12), (1, 11), (6, 20), (7, 19)]}
    tert = {'genus': 1, 'pairs': [(0, 12), (1, 11), (6, 20)]}
    assert pk_kind(nested) == 'nested'
    assert pk_kind(htype) == 'helical_pk'
    assert pk_kind(tert) == 'tertiary_pk'


def test_family_keyword():
    assert family_keyword('tRNA(Phe)') == 'trna'
    assert family_keyword('PreQ1 riboswitch') == 'riboswitch_preq1'
    assert family_keyword('5S rRNA') == '5s_rrna'
    assert family_keyword('RNA (29-MER)') == ''


def test_cap_clusters():
    recs = [{'method': 'SOLUTION NMR', 'resolution': None}] * 5 + \
           [{'method': 'X-RAY DIFFRACTION', 'resolution': 2.0}]
    keep = cap_clusters(recs, [0] * 6, cap=2)
    assert sum(keep) == 2 and keep[5]


def test_metrics():
    m = structure_metrics([(0, 10), (1, 9), (2, 7)], [(0, 10), (1, 9), (3, 8)], 12)
    assert (m['tp'], m['fp'], m['fn']) == (2, 1, 1)
    assert abs(m['sensitivity'] - 2 / 3) < 1e-12
    assert 0 < m['mcc'] < 1
