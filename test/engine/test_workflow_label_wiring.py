# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The secondary labels (``tsv``) edge carried alongside a role's primary output."""

import pytest

pytest.importorskip('nipype')

from nipype.interfaces import utility as niu  # noqa: E402
from nipype.pipeline import engine as pe  # noqa: E402

from bdt.engine.workflow import _identity_fields, init_bdt_wf  # noqa: E402
from bdt.spec import parse_spec  # noqa: E402


def _wf(inputs=None, outputs=None):
    wf = pe.Workflow(name='sub')
    if inputs:
        wf.add_nodes([pe.Node(niu.IdentityInterface(fields=inputs), name='inputnode')])
    if outputs:
        wf.add_nodes([pe.Node(niu.IdentityInterface(fields=outputs), name='outputnode')])
    return wf


def test_identity_fields_reads_declared_fields():
    wf = _wf(inputs=['atlas', 'atlas_labels'], outputs=['out', 'tsv'])
    assert _identity_fields(wf, 'inputnode') == {'atlas', 'atlas_labels'}
    assert _identity_fields(wf, 'outputnode') == {'out', 'tsv'}


def test_identity_fields_missing_node_is_empty():
    assert _identity_fields(_wf(), 'inputnode') == set()


# ---------------------------------------------------------------------------
# The wiring loop itself, driven through init_bdt_wf.
#
# _identity_fields is only a helper; the behaviour this task exists to add is
# the secondary ``outputnode.tsv`` -> ``inputnode.<role>_labels`` edge that
# init_bdt_wf's role-wiring loop conditionally adds.  These tests build tiny
# stub processing factories (registered into the real WORKFLOW_FACTORIES
# registry for the duration of one test via monkeypatch, so nothing leaks) and
# compile a spec through the real entry point, then inspect the compiled
# graph's edges the same way test_nipype_workflow.py does.
# ---------------------------------------------------------------------------


def _stub_factory(in_fields, out_fields):
    """A minimal nipype sub-workflow factory: bare inputnode/outputnode, unconnected.

    Mirrors the ``_wf`` helper above (and the assembly-test pattern in
    test_nipype_workflow.py): the wiring loop only inspects the declared
    IdentityInterface field names, so the nodes need not be internally wired.
    """

    def factory(node, name=None, context=None):
        wf = pe.Workflow(name=name or node.name)
        wf.add_nodes(
            [
                pe.Node(niu.IdentityInterface(fields=list(in_fields)), name='inputnode'),
                pe.Node(niu.IdentityInterface(fields=list(out_fields)), name='outputnode'),
            ]
        )
        return wf

    return factory


@pytest.fixture
def register_factory(monkeypatch):
    """Register a stub factory under a scratch action name for one test only.

    Uses monkeypatch.setitem on the real WORKFLOW_FACTORIES dict so the
    registration is automatically undone at teardown, regardless of whether the
    key was previously absent (deleted) or present (restored).
    """
    from bdt.engine.factories import WORKFLOW_FACTORIES

    def _register(action_name, in_fields, out_fields):
        monkeypatch.setitem(WORKFLOW_FACTORIES, action_name, _stub_factory(in_fields, out_fields))

    return _register


def _edges(wf):
    """{(upstream_name, downstream_name): [(src_field, dest_field), ...]}"""
    return {(u.name, v.name): data.get('connect', []) for u, v, data in wf._graph.edges(data=True)}


def _spec_with_producer_consumer(producer_action, consumer_action):
    return parse_spec(
        {
            'nodes': [
                {'name': 'src', 'action': 'select_data', 'dataset': 'stub', 'filters': {}},
                {'name': 'producer', 'action': producer_action, 'inputs': {'in_': 'src'}},
                {'name': 'consumer', 'action': consumer_action, 'inputs': {'data': 'producer'}},
            ]
        }
    )


def test_labels_edge_wired_when_both_sides_opt_in(register_factory):
    """Case 1: producer exposes outputnode.tsv, consumer declares inputnode.data_labels
    -> the secondary edge exists."""
    register_factory('stub_produce_tsv', ['in_'], ['out', 'tsv'])
    register_factory('stub_consume_labels', ['data', 'data_labels'], ['out'])
    spec = _spec_with_producer_consumer('stub_produce_tsv', 'stub_consume_labels')

    wf = init_bdt_wf(spec, {'src': '/x/f.ext'})

    edges = _edges(wf)
    assert ('outputnode.out', 'inputnode.data') in edges[('producer', 'consumer')]
    assert ('outputnode.tsv', 'inputnode.data_labels') in edges[('producer', 'consumer')]


def test_no_labels_edge_when_consumer_does_not_declare_role_labels(register_factory):
    """Case 2: producer exposes outputnode.tsv, but the consumer's inputnode has no
    ``data_labels`` field -> no secondary edge (and the primary edge is unaffected)."""
    register_factory('stub_produce_tsv2', ['in_'], ['out', 'tsv'])
    register_factory('stub_consume_no_labels', ['data'], ['out'])
    spec = _spec_with_producer_consumer('stub_produce_tsv2', 'stub_consume_no_labels')

    wf = init_bdt_wf(spec, {'src': '/x/f.ext'})

    edges = _edges(wf)
    assert ('outputnode.out', 'inputnode.data') in edges[('producer', 'consumer')]
    assert ('outputnode.tsv', 'inputnode.data_labels') not in edges[('producer', 'consumer')]


def test_no_labels_edge_when_producer_has_no_tsv(register_factory):
    """Case 3: the consumer declares inputnode.data_labels, but the producer's
    outputnode has no ``tsv`` field -> no secondary edge."""
    register_factory('stub_produce_no_tsv', ['in_'], ['out'])
    register_factory('stub_consume_labels2', ['data', 'data_labels'], ['out'])
    spec = _spec_with_producer_consumer('stub_produce_no_tsv', 'stub_consume_labels2')

    wf = init_bdt_wf(spec, {'src': '/x/f.ext'})

    edges = _edges(wf)
    assert ('outputnode.out', 'inputnode.data') in edges[('producer', 'consumer')]
    assert ('outputnode.tsv', 'inputnode.data_labels') not in edges[('producer', 'consumer')]


def test_selection_upstream_feeding_labels_consumer_does_not_raise(register_factory):
    """Case 4: a *selection* node feeds a role whose consumer declares
    ``<role>_labels``.  A selection's built object is a bare ``pe.Node`` (no
    ``.get_node()``); the guard must check ``up_kind == 'processing'`` before it
    ever calls ``_identity_fields(up_obj, 'outputnode')``, else this raises
    AttributeError instead of compiling.  This test pins that ordering: if a
    future change probes ``_identity_fields`` first, the guard crashes at
    graph-build time and this test fails loudly (AttributeError) instead of the
    reorder going unnoticed.
    """
    register_factory('stub_consume_labels3', ['data', 'data_labels'], ['out'])
    spec = parse_spec(
        {
            'nodes': [
                {'name': 'src', 'action': 'select_data', 'dataset': 'stub', 'filters': {}},
                {'name': 'consumer', 'action': 'stub_consume_labels3', 'inputs': {'data': 'src'}},
            ]
        }
    )

    wf = init_bdt_wf(spec, {'src': '/x/f.ext'})  # must not raise AttributeError

    edges = _edges(wf)
    assert ('out', 'inputnode.data') in edges[('src', 'consumer')]
    assert ('out', 'inputnode.data_labels') not in edges[('src', 'consumer')]
    assert not any(dest == 'inputnode.data_labels' for _src, dest in edges[('src', 'consumer')])
