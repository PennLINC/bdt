# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Routing tests for functional_connectivity (CIFTI vs volumetric)."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import FactoryContext, init_functional_connectivity_wf  # noqa: E402


def _match(path, entities):
    from bdt.engine.selection import Match

    return Match(path, entities)


def _node(name, role_to_upstreams):
    return SimpleNamespace(name=name, inputs=role_to_upstreams, parameters={})


def test_fc_cifti_uses_cifti_correlation():
    """Both branches now build a node named ``correlate``, so assert on the
    interface type -- a name check would pass even if CIFTI fell through to the
    volumetric path."""
    from bdt.interfaces.workbench import CiftiCorrelation

    node = _node('fc', {'timeseries': ['parc']})
    ctx = FactoryContext(resolved={'parc': _match('p.ptseries.nii', {})})
    wf = init_functional_connectivity_wf(node, context=ctx)
    assert isinstance(wf.get_node('correlate').interface, CiftiCorrelation)


def test_fc_volumetric_uses_xcpd_tsv_connect():
    """A volumetric parcellated TSV correlates via XCP-D's TSVConnect."""
    from bdt.interfaces.connectivity import TSVConnect

    node = _node('fc', {'timeseries': ['parc']})
    ctx = FactoryContext(resolved={'parc': _match('p.tsv', {})})
    wf = init_functional_connectivity_wf(node, context=ctx)
    correlate = wf.get_node('correlate')
    assert isinstance(correlate.interface, TSVConnect)
