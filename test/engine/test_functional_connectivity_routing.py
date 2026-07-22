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


def _spec_with_parcellation(statistics):
    """A two-node spec: parcellate_timeseries -> functional_connectivity."""
    from bdt.spec import parse_spec

    parameters = {'statistics': statistics} if statistics is not None else {}
    return parse_spec(
        {
            'nodes': [
                {'name': 'load', 'action': 'select_data', 'dataset': 'fmriprep'},
                {
                    'name': 'parc',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'load', 'atlas': 'load'},
                    'parameters': parameters,
                },
                {
                    'name': 'fc',
                    'action': 'functional_connectivity',
                    'inputs': {'timeseries': 'parc'},
                },
            ]
        }
    )


def _build_fc(statistics, caplog):
    import logging

    spec = _spec_with_parcellation(statistics)
    ctx = FactoryContext(spec=spec, resolved={'parc': _match('p.tsv', {})})
    with caplog.at_level(logging.WARNING, logger='nipype.workflow'):
        init_functional_connectivity_wf(spec.by_name()['fc'], context=ctx)
    return caplog.text


def test_fc_announces_which_statistic_it_correlates(caplog):
    """Role wiring carries ``out`` = the first requested statistic.

    That makes the connectivity matrix depend on YAML list order, which is easy to
    miss, so it must be stated rather than assumed.
    """
    text = _build_fc(['median', 'mean'], caplog)
    assert "'median'" in text
    assert 'first requested' in text


def test_fc_is_quiet_for_a_single_statistic(caplog):
    """No warning when there is no choice being made -- including the default."""
    assert _build_fc(['mean'], caplog) == ''
    assert _build_fc(None, caplog) == ''
