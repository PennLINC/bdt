# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Statistic vocabulary, entity normalization, and stat- composition."""

import pytest

from bdt.utils.statistics import (
    SUPPORTED_STATISTICS,
    WEIGHTED_STATISTICS,
    WORKBENCH_METHOD,
    WORKBENCH_UNWEIGHTED,
    compose_statistic_entity,
    normalize_statistic,
    parse_statistics,
)


def test_default_is_mean_alone():
    assert parse_statistics({}) == ['mean']
    assert parse_statistics({'min_coverage': 0.5}) == ['mean']


def test_requested_order_is_preserved():
    got = parse_statistics({'statistics': ['standard_deviation', 'mean']})
    assert got == ['standard_deviation', 'mean']


def test_a_single_string_is_accepted():
    assert parse_statistics({'statistics': 'mean'}) == ['mean']


def test_unsupported_statistic_names_the_supported_set():
    with pytest.raises(ValueError, match='mode'):
        parse_statistics({'statistics': ['mean', 'mode']})
    with pytest.raises(ValueError, match='standard_deviation'):
        parse_statistics({'statistics': ['trimmed_mean']})


def test_duplicates_are_rejected():
    """A repeated statistic would emit two identically-named outputs."""
    with pytest.raises(ValueError, match='duplicate'):
        parse_statistics({'statistics': ['mean', 'mean']})


def test_empty_list_is_rejected():
    with pytest.raises(ValueError, match='at least one'):
        parse_statistics({'statistics': []})


def test_bare_yaml_key_is_rejected_like_an_empty_list():
    """`statistics:` with nothing after it parses to None, not to a missing key."""
    with pytest.raises(ValueError, match='at least one'):
        parse_statistics({'statistics': None})


def test_normalize_strips_non_alphanumerics():
    assert normalize_statistic('standard_deviation') == 'standarddeviation'
    assert normalize_statistic('mean') == 'mean'


def test_normalize_keeps_plus_signs():
    """'+' is legal in a BIDS entity value and is the composition separator."""
    assert normalize_statistic('alff+mean') == 'alff+mean'


def test_compose_joins_source_first_with_a_plus():
    assert compose_statistic_entity('alff', 'mean') == 'alff+mean'
    assert compose_statistic_entity('alff', 'standard_deviation') == 'alff+standarddeviation'


def test_compose_without_a_source_statistic():
    assert compose_statistic_entity(None, 'mean') == 'mean'
    assert compose_statistic_entity('', 'standard_deviation') == 'standarddeviation'


def test_supported_set_is_exactly_nilearns_strategies():
    """The vocabulary IS nilearn's, so the volumetric path passes names through.

    Pinned against nilearn 0.14.0's own error message, which enumerates the set it
    accepts. Workbench's SAMPSTDEV (ddof=1) is deliberately excluded: it has no
    nilearn counterpart, so offering it would make the two backends disagree.
    """
    assert set(SUPPORTED_STATISTICS) == {
        'mean',
        'median',
        'sum',
        'minimum',
        'maximum',
        'standard_deviation',
        'variance',
    }


def test_every_supported_statistic_has_a_workbench_method():
    """A statistic the grayordinate path cannot express would fail at run time.

    NOTE: name coverage is NOT sufficient on its own -- see
    :data:`WORKBENCH_UNWEIGHTED`. This assertion originally stood alone and let a
    run-time crash through, because MIN/MAX have methods but reject
    ``-cifti-weights``, which the CIFTI path always passed.
    """
    assert set(WORKBENCH_METHOD) == set(SUPPORTED_STATISTICS)


def test_unweighted_statistics_are_supported_and_are_the_selections():
    """MIN/MAX reject -cifti-weights because they select rather than compute.

    Measured against wb_command: MEAN/MEDIAN/SUM/STDEV/VARIANCE all accept weights.
    If a future statistic joins the vocabulary, this is the list to re-measure --
    a wrong entry here is a crash (missing) or a silently unmasked reduction (extra).
    """
    assert set(WORKBENCH_UNWEIGHTED) < set(SUPPORTED_STATISTICS)
    assert set(WORKBENCH_UNWEIGHTED) == {'minimum', 'maximum'}


def test_weighted_statistics_are_a_subset():
    assert set(WEIGHTED_STATISTICS) <= set(SUPPORTED_STATISTICS)
