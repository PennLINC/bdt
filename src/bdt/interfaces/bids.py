"""BIDS-related interfaces for BDT."""

from json import loads

from bids.layout import Config
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink

from bdt.data import load as load_data

# NOTE: Modified for bdt's purposes
bdt_spec = loads(load_data('io_spec.json').read_text())
bids_config = Config.load('bids')
deriv_config = Config.load('derivatives')

bdt_entities = {v['name']: v['pattern'] for v in bdt_spec['entities']}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **bdt_entities}
merged_entities = [{'name': k, 'pattern': v} for k, v in merged_entities.items()]
config_entities = frozenset({e['name'] for e in merged_entities})


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink,
    using bdt's configuration files.
    """

    out_path_base = ''
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = bdt_spec['default_path_patterns']
