
import sys

from tonic.io import read_config
from tonic.models.vic.vic2netcdf import vic2nc


# --- Load config file --- #
cfg_file = sys.argv[1]

# --- Run vic2nc --- #
cfg_vic2nc = read_config(cfg_file)
options = cfg_vic2nc.pop('OPTIONS')
global_atts = cfg_vic2nc.pop('GLOBAL_ATTRIBUTES')
if not options['regular_grid']:
    domain_dict = cfg_vic2nc.pop('DOMAIN')
else:
    domain_dict = None

# set aside fields dict
fields = cfg_vic2nc

vic2nc(options, global_atts, domain_dict, fields)

