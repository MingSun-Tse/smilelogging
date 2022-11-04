from smilelogging.logger import Logger
import configargparse

argparser = configargparse.ArgumentParser()
argparser.add_argument('--project_name', '--experiment_name', dest='project_name', type=str, default='')
argparser.add_argument('--experiments_dir', type=str, default='Experiments')
argparser.add_argument('--debug', action="store_true")
argparser.add_argument('--cache_ignore', type=str, default='')
argparser.add_argument('--resume_ExpID', type=str, default='')

# Customize smilelogging setups
argparser.add_argument('--sl.ON', action='store_true')
argparser.add_argument('--sl.config', type=str, default='.smilelogging_cfg')