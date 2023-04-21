from smilelogging.logger import Logger
import configargparse

argparser = configargparse.ArgumentParser()
argparser.add_argument('--project_name',
                       '--experiment_name',
                       dest='project_name',
                       type=str,
                       default='',
                       help='experiment name')
argparser.add_argument('--experiments_dir',
                       type=str,
                       default='Experiments')
argparser.add_argument('--debug',
                       action="store_true",
                       help='if debugging, if so, all the logs will be saved to `Debug_Dir`')
argparser.add_argument('--no_cache',
                       action='store_true',
                       help='not cache code')
argparser.add_argument('--cache_code',
                       type=str,
                       default='scripts/cache_code.sh',
                       help='the script to cache code')
argparser.add_argument('--resume_TimeID',
                       type=str,
                       default='',
                       help='the time ID used to uniquely identify an experiment')

# Customize smilelogging setups
argparser.add_argument('--sl.ON',
                       action='store_true')
argparser.add_argument('--sl.config',
                       type=str,
                       default='.smilelogging_cfg')
