# region - imports
import functools
import sys
import configargparse

from smilelogging.logger import Logger
from smilelogging.slutils import blue, green, red, yellow, bold, update_args

# endregion - imports


# region - arg parse
argparser = configargparse.ArgumentParser()

argparser.add_argument(
    "--experiment_name", type=str, default="", help="Experiment name")

argparser.add_argument(
    "--experiments_dir",
    type=str,
    default="./Experiments",
    help="Path of the folder to store all experiments.",)

argparser.add_argument(
    "--debug",
    action="store_true",
    help="All the logs will be saved to `Debug_Dir`",)

argparser.add_argument(
    "--no_cache", 
    action="store_true", 
    help="not cache code")

argparser.add_argument(
    "--cache_code",
    type=str,
    default="scripts/cache_code.sh",
    help="Path of the shell script to cache code",)

argparser.add_argument(
    "--resume_expid",
    type=str,
    default="",
    help="The expid used to uniquely identify an experiment",)

# Customize smilelogging setups
argparser.add_argument("--sl.ON", action="store_true")
argparser.add_argument("--sl.config", type=str, default=".smilelogging_cfg")

# endregion - arg parse


def warn_deprecated_args(old, new):
    if old in sys.argv:
        print(f"[Smilelogging Error] {old} is deprecated now, please use {new} instead and rerun")
        exit(0)


def add_update_args(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        ret = update_args(ret)
        return ret

    return wrapper


warn_deprecated_args("--project_name", "--experiment_name")
warn_deprecated_args("--project", "--experiment_name")
argparser.parse_args = add_update_args(argparser.parse_args)
