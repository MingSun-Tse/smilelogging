import functools
import sys

import configargparse

from smilelogging.logger import Logger
from smilelogging.slutils import blue, green, red, yellow, bold, update_args


argparser = configargparse.ArgumentParser()
argparser.add_argument(
    "--experiment_name", type=str, default="", help="Experiment name."
)
argparser.add_argument(
    "--debug",
    action="store_true",
    help="All the logs will be saved to `Debug_Dir`.",
)
argparser.add_argument(
    "--resume_expid",
    type=str,
    default="",
    help="The expid used to uniquely identify an experiment.",
)


def warn_deprecated_args(old, new):
    if old in sys.argv:
        print(
            f"[Smilelogging Error] {old} is deprecated now, please use {new} instead and rerun"
        )
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
