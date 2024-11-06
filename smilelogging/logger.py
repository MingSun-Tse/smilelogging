# region - imports
import builtins
import getpass
import glob
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from fnmatch import fnmatch

import numpy as np
import pynvml
import yaml

from smilelogging.slutils import blue, green, red, yellow

#* @graenys
import copy  

pjoin = os.path.join
timezone = datetime.now().astimezone().tzinfo  # Use local timezone.
# endregion - imports


# region - utils
def run_shell_command(cmd):
    """Run shell command and return the output (string) in a list.
    Args:
        cmd: shell command

    Returns:
        result: a list of the output returned by that shell command
    """
    cmd = " ".join(cmd.split())
    if " | " in cmd:  # Refer to: https://stackoverflow.com/a/13332300/12554945
        cmds = cmd.split(" | ")
        assert len(cmds) == 2, "Only support one pipe now"
        fn = subprocess.Popen(cmds[0].split(), stdout=subprocess.PIPE)
        result = subprocess.run(
            cmds[1].split(), stdin=fn.stdout, stdout=subprocess.PIPE
        )
    else:
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode("utf-8").strip().split("\n")
    return result


def moving_average(x, N=10):
    """Refer to: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean"""
    import scipy.ndimage as ndi

    return ndi.uniform_filter1d(x, N, mode="constant", origin=-(N // 2))[: -(N - 1)]


def get_experiment_path(ExpID):
    full_path = glob.glob("Experiments/*%s*" % ExpID)
    assert (
        len(full_path) == 1
    ), "There should be only ONE folder with <ExpID> in its name"
    return full_path[0]


def parse_ExpID(path):
    """parse out the ExpID from 'path', which can be a file or directory.
    Example: Experiments/AE__ckpt_epoch_240.pth__LR1.5__originallabel__vgg13_SERVER138-20200829-202307/gen_img
    Example: Experiments/AE__ckpt_epoch_240.pth__LR1.5__originallabel__vgg13_SERVER-20200829-202307/gen_img
    """
    rank = ""
    if "RANK" in path:
        rank = path.split("RANK")[1].split("-")[0]
        rank = f"RANK{rank}-"
    return rank + "SERVER" + path.split("SERVER")[1].split("/")[0]


def mkdirs(*paths, exist_ok=False):
    for p in paths:
        os.umask(0)
        os.makedirs(
            p, mode=0o777, exist_ok=exist_ok
        )  # 777 mode may not be safe but easy for now


class DoubleWriter:
    def __init__(self, f1, f2):
        self.f1, self.f2 = f1, f2

    def write(self, msg):
        self.f1.write(msg)
        self.f2.write(msg)

    def flush(self):
        self.f1.flush()
        self.f2.flush()


class LogTracker:
    """Logging all numerical results."""

    def __init__(self):
        self._metrics = OrderedDict()
        self._print_format = {}

    def update(self, k, v):
        """ """
        if ":" in k:
            k, format_ = k.split(":")
            self._print_format[k] = format_

        if k in self._metrics:
            self._metrics[k] = np.append(self._metrics[k], v)
        else:
            self._metrics[k] = np.array([v])

    def reset(self):
        self._metrics = OrderedDict()

    def get_metrics(self, k=None):
        if k is not None:
            return self._metrics[k]
        else:
            return self._metrics

    def format(self, selected=None, not_selected=None, sep=" "):
        """Format for print."""
        logstr = []
        for k, v in self._metrics.items():
            in_selected = True

            if selected is not None:
                in_selected = False
                for s in selected.split(","):
                    if fnmatch(k, s):
                        in_selected = True
                        break

            if not_selected is not None:
                for s in not_selected.split(","):
                    if fnmatch(k, s):
                        in_selected = False
                        break

            if in_selected:
                f = self._print_format[k] if k in self._print_format else f"%s"
                logstr += [f"{k} {f}" % v[-1]]
        return sep.join(logstr)

    def get_ma(self, k, window=10):
        """Moving average."""
        return moving_average(self._metrics[k], window)

# endregion - utils


class Logger(object):
    passer = {}

    def __init__(self, args, overwrite_print=False):
        self.args = args
        self.overwrite_print = overwrite_print
        self.expname = self.args.experiment_name
        self.debug = self.args.debug or "debug" in self.expname.lower()

        # Logging folder names.
        self._experiments_dir = args.experiments_dir
        self._debug_dir = "debug_folder"
        self._weights_dir = "weights"
        self._gen_img_dir = "gen_img"
        self._log_dir = "log"

        #* effysion folders
        self._checkpoints_dir = "checkpoints"
        self._model_snapshot_dir = "model_snapshot"
        self._validation_images_dir = "validation_images"

        # Handle the DDP case.
        self._figure_out_rank()
        self.ddp = self.global_rank >= 0

        # Set up a unique experiment folder.
        self.userip, self.hostname = self.get_userip()
        self.set_up_experiment_dir()
        self.set_up_logtxt()

        # Set up logging utils
        self.log_tracker = LogTracker()
        # self._set_up_py_logging() # TODO-@mst: Not finished, a lot of problems

        # Initial print
        self.print_script()
        args.CodeID = (
            self.get_CodeID()
        )  # get CodeID before 'print_args()' because it will be printed in log.
        self.print_args()

        # Cache misc environment info  (e.g., args, GPU, git, etc.) and codes.
        self.save_env_snapshot()
        if self.global_rank in [-1, 0]:  # Only the main process does caching.
            self.cache_code()

    def save_env_snapshot(self):
        """Save environment snapshot (such as args, gpu info, users, git info)."""
        # Save args.
        with open(pjoin(self.log_path, "args.yaml"), "w") as f:
            yaml.dump(self.args.__dict__, f, indent=4)

        # Save system info.
        mkdirs(pjoin(self.exp_path, "machine_information"), exist_ok = True)
        os.system("nvidia-smi >> {}".format(pjoin(self.exp_path, "machine_information/nvidia-smi.log")))
        os.system("gpustat >> {}".format(pjoin(self.exp_path, "machine_information/gpustat.log")))
        os.system("who -b >> {}".format(pjoin(self.exp_path, "machine_information/who.log")))
        os.system("who >> {}".format(pjoin(self.exp_path, "machine_information/who.log")))

        # Save git info.
        if self.use_git:
            os.system("git status >> {}".format(pjoin(self.exp_path, "machine_information/git_status.log")))
    
    def set_up_experiment_dir(self):
        """Set up a unique directory for each experiment."""
        # Set up a unique experiment folder for each process.
        experiment_path = self._get_experiment_path()

        # Output interface.
        self.exp_path = experiment_path
        self.weights_path = pjoin(experiment_path, self._weights_dir)
        self.gen_img_path = pjoin(experiment_path, self._gen_img_dir)
        self.log_path = pjoin(experiment_path, self._log_dir)
        self.logplt_path = pjoin(self.log_path, "plot")
        self.logtxt_path = pjoin(self.log_path, "log.txt")  #* @graenys
        self._cache_path = pjoin(experiment_path, ".caches")

        #* effysion specific
        self.checkpoints_dir = pjoin(experiment_path, self._checkpoints_dir)
        self.model_snapshot_dir = pjoin(experiment_path, self._model_snapshot_dir)
        self.validation_images_dir = pjoin(experiment_path, self._validation_images_dir)

        #TODO Modify as your requirement~
        mkdirs(self.log_path, exist_ok=True)
        mkdirs(self.checkpoints_dir)
        mkdirs(self.model_snapshot_dir)
        mkdirs(self.validation_images_dir)

        # When resuming experiments, do not append to existing log.txt. Instead, we
        # prefer to create a new log txt and archive the old versions.
        if os.path.exists(self.logtxt_path):
            timestamp = os.path.getmtime(self.logtxt_path)
            timestamp = datetime.fromtimestamp(timestamp).strftime(
                "lastmodified-%Y%m%d-%H%M%S"
            )
            new_f = pjoin(self.log_path, f"log_{timestamp}.txt")
            os.rename(self.logtxt_path, new_f)
    
    def print_script(self):
        script = f"hostname: {self.hostname} | userip: {self.userip}\n"
        script += "cd %s\n" % os.path.abspath(os.getcwd())
        if self.ddp:
            program = (
                "OMP_NUM_THREADS=12 MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=<ToAdd> "
                "python -m torch.distributed.run --nproc_per_node <ToAdd>"
            )
            script += " ".join([program, *sys.argv])
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                pynvml.nvmlInit()
                num_total_gpus = pynvml.nvmlDeviceGetCount()
                gpu_id = ",".join([str(x) for x in range(num_total_gpus)])
            script += " ".join(["CUDA_VISIBLE_DEVICES=%s python" % gpu_id, *sys.argv])
        # script = blue(script)
        script += "\n"
        self.print(script, unprefix=True, color='blue')

    def print_args(self):
        """Example: ('batch_size', 16) ('CodeID', 12defsd2) ('decoder', models/small16x_ae_base/d5_base.pth)
        It will sort the arg keys in alphabeta order, case insensitive.
        """
        # Build a key map for later sorting.
        key_map = {}
        for k in self.args.__dict__:
            k_lower = k.lower()
            if k_lower in key_map:
                key_map[k_lower + "_" + k_lower] = k
            else:
                key_map[k_lower] = k

        # Print in the order of sorted lower keys.
        logtmp = ""
        cnt = 0
        for k_ in sorted(key_map.keys()):
            cnt = cnt + 1
            real_key = key_map[k_]
            logtmp += "('%s': %s) " % (real_key, self.args.__dict__[real_key])
            if cnt % 5 == 0:
                logtmp += '\n'
        self.print(logtmp + "\n", unprefix=True)
    
    def _get_experiment_path(self):
        """Get a unique directory for each experiment."""
        # Get rank for each process (used in multi-process training, e.g., DDP).
        if self.global_rank == -1:
            other_ranks_folder = ""
            rank = ""
        else:
            other_ranks_folder = "" if self.global_rank == 0 else ".OtherRanks"
            rank = f"RK{self.global_rank}-"

        experiment_path = ""

        # If resuming experiments, check if the specified experiment folder exists.
        if self.args.resume_expid is not None:
            if self.args.resume_expid == "latest":
                # Select the latest experiment folder.
                exp_mark = f"%s/%s/%s_%sSVR.*" % (
                    self._experiments_dir,
                    other_ranks_folder,
                    self.expname,
                    rank,
                )
                exps = glob.glob(exp_mark)
                if len(exps) > 0:
                    experiment_path = sorted(exps)[-1]

            elif self.args.resume_expid.isdigit():
                raise NotImplementedError  # Resume a specific expid.

        if experiment_path != "":  # Use existing folder.
            self.ExpID = parse_ExpID(experiment_path)
        else:
            # Create a new folder.
            server = "SVR.%03d_" % int(self.userip.split(".")[-1])
            self.ExpID = rank + server + self._get_time_id()
            experiment_path = "%s/%s/%s_%s" % (
                self._experiments_dir,
                other_ranks_folder,
                self.expname,
                self.ExpID,
            )

        if (
            self.args.debug
        ):  # --debug has the highest priority. If --debug, all the things will be saved in `Debug_dir`.
            experiment_path = self._debug_dir
        experiment_path = os.path.normpath(experiment_path)
        return experiment_path
    
    def _get_time_id(self) -> str:
        """Get time stamp for the experiment folder name. The expid must be unique.

        Returns:
            time_id: A string that indicates the time stamp of the experiment.
        """
        time_id = datetime.now(timezone).strftime("%m-%d-%y_%H:%M:%S")
        #* @graenys
        meta_exp_id = datetime.now(timezone).strftime("%m-%d-%y_%H%M%S")
        expid = meta_exp_id[-6:]
        existing_exps = glob.glob(f"{self._experiments_dir}/*-{expid}")
        t0 = time.time()
        # Make sure the expid is unique.
        while len(existing_exps) > 0:
            time.sleep(1)
            time_id = datetime.now(timezone).strftime("%m-%d-%y_%H:%M:%S")
            #* @graenys
            meta_exp_id = datetime.now(timezone).strftime("%m-%d-%y_%H%M%S")
            expid = meta_exp_id[-6:]
            existing_exps = glob.glob(f"{self._experiments_dir}/*-{expid}")
            if time.time() - t0 > 120:
                self.print(
                    "Hanged for more than 2 mins when creating the experiment folder, "
                    "which is unusual. Please try again."
                )
                exit(1)
        return time_id
    
    
    #*===========================================================================*#
    
    def _figure_out_rank(self):
        global_rank = os.getenv("GLOBAL_RANK", -1)
        local_rank = os.getenv("LOCAL_RANK", -1)
        if local_rank != -1:  # DDP is used
            if global_rank == -1:  # single-node DDP
                global_rank = local_rank
        self.global_rank = int(global_rank)
        self.local_rank = int(local_rank)

        # Get ranks from args
        if hasattr(self.args, "global_rank") and self.args.global_rank >= 0:
            self.global_rank = self.args.global_rank
        if hasattr(self.args, "local_rank") and self.args.local_rank >= 0:
            self.local_rank = self.args.local_rank

    def get_CodeID(self) -> str:
        """Return git commit ID as CodeID.

        Returns:
            CodeID: A string that indicates the git commit ID.
        """
        git_check = run_shell_command("git rev-parse --is-inside-work-tree")
        self.use_git = len(git_check) == 1 and git_check[0] == "true"
        self.git_root = run_shell_command("git rev-parse --show-toplevel")[
            0
        ]  # The path where .git is placed

        if self.use_git:
            CodeID = run_shell_command("git rev-parse --short HEAD")[0]
            changed_files = run_shell_command("git diff --name-only")
            new_real_change = False
            for f in changed_files:
                if (
                    f and not f.endswith(".sh") and f.endswith(".slurm")
                ):  # Not consider shell script change as a formal change
                    new_real_change = True
            if new_real_change:
                logtmp = (
                    "Warning! There is (at least) one uncommitted change in your code. "
                    "Recommended to git commit it first"
                )
                self.print(logtmp)
                time.sleep(3)
        else:
            CodeID = "GitNotFound"
            logtmp = "Warning! Git not found under this project. Highly recommended to use Git to manage code"
            self.print(logtmp)
        return CodeID

    def set_up_logtxt(self):
        self.logtxt = open(self.logtxt_path, "a+")
        # Globally redirect stderr and stdout. Overwriting the builtins.print fn may not be the best way to do so.
        if self.overwrite_print:
            sys.stderr = DoubleWriter(sys.stderr, self.logtxt)
            self.original_print = builtins.print  # Keep a copy of the original print fn
            builtins.print = self.print

    def print(
        self,
        *msg,
        sep=" ",
        end="\n",
        file=None,
        flush=False,
        callinfo=None,
        unprefix=False,
        acc=False,
        level=0,
        main_process_only=True,
        color=None,
    ):
        """Replace the standard print func. Print to console and logtxt file."""
        if main_process_only:
            if self.global_rank not in [-1, 0]:
                return

        # Get the caller file name and line number
        if callinfo is None:
            result = traceback.extract_stack()
            caller = result[len(result) - 2]
            file_path_of_caller = (
                str(caller).split(",")[0].lstrip("<FrameSummary file ")
            )
            filename = os.path.relpath(file_path_of_caller)
            if f"{os.sep}site-packages{os.sep}" in filename:
                filename = filename.split(f"{os.sep}site-packages{os.sep}")[1]
            linenum = sys._getframe().f_back.f_lineno
            callinfo = f"{filename}:{linenum}"

        # Get the level info: https://docs.python.org/3/library/logging.html#levels
        level = str(level).lower()
        assert level in [
            "0",
            "10",
            "20",
            "30",
            "40",
            "50",
            "notset",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        ]
        if level in ["0", "notset"]:
            info = ""
        elif level in ["10", "debug"]:
            info = yellow("[Debug] ")
        elif level in ["20", "info"]:
            info = ""
        elif level in ["30", "warning"]:
            info = yellow("[Warning] ")
        elif level in ["40", "error"]:
            info = red("[Error] ")
        elif level in ["50", "critical"]:
            info = red("[Critical] ")

        # Get the final message to print
        msg = sep.join([str(m) for m in msg]) + end
        if acc:
            msg = (
                "  " * int(self.ExpID[-1]) + msg
            )  # Add blanks to acc lines for easier identification
        msg_gray_scale = copy.deepcopy(msg)  #* @graenys

        if not unprefix:
            now = datetime.now(timezone).strftime("%m/%d/%y - %H:%M:%S")
            # prefix = "[%s %s %s] %s" % (self.ExpID[-6:], os.getpid(), now, info)
            prefix = "[%s - %s] %s" % (self.expname, now, info)  #* @graenys
            if self.debug:
                prefix = "[%s - %s] [%s] %s" % (
                    self.expname,
                    # os.getpid(),
                    now,
                    callinfo,
                    info,
                )
            original_prefix = copy.deepcopy(prefix)
            prefix = green(prefix)  # Render prefix with blue color

            # Render msg with colors
            # color = None #* @graenys
            if color is not None:
                if color in ["green", "g"]:
                    msg = green(msg)
                elif color in ["red", "r"]:
                    msg = red(msg)
                elif color in ["blue", "b"]:
                    msg = blue(msg)
                elif color in ["yellow", "y"]:
                    msg = yellow(msg)
                else:
                    raise NotImplementedError
            msg_gray_scale = original_prefix + msg  #* @graenys 
            msg = prefix + msg
        else:  #* @graenys
            if color is not None:
                if color in ["green", "g"]:
                    msg = green(msg)
                elif color in ["red", "r"]:
                    msg = red(msg)
                elif color in ["blue", "b"]:
                    msg = blue(msg)
                elif color in ["yellow", "y"]:
                    msg = yellow(msg)
                else:
                    raise NotImplementedError


        # Print
        flush = True
        if file is None:
            self.logtxt.write(msg_gray_scale)
            sys.stdout.write(msg)
            if flush:
                self.logtxt.flush()
                sys.stdout.flush()
        else:
            file.write(msg)
            if flush:
                file.flush()

    def _set_up_py_logging(self):
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.logtxt_path)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        self._fmt_unprefix = logging.Formatter(fmt="%(message)s")
        fmt = logging.Formatter(
            fmt=f"[{self.ExpID[-6:]} p%(process)d %(asctime)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s",
            datefmt="%Y/%m/%d-%H:%M:%S",
        )
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

    def info(
        self,
        *msg,
        sep=" ",
        end="\n",
        file=None,
        flush=False,
        unprefix=False,
        acc=False,
        main_process_only=True,
        color=None,
    ):
        """Reload logger.info in python logging"""
        result = traceback.extract_stack()
        caller = result[len(result) - 2]
        file_path_of_caller = str(caller).split(",")[0].lstrip("<FrameSummary file ")
        filename = os.path.relpath(file_path_of_caller)
        if f"{os.sep}site-packages{os.sep}" in filename:
            filename = filename.split(f"{os.sep}site-packages{os.sep}")[1]
        linenum = sys._getframe().f_back.f_lineno
        callinfo = f"{filename}:{linenum}"

        self.print(
            *msg,
            sep=sep,
            end=end,
            file=file,
            flush=flush,
            callinfo=callinfo,
            unprefix=unprefix,
            acc=acc,
            level="info",
            main_process_only=main_process_only,
            color=color,
        )

    def warn(
        self,
        *msg,
        sep=" ",
        end="\n",
        file=None,
        flush=False,
        unprefix=False,
        acc=False,
        main_process_only=True,
        color=None,
    ):
        """Reload logger.warn in python logging"""
        self.print(
            *msg,
            sep=sep,
            end=end,
            file=file,
            flush=flush,
            unprefix=unprefix,
            acc=acc,
            level="warning",
            main_process_only=main_process_only,
            color="yellow",
        )

    #* Never used
    def print_v2(
        self,
        *value,
        sep=" ",
        end="\n",
        file=None,
        flush=False,
        unprefix=False,
        acc=False,
        level="info",
    ):
        """Replace the standard print func. Print to console and logtxt file."""
        msg = sep.join([str(v) for v in value]) + end
        if acc:
            msg = "  " * int(self.ExpID[-1]) + msg

        if unprefix:
            pass

        if file is not None:
            file.write(msg)
            if flush:
                file.flush()
        else:
            level = str(level).lower()
            assert level in [
                "0",
                "10",
                "20",
                "30",
                "40",
                "50",
                "notset",
                "debug",
                "info",
                "warning",
                "error",
                "critical",
            ]
            # See https://docs.python.org/3/library/logging.html#levels
            if level in ["10", "debug"]:
                self._logger.debug(msg)
            elif level in ["20", "info"]:
                self._logger.info(msg)
            elif level in ["30", "warning"]:
                self._logger.warning(msg)
            elif level in ["40", "error"]:
                self._logger.error(msg)
            elif level in ["50", "critical"]:
                self._logger.critical(msg)

    def get_userip(self):
        user = getpass.getuser()

        # Get IP address. Refer to: https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()

        userip = f"{user}@{ip}"
        hostname = socket.gethostname()
        return userip, hostname

    def plot(self, name, out_path):
        self.log_tracker.plot(name, out_path)

    def cache_code(self):
        """Back up the code. Do not back up in debug mode, or, explicitly using --no_cache."""
        if self.args.debug or self.args.no_cache:
            return

        cache_script = self.args.cache_code
        if os.path.exists(cache_script):
            t0 = time.time()
            logtmp = f"==> Caching code to '{yellow(self._cache_path)}' using the provided script {yellow(cache_script)}"
            self.print(logtmp)
            cmd = f"sh {cache_script} {self._cache_path}"
            os.system(cmd)
            logtmp = f"==> Caching done (time: {time.time() - t0:.2f}s)"
            self.print(logtmp)

    def get_project_name(self):
        """For example, 'Projects/FasterRCNN/logger.py', then return 'FasterRCNN'"""
        file_path = os.path.abspath(__file__)
        return file_path.split("/")[-2]

    def timenow(self):
        return datetime.now(timezone).strftime("%Y/%m/%d-%H:%M:%S")
