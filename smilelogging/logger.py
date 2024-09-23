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
import shutil
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import yaml

from smilelogging.slutils import bold, blue, green, red, yellow

pjoin = os.path.join
TIMEZONE = datetime.now().astimezone().tzinfo  # Use local time zone.


class Folder:
    def __init__(self, path: str):
        self.path = path
        self._subfolders = {}

    def __repr__(self):
        return f"Folder({self.path})"

    def add_subfolder(self, key, subfolder):
        self._subfolders[key] = subfolder

    def __getattr__(self, name):
        if name in self._subfolders:
            return self._subfolders[name]
        raise AttributeError(f"'{self.name}' object has no attribute '{name}'")


def print_tree(directory, indent="", is_root=True):
    """Print a directory in a tree-like fashion."""
    # Print the root directory name if it's the root call
    if is_root:
        print(directory)

    # Get list of all files and directories in the given directory
    items = os.listdir(directory)

    # Iterate over each item
    for index, item in enumerate(items):
        # Create the full path for the item
        item_path = os.path.join(directory, item)

        # Check if it's the last item in the list for formatting
        is_last = index == len(items) - 1

        # Print the current item with appropriate indentation
        print(indent + ("└── " if is_last else "├── ") + item)

        # If it's a directory, recurse into it
        if os.path.isdir(item_path):
            print_tree(
                item_path, indent + ("    " if is_last else "│   "), is_root=False
            )


def _pretty_dict_format(a_dict: dict):
    # Build a key map for later sorting.
    key_map = {}
    for k in a_dict:
        k_lower = k.lower()
        if k_lower in key_map:
            key_map[k_lower + "_" + k_lower] = k
        else:
            key_map[k_lower] = k

    # Print in the order of sorted lower keys.
    logstr = ""
    for k_ in sorted(key_map.keys()):
        real_key = key_map[k_]
        logstr += "('%s': %s) " % (real_key, a_dict[real_key])
    return logstr


def create_folder_structure(
    experiments_path: str,
    experiment_folder_name: str,
    experiment_folder_structure: dict,
):
    uniform_experiment_name = "experiment"
    expfolder = Folder(path=experiments_path)

    def _create_structure(base_folder, structure):
        base_folder_path = base_folder.path
        for subfolder, content in structure.items():
            key = subfolder
            if subfolder.startswith("<") and subfolder.endswith(
                ">"
            ):  # This indicates the name should be replaced. E.g., <experiment_folder_name>
                subfolder = experiment_folder_name
                key = uniform_experiment_name

            # Create the full path for the current item
            subfolder_path = os.path.join(base_folder_path, subfolder)
            subfolder = Folder(path=subfolder_path)
            os.makedirs(
                subfolder_path, mode=0o777, exist_ok=True
            )  # Use mode 777 for docker. Otherwise, the folder will belong to root.
            base_folder.add_subfolder(key, subfolder)

            # Process the content.
            if content is not None:
                if isinstance(content, dict):
                    # Recursively create sub-structure
                    _create_structure(subfolder, content)
                else:
                    content = content.split()  # The content may be multiple files.
                    for each_content in content:
                        each_content = each_content.strip()
                        with open(pjoin(subfolder_path, each_content), "w") as f:
                            f.write("")

    # Start creating from the root of the structure
    _create_structure(expfolder, experiment_folder_structure)
    return expfolder


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


def check_command_installed(cmd: str):
    """Check whether a system command, indicated by `cmd`, is installed."""
    try:
        result = subprocess.run(
            [f"{cmd}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


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


class Logger(object):
    passer = {}

    def __init__(self, args, overwrite_print=False):
        self.args = args
        self.overwrite_print = overwrite_print
        self.debug = self.args.debug or "debug" in self.args.experiment_name.lower()
        self._logtxt_name = "log.txt"

        # Load smilelogging config.
        smileconfig_file = ".smilelogging_config.yaml"
        if not os.path.exists(smileconfig_file):
            smileconfig_template = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), smileconfig_file
            )
            shutil.copyfile(smileconfig_template, smileconfig_file)
        with open(smileconfig_file, "r") as f:
            smileconfig_dict = yaml.safe_load(f)
        smileconfig = SimpleNamespace(**smileconfig_dict)  # Convert dict to attributes.
        self.smileconfig = smileconfig
        self.logging_prefix = smileconfig.logging_prefix
        self.timeid_format = smileconfig.experiment_folder_name["timeid_format"]

        # Some default folder names.
        self._experiments_path = smileconfig_dict.get("experiments_path", "Experiments")
        self._debug_path = smileconfig_dict.get("debug_path", "Debug_Dir")

        # Handle the DDP case.
        self.global_rank, self.local_rank = self._figure_out_rank()
        self._ddp = self.global_rank >= 0

        # Get user and ip info.
        self.userip, self.hostname = self.get_userip()

        # Get various id's.
        self.timeid = self._get_timeid()
        self.expid = self.timeid[-6:]
        self.pid = str(os.getpid())
        self.codeid = self._get_codeid()
        self.nodeid = "%03d" % int(self.userip.split(".")[-1])

        # Get experiment folder name.
        self._experiment_folder_name = self._get_experiment_folder_name()

        # Create the folder structure. This can be done only after the ExpID is created.
        experiments_folder = create_folder_structure(
            experiments_path=self._experiments_path,
            experiment_folder_name=self._experiment_folder_name,
            experiment_folder_structure=smileconfig.experiment_folder_structure,
        )
        print("Creating experiment folder is done. Tree structure:")
        print_tree(experiments_folder.experiment.path)
        self.experiment_folder = experiments_folder.experiment
        self._check_folder_structure()
        self.log_path = self.experiment_folder.log.path
        self.logtxt_path = pjoin(self.experiment_folder.log.path, self._logtxt_name)
        self.weights_path = self.experiment_folder.weights.path
        self._cache_path = pjoin(self.experiment_folder.path, ".caches")
        self.set_up_logtxt()

        # Set up logging prefix color.
        self.logging_prefix_color_fn = self._set_up_logging_prefix_color()

        # Handle the DDP case.
        self._figure_out_rank()
        self._ddp = self.global_rank >= 0

        # Set up logging utils.
        self.log_tracker = LogTracker()

        # Initial print.
        self.print_script()
        self.print_args()

        # Cache misc environment info  (e.g., args, GPU, git, etc.) and codes.
        self.save_env_snapshot()
        if self.global_rank in [-1, 0]:  # Only the main process does caching.
            self._backup_code()

    def _set_up_logging_prefix_color(self):
        """Set up log prefix color function."""
        logging_prefix_color = self.logging_prefix["color"].lower()
        if logging_prefix_color == "none":
            return None
        else:
            assert self.logging_prefix["color"] in [
                "blue",
                "green",
                "yellow",
                "red",
            ], "Invalid `color` option in `logging_prefix`. Please check!"
            return eval(logging_prefix_color)

    def _check_folder_structure(self):
        logtxt_path = pjoin(self.experiment_folder.log.path, self._logtxt_name)
        assert os.path.exists(
            logtxt_path
        ), "File `{}` not exist under {}. Please check!".format(
            self._logtxt_name, self.experiment_folder.log.path
        )
        assert os.path.exists(
            self.experiment_folder.weights.path
        ), "Folder `weights` not exist under {}. Please check!".format(
            self.experiment_folder.path
        )
        assert os.path.exists(
            self.experiment_folder.log.path
        ), "Folder `log` not exist under {}. Please check!".format(
            self.experiment_folder.path
        )
        assert os.path.exists(
            pjoin(self.experiment_folder.path, ".caches")
        ), "Folder `.caches` not exist under {}. Please check!".format(
            self.experiment_folder.path
        )

    def _get_experiment_folder_name(self):
        """Get the experiment folder name based on the provided template in smilelogging config file."""
        folder_name = self.smileconfig.experiment_folder_name["format"]
        if self._ddp:
            folder_name = self.smileconfig.experiment_folder_name["format_ddp"]
        if "<timeid>" in folder_name:
            folder_name = folder_name.replace("<timeid>", self.timeid)
        if "<rank>" in folder_name:
            folder_name = folder_name.replace("<rank>", str(self.global_rank))
        if "<nodeid>" in folder_name:
            folder_name = folder_name.replace("<nodeid>", self.nodeid)
        if "<experiment_name>" in folder_name:
            folder_name = folder_name.replace(
                "<experiment_name>", self.args.experiment_name
            )
        return folder_name

    def _figure_out_rank(self):
        """Figure out the process rank when using multi-processing (e.g., DDP).

        Returns:
            gloabl_rank: Process rank in all processes (across multiple nodes).
            local_rank: Process rank in the processes on the current node.
        """
        global_rank = os.getenv("GLOBAL_RANK", -1)
        local_rank = os.getenv("LOCAL_RANK", -1)
        if local_rank != -1:  # DDP is used
            if global_rank == -1:  # single-node DDP
                global_rank = local_rank
        global_rank = int(global_rank)
        local_rank = int(local_rank)

        # Get ranks from args
        if hasattr(self.args, "global_rank") and self.args.global_rank >= 0:
            global_rank = self.args.global_rank
        if hasattr(self.args, "local_rank") and self.args.local_rank >= 0:
            local_rank = self.args.local_rank

        return global_rank, local_rank

    def _get_codeid(self) -> str:
        """Return git commit ID as codeid.

        Returns:
            codeid: A string that indicates the git commit ID.
        """
        git_check = run_shell_command("git rev-parse --is-inside-work-tree")
        self.use_git = len(git_check) == 1 and git_check[0] == "true"
        self.git_root = run_shell_command("git rev-parse --show-toplevel")[
            0
        ]  # The path where .git is placed

        if self.use_git:
            codeid = run_shell_command("git rev-parse --short HEAD")[0]
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
            codeid = "GitNotFound"
            logtmp = "Warning! Git not found under this project. Highly recommended to use Git to manage code"
            self.print(logtmp)
        return codeid

    def _get_timeid(self) -> str:
        """Get time stamp for the experiment folder name. The expid MUST be unique.

        Returns:
            timeid: A string that indicates the time stamp of the experiment.
        """
        timeid = datetime.now(TIMEZONE).strftime(self.timeid_format)
        expid = timeid[-6:]
        existing_exps = glob.glob(f"{self._experiments_path}/*{expid}*")
        t0 = time.time()
        # Make sure the expid is unique.
        while len(existing_exps) > 0:
            time.sleep(1)
            timeid = datetime.now(TIMEZONE).strftime(self.timeid_format)
            expid = timeid[-6:]
            existing_exps = glob.glob(f"{self._experiments_path}/*{expid}*")
            if time.time() - t0 > 120:
                self.print(
                    "Hanged for more than 2 mins when creating the experiment folder, "
                    "which is unusual. Please try again."
                )
                exit(1)
        return timeid

    def _get_experiment_path(self):
        """Get a unique directory for each experiment."""
        # Get rank for each process (used in multi-process training, e.g., DDP).
        if self.global_rank == -1:
            other_ranks_folder = ""
            rank = ""
        else:
            other_ranks_folder = "" if self.global_rank == 0 else "OtherRanks"
            rank = f"RANK{self.global_rank}-"

        experiment_path = ""

        if experiment_path != "":  # Use existing folder.
            self.ExpID = parse_ExpID(experiment_path)
        else:
            # Create a new folder.
            server = "SERVER%03d-" % int(self.userip.split(".")[-1])
            self.ExpID = rank + server + self._get_timeid()
            experiment_path = "%s/%s/%s_%s" % (
                self._experiments_path,
                other_ranks_folder,
                self.args.experiment_name,
                self.ExpID,
            )

        # --debug has the highest priority. If --debug, all the things will be saved in the debug folder.
        if self.args.debug:
            experiment_path = self._debug_path
        experiment_path = os.path.normpath(experiment_path)
        return experiment_path

    def set_up_logtxt(self):
        """Open the object of txt file and if necessary, overwrite the system print fn.

        (TODO: Overwriting the builtins.print fn may not be the best way to do so.)
        """
        self.logtxt = open(self.logtxt_path, "a+")
        if self.overwrite_print:
            sys.stderr = DoubleWriter(sys.stderr, self.logtxt)
            self.original_print = builtins.print  # Keep a copy of the original print fn
            builtins.print = self.print

    def _get_logging_prefix(self, callinfo: str):
        """Get the prefix for each line during logging."""
        prefix = self.logging_prefix["format"]
        if self.debug:
            prefix = self.logging_prefix["format_debug"]
        if "<time>" in prefix:
            now = datetime.now(TIMEZONE).strftime(self.logging_prefix["time_format"])
            prefix = prefix.replace("<time>", now)
        if "<pid>" in prefix:
            prefix = prefix.replace("<pid>", self.pid)
        if "<expid>" in prefix:
            prefix = prefix.replace("<expid>", self.expid)
        if "<callinfo>" in prefix:
            prefix = prefix.replace("<callinfo>", callinfo)
        prefix += " $ "

        # Add color.
        if self.logging_prefix_color_fn is not None:
            prefix = self.logging_prefix_color_fn(prefix)

        return prefix

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
            caller = result[-2]
            file_path_of_caller = caller.filename
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

        # Get the final message to print.
        msg = sep.join([str(m) for m in msg]) + end
        if acc:
            # Add blanks to accuracy lines for easier identification.
            msg = "  " * int(self.ExpID[-1]) + msg

        if not unprefix:
            prefix = self._get_logging_prefix(callinfo)

            # Render msg with colors.
            if color is not None:
                assert color in [
                    "green",
                    "red",
                    "blue",
                    "yellow",
                ], "Invalid `color` option. Please check!"
                color_fn = eval(color)
                msg = color_fn(msg)

            msg = prefix + msg

        # Print (TODO: Add flush?)
        if file is None:
            self.logtxt.write(msg)
            sys.stdout.write(msg)
            self.logtxt.flush()
            sys.stdout.flush()
        else:
            file.write(msg)
            file.flush()

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
        caller = result[-2]  # The second last file is the caller of logger.info.
        file_path_of_caller = caller.filename
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

    def print_script(self):
        """Print the script to the start of the log txt file."""
        hostname_userip = f"hostname: {self.hostname} - userip: {self.userip} - codeid: {self.codeid} - pid: {self.pid} - timezone: {TIMEZONE}"
        cd_cmd = "cd %s" % os.path.abspath(os.getcwd())
        executable = os.path.basename(sys.executable)
        script = executable + " " + " ".join(sys.argv)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
            script = f"CUDA_VISIBLE_DEVICES {gpu_id}" + script
        script = yellow("\n".join([hostname_userip, cd_cmd, script]))
        script += "\n"
        if self._ddp:
            script += yellow("Note: DDP (Data Distributed Parallel) is used.\n")
        self.print(script, unprefix=True)

    def print_args(self):
        """Example: ('batch_size', 16) ('decoder', models/small16x_ae_base/d5_base.pth).
        It will sort the arg keys in alphabeta order, case insensitive.
        """
        self.print(_pretty_dict_format(self.args.__dict__) + "\n", unprefix=True)

    def save_env_snapshot(self):
        """Save environment snapshot (such as args, gpu info, users, git info)."""
        # Save args.
        with open(pjoin(self.log_path, "args.yaml"), "w") as f:
            yaml.dump(self.args.__dict__, f, indent=4)

        # Save smilelogging config.
        with open(pjoin(self.log_path, "smilelogging_config.yaml"), "w") as f:
            yaml.dump(self.smileconfig.__dict__, f, indent=4)

        # Save system info.
        if check_command_installed("nvidia-smi"):
            os.system("nvidia-smi >> {}".format(pjoin(self.log_path, "nvidia-smi.txt")))
        if check_command_installed("gpustat"):
            os.system("gpustat >> {}".format(pjoin(self.log_path, "gpustat.txt")))
        if check_command_installed("who"):
            os.system("who -b >> {}".format(pjoin(self.log_path, "who.txt")))
            os.system("who >> {}".format(pjoin(self.log_path, "who.txt")))

        # Save git info.
        if self.use_git:
            os.system("git status >> {}".format(pjoin(self.log_path, "git_status.log")))

    def plot(self, name, out_path):
        self.log_tracker.plot(name, out_path)

    def _backup_code(self):
        """Back up the code. Do not back up in debug mode, or, explicitly indicated."""
        cache_code = self.smileconfig.cache_code
        if self.args.debug or not cache_code["is_open"]:
            return

        cache_script = os.path.expanduser(cache_code["script"])  # In case ~ is used.
        if os.path.exists(cache_script):
            t0 = time.time()
            logtmp = f"==> Caching code to {bold(self._cache_path)} using the provided script {bold(cache_script)}"
            self.print(logtmp)
            cmd = f"sh {cache_script} {self._cache_path} > /dev/null"
            os.system(cmd)
            logtmp = f"==> Caching done (time: {time.time() - t0:.2f}s)"
            self.print(logtmp)

    def print_model_arch(self, model):
        """Print model architecture to log."""
        with open(pjoin(self.log_path, "model_arch.txt"), "w") as f:
            print("%s\n" % str(model), file=f, flush=True)

    def timenow(self):
        return datetime.now(TIMEZONE).strftime("%Y/%m/%d-%H:%M:%S")
