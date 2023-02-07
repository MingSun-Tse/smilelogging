import time, os, sys, numpy as np, shutil as sh
import getpass
import logging
from .utils import get_project_path, parse_ExpID, mkdirs, run_shell_command, moving_average
from collections import OrderedDict
import socket
import yaml
import builtins
import traceback
from fnmatch import fnmatch

pjoin = os.path.join


class DoubleWriter():
    def __init__(self, f1, f2):
        self.f1, self.f2 = f1, f2

    def write(self, msg):
        self.f1.write(msg)
        self.f2.write(msg)

    def flush(self):
        self.f1.flush()
        self.f2.flush()


class LogTracker():
    r"""Logging all numerical results.
    """

    def __init__(self):
        self._metrics = OrderedDict()
        self._print_format = {}

    def update(self, k, v):
        r"""
        """
        if ':' in k:
            k, format_ = k.split(':')
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

    def format(self, selected=None, not_selected=None, sep=' '):
        r"""Format for print.
        """
        logstr = []
        for k, v in self._metrics.items():
            in_selected = True

            if selected is not None:
                in_selected = False
                for s in selected.split(','):
                    if fnmatch(k, s):
                        in_selected = True
                        break

            if not_selected is not None:
                for s in not_selected.split(','):
                    if fnmatch(k, s):
                        in_selected = False
                        break

            if in_selected:
                f = self._print_format[k] if k in self._print_format else f'%s'
                logstr += [f'{k} {f}' % v[-1]]
        return sep.join(logstr)

    def get_ma(self, k, window=10):
        r"""Moving average.
        """
        return moving_average(self._metrics[k], window)


class Logger(object):
    passer = {}

    def __init__(self, args):
        self.args = args
        self.sl_cfg = '.smilelogging.cfg'

        # logging folder names. Below are the default names, which can also be customized via 'args.hacksmile.config'
        self._experiments_dir = 'Experiments'
        self._debug_dir = 'Debug_Dir'
        self._weights_dir = 'weights'
        self._gen_img_dir = 'gen_img'
        self._log_dir = 'log'

        # customize logging folder names
        if hasattr(args, 'hacksmile') and args.hacksmile.config:
            for line in open(args.hacksmile.config):
                line = line.strip()
                if line.startswith('!reserve_dir'):
                    pass
                else:
                    attr, value = [x.strip() for x in line.split(':')]
                    self.__setattr__(attr, value)

        # Set up a unique experiment folder
        self.userip, self.hostname = self.get_userip()
        self.ExpID = self.get_ExpID()
        self.set_up_dir()
        self.set_up_cache_ignore()

        # Set up logging utils
        self.log_tracker = LogTracker()
        # self._set_up_py_logging() # TODO-@mst: Not finished, a lot of problems

        # Initial print
        self.print_script()
        args.CodeID = self.get_CodeID()  # get CodeID before 'print_args()' because it will be printed in log
        self.print_args()

        # Cache misc environment info  (e.g., GPU, git, etc.) and code files
        self.save_args()  # to .yaml
        self.save_nvidia_smi()  # print GPU info
        self.save_git_status()
        self.cache_done = False
        self.cache_model()  # backup code
        self.n_log_item = 0

    def get_CodeID(self):
        r"""Return git commit ID as CodeID
        """
        git_check = run_shell_command('git rev-parse --is-inside-work-tree')
        self.use_git = len(git_check) == 1 and git_check[0] == 'true'
        self.git_root = run_shell_command('git rev-parse --show-toplevel')[0]  # The path where .git is placed

        if self.use_git:
            CodeID = run_shell_command('git rev-parse --short HEAD')[0]
            changed_files = run_shell_command('git diff --name-only')
            new_real_change = False
            for f in changed_files:
                if f and not f.endswith('.sh') and f.endswith(
                        '.slurm'):  # Not consider shell script change as a formal change
                    new_real_change = True
            if new_real_change:
                logtmp = "Warning! There is (at least) one uncommitted change in your code. Recommended to git commit it first"
                self.print(logtmp)
                time.sleep(3)
        else:
            CodeID = 'GitNotFound'
            logtmp = "Warning! Git not found under this project. Highly recommended to use Git to manage code"
            self.print(logtmp)
        return CodeID

    def get_ExpID(self):
        r"""Assign a unique experiment id for the current experiment
        """
        self.SERVER = '%03d' % int(self.userip.split('.')[-1])

        if hasattr(self.args, 'resume_ExpID') and self.args.resume_ExpID:
            project_path = get_project_path(self.args.resume_ExpID)
            ExpID = parse_ExpID(project_path)
            TimeID = '-'.join(ExpID.split('-')[1:])
        else:
            TimeID = time.strftime("%Y%m%d-%H%M%S")
        ExpID = 'SERVER' + self.SERVER + '-' + TimeID
        return ExpID

    def set_up_dir(self):
        project_path = pjoin("%s/%s_%s" % (self._experiments_dir, self.args.project_name, self.ExpID))
        if hasattr(self.args, 'resume_ExpID') and self.args.resume_ExpID:
            project_path = get_project_path(self.args.resume_ExpID)
        if self.args.debug:  # debug has the highest priority. If debug, all the things will be saved in Debug_dir
            project_path = self._debug_dir

        # Output interface
        self.exp_path = project_path
        self.weights_path = pjoin(project_path, self._weights_dir)
        self.gen_img_path = pjoin(project_path, self._gen_img_dir)
        self.log_path = pjoin(project_path, self._log_dir)
        self.logplt_path = pjoin(self.log_path, "plot")
        self.logtxt_path = pjoin(self.log_path, "log.txt")
        self._cache_path = pjoin(project_path, ".caches")
        mkdirs(self.weights_path, self.gen_img_path, self.logplt_path, self._cache_path)
        self.logtxt = open(self.logtxt_path, "a+")

        # user can customize the folders in experiment dir
        if hasattr(self.args, 'hacksmile') and self.args.hacksmile.config:
            for line in open(self.args.hacksmile.config):
                line = line.strip()
                if line.startswith('!reserve_dir'):  # example: !reserve_dir: misc_results
                    assert len(line.split(':')) == 2, f"There should be only one ':' in the line. Please check: f{line}"
                    dir_name = line.split(':')[1].strip()
                    dir_path = f'{self.exp_path}/{dir_name}'
                    mkdirs(dir_path)
                    self.__setattr__(dir_name.replace('/', '__'), dir_path)

        # Globally redirect stderr and stdout. Overwriting the builtins.print fn may not be the best way to do so.
        sys.stderr = DoubleWriter(sys.stderr, self.logtxt)
        self.original_print = builtins.print # Keep a copy of the original print fn
        builtins.print = self.print

    def set_up_cache_ignore(self):
        ignore_default = ['__pycache__', 'Experiments', 'Debug_Dir', '.git']

        ignore_from_file = []
        if os.path.isfile('.cache_ignore'):
            for line in open('.cache_ignore'):
                ignore_from_file += line.strip().split(',')

        ignore_from_arg = []
        if hasattr(self.args, 'cache_ignore') and self.args.cache_ignore:
            ignore_from_arg += self.args.cache_ignore.split(',')

        ignore = ignore_default + ignore_from_file + ignore_from_arg
        ignore = list(set(ignore))  # Remove repeated items

        # If there is new cache_ignore, save it
        if set(ignore) != set(ignore_from_file):
            with open('.cache_ignore', 'w') as f:
                f.write(','.join(ignore))
        self.cache_ignore = ignore

    def print(self, *msg, sep=' ', end='\n', file=None, flush=False,
              unprefix=False, acc=False, level=0):
        r"""Replace the standard print func. Print to console and logtxt file.
        """
        # Get the caller file name and line number
        result = traceback.extract_stack()
        caller = result[len(result) - 2]
        file_path_of_caller = str(caller).split(',')[0].lstrip('<FrameSummary file ')
        filename = os.path.relpath(file_path_of_caller)
        if f'{os.sep}site-packages{os.sep}' in filename:
            filename = filename.split(f'{os.sep}site-packages{os.sep}')[1]
        lineno = sys._getframe().f_back.f_lineno

        # Get the level info
        level = str(level).lower()
        assert level in ['0', '10', '20', '30', '40', '50',
                         'notset', 'debug', 'info', 'warning', 'error', 'critical']
        # See https://docs.python.org/3/library/logging.html#levels
        if level in ['0', 'notset']:
            info = '\b'
        elif level in ['10', 'debug']:
            info = '[DEBUG]'
        elif level in ['20', 'info']:
            info = '[INFO]'
        elif level in ['30', 'warning']:
            info = '[WARNING]'
        elif level in ['40', 'error']:
            info = '[ERROR]'
        elif level in ['50', 'critical']:
            info = '[CRITICAL]'

        # Get the final message to print
        msg = sep.join([str(m) for m in msg]) + end
        if acc:
            msg = '  ' * int(self.ExpID[-1]) + msg  # Add blanks to acc lines for easier identification

        if not unprefix:
            prefix = "[%s %s %s] [%s:%d] %s " % (self.ExpID[-6:],
                                                 os.getpid(),
                                                 time.strftime("%Y/%m/%d-%H:%M:%S"),
                                                 filename,
                                                 lineno,
                                                 info)
            msg = prefix + msg

        # Print
        if file is None:
            self.logtxt.write(msg)
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
        self._fmt_unprefix = logging.Formatter(fmt='%(message)s')
        fmt = logging.Formatter(
            fmt=f'[{self.ExpID[-6:]} p%(process)d %(asctime)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
            datefmt='%Y/%m/%d-%H:%M:%S',
        )
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

    def print_v2(self, *value, sep=' ', end='\n', file=None, flush=False, unprefix=False, acc=False, level='info'):
        r"""Replace the standard print func. Print to console and logtxt file.
        """
        msg = sep.join([str(v) for v in value]) + end
        if acc:
            msg = '  ' * int(self.ExpID[-1]) + msg

        if unprefix:
            pass

        if file is not None:
            file.write(msg)
            if flush:
                file.flush()
        else:
            level = str(level).lower()
            assert level in ['0', '10', '20', '30', '40', '50',
                             'notset', 'debug', 'info', 'warning', 'error', 'critical']
            # See https://docs.python.org/3/library/logging.html#levels
            if level in ['10', 'debug']:
                self._logger.debug(msg)
            elif level in ['20', 'info']:
                self._logger.info(msg)
            elif level in ['30', 'warning']:
                self._logger.warning(msg)
            elif level in ['40', 'error']:
                self._logger.error(msg)
            elif level in ['50', 'critical']:
                self._logger.critical(msg)

    def get_userip(self):
        user = getpass.getuser()

        # Get IP address. Refer to: https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()

        userip = f'{user}@{ip}'
        hostname = socket.gethostname()
        return userip, hostname

    def print_script(self):
        script = f'hostname: {self.hostname}  userip: {self.userip}\n'
        script += 'cd %s\n' % os.path.abspath(os.getcwd())
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
            script += ' '.join(['CUDA_VISIBLE_DEVICES=%s python' % gpu_id, *sys.argv])
        else:
            script += ' '.join(['python', *sys.argv])
        script += '\n'
        self.print(script, unprefix=True)

    def print_args(self):
        '''Example: ('batch_size', 16) ('CodeID', 12defsd2) ('decoder', models/small16x_ae_base/d5_base.pth)
            It will sort the arg keys in alphabeta order, case insensitive.'''
        # build a key map for later sorting
        key_map = {}
        for k in self.args.__dict__:
            k_lower = k.lower()
            if k_lower in key_map:
                key_map[k_lower + '_' + k_lower] = k
            else:
                key_map[k_lower] = k

        # print in the order of sorted lower keys 
        logtmp = ''
        for k_ in sorted(key_map.keys()):
            real_key = key_map[k_]
            logtmp += "('%s': %s) " % (real_key, self.args.__dict__[real_key])
        self.print(logtmp + '\n', unprefix=True)

    def save_nvidia_smi(self):
        out = pjoin(self.log_path, 'gpu_info.txt')
        script = 'nvidia-smi >> %s' % out
        os.system(script)

    def save_git_status(self):
        if self.use_git:
            script = 'git status >> %s' % pjoin(self.log_path, 'git_status.txt')
            os.system(script)

    def plot(self, name, out_path):
        self.log_tracker.plot(name, out_path)

    def cache_model(self):
        r"""Save the modle architecture, loss, configs, in case of future check.
        """
        if self.args.debug or self.cache_done: return

        t0 = time.time()
        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path, exist_ok=True)
        logtmp = f"==> Caching various config files to '{self._cache_path}'"
        self.print(logtmp)

        extensions = ['.py', '.json', '.yaml', '.sh', '.txt', '.md']  # files of these types will be cached

        def copy_folder(folder_path):
            for root, _, files in os.walk(folder_path):
                if '__pycache__' in root: continue
                for f in files:
                    _, ext = os.path.splitext(f)
                    if ext in extensions:
                        dir_path = pjoin(self._cache_path, root)
                        f_path = pjoin(root, f)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path, exist_ok=True)
                        if os.path.exists(f_path):
                            sh.copy(f_path, dir_path)

        # copy files in current dir
        [sh.copy(f, self._cache_path) for f in os.listdir('.') if
         os.path.isfile(f) and os.path.splitext(f)[1] in extensions]

        # copy dirs in current dir
        [copy_folder(d) for d in os.listdir('.') if os.path.isdir(d) and d not in self.cache_ignore]
        logtmp = f'==> Caching done (time: {time.time() - t0:.2f}s)'
        self.print(logtmp)
        self.cache_done = True

    def get_project_name(self):
        ''' For example, 'Projects/FasterRCNN/logger.py', then return 'FasterRCNN' '''
        file_path = os.path.abspath(__file__)
        return file_path.split('/')[-2]

    def save_args(self):
        with open(pjoin(self.log_path, 'args.yaml'), 'w') as f:
            yaml.dump(self.args.__dict__, f, indent=4)

    def netprint(self, net, comment=''):
        r"""Deprecated. Will be removed.
        """
        with open(pjoin(self.log_path, 'model_arch.txt'), 'w') as f:
            if comment:
                print('%s:' % comment, file=f)
            print('%s\n' % str(net), file=f, flush=True)
