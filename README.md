# smilelogging
[![PyPI version](https://badge.fury.io/py/smilelogging.svg)](https://badge.fury.io/py/smilelogging)

Python logging package for easy reproducible experimenting in research. Developed by the members of [SMILE Lab](https://web.northeastern.edu/smilelab/).

_**[Update: 05/27/2023] 🔥🔥🔥Please install from source for the latest feature. The pip version is NOT the latest one and may have problems when using DDP.**_


## Why this package may help you
This project is meant to provide an easy-to-use (as easy as possible) package to enable *reproducible* experimenting in research. Here is a struggling situation you may also encountered:
> I am doing some project. I got a fatanstic idea some time (one week, one month, or even one year) ago. Now I am looking at the results of that experiment, but I just cannot reproduce them anymore. I cannot remember which script and what hyper-prarameters I used. Even worse, since then I've modified the code (a lot). I don't know where I messed it up ...:cold_sweat:

Usually, what you can do may be:
- First, use Github to manage your code. Always run experiments after `git commit`. 
- Second, before each experiment, set up a *unique* experiment folder (with a unique ID to label that experiment -- we call it `ExpID`). 
- Third, when running an experiment, print your git commit ID (we call it `CodeID`) and `arguments` in the log.

Every result is uniquely binded with an `ExpID`, corresponding to a unique experiment folder. In that folder, `CodeID`, `arguments`, and others (logs, checkpoints, etc.) are saved. So ideally, as long as we know the `ExpID`, we should be able to rerun the experiment under the same condition.

These steps are pretty simple, but if you implement them over and over again in each project, it can still be quite annoying. **This package is meant to save you with basically 2~3 lines of code change**.


## Usage

**Step 0: Install the package (>= python3.4)**
```console
pip install smilelogging

# next we will use PyTorch code as an example, so please also install PyTorch here
pip install torch torchvision

# clone this repo to continue
git clone https://github.com/MingSun-Tse/smilelogging.git
cd smilelogging
```

**Step 1: Modify your code**

Here we use the [PyTorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist) to give a step-by-step example. In total, you only need to **add 2 lines of code and replace 1 line**.

```python
from torch.optim.lr_scheduler import StepLR
from smilelogging import Logger # ==> add this line

# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
from smilelogging import argparser as parser # ==> replace above with this line

args = parser.parse_args()
logger = Logger(args) # ==> add this line
```

We already put the modified code at `test_example/main.py`, so you do not need to edit any file now. Simply `cd test_example` and continue to next step.

**Step 2: Run experiments**

The original MNIST training snippet is:
```console
python main.py
```

Now, try this:
```console
python main.py --project_name lenet_mnist
```
> This snippet will set up an experiment folder at path `Experiments/lenet_mnist_XXX`. That `XXX` thing is an `ExpID` automatically assigned by the time running this snippet. Below is an example on my PC:
```
Experiments/
└── lenet_mnist_SERVER138-20211022-184126
    ├── gen_img
    ├── log
    │   ├── git_status.txt
    │   ├── gpu_info.txt
    │   ├── log.txt
    │   ├── params.yaml
    │   └── plot
    └── weights
```
<h4 align="center">:sparkles:Congrats:beers:You're all set:exclamation:</h4>


As seen, there will be 3 folders automatically created: `gen_img`, `weights`, `log`. Log text will be saved in `log/log.txt`, arguments saved in `log/params.yaml` and in the head of `log/log.txt`. Below is an example of the first few lines of `log/log.txt`:
```console
cd /home/wanghuan/Projects/smilelogging/test_example
python main.py --project_name lenet_mnist

('batch_size': 64) ('cache_ignore': ) ('CodeID': 023534a) ('debug': False) ('dry_run': False) ('epochs': 14) ('gamma': 0.7) ('log_interval': 10) ('lr': 1.0) ('no_cuda': False) ('note': ) ('project_name': lenet_mnist) ('save_model': False) ('seed': 1) ('test_batch_size': 1000)

[184126 6424 2021/10/22-18:41:29] ==> Caching various config files to 'Experiments/lenet_mnist_SERVER138-20211022-184126/.caches'
```
Note, it tells us 
- (1) where is the code
- (2) what snippet is used when running this experiment
- (3) what arguments are used
- (4) what is the CodeID -- useful when rolling back to prior code versions (`git reset --hard <CodeID>`)
- (5) where the code files (*.py, *.json, *.yaml etc) are backuped -- note the log line `==> Caching various config files to ...`. Ideally, CodeID is already enough to get previous code. Caching code files is a double insurance
- (6) At the begining of each log line, the prefix `[184126 6424 2021/10/22-18:41:29]` is automatically added if the `logger.print` func is used for print, where `184126` is short for the full ExpID `SERVER138-20211022-184126`, `6424` is the program pid (useful if you want to kill the job, e.g., `kill -9 6424`)


## More Explanantions about the Folder Settings

The `weights` folder is supposed to store the checkpoints during training; and `gen_img` is supposed to store the generated images during training (like in a generative model project). To use them in the code:
```python
weights_path = logger.weights_path
gen_img_path = logger.gen_img_path
log_path = logger.log_path
```
For more these path names, see [here](https://github.com/MingSun-Tse/smilelogging/blob/59b874947238aabd4abd08c065eea499ffdbbdfa/smilelogging/logger.py#L285).

The folder names are pre-specified. If you do not like them and want to use your own folder setups, this code also provide such customization feature. Here is what you can do:

- Step 1: create a config txt file, such as `smilelogging_config.txt`. An example is below:
```txt
_experiments_dir: Other_Name_You_Like
_weights_dir: Other_Name_You_Like
_gen_img_dir: Other_Name_You_Like
_log_dir: Other_Name_You_Like
!reserve_dir: test/misc_results
```
where the `!reserve_dir` line is to indicate that you want to create a folder at path `test/misc_results` (under each experiment folder). The path of this folder will be assigned as an attribute of the `Logger` class, so you may use `logger.test__misc_results` (note `/` is replaced with `__`) to access it.

- Step 2: Add the following 2 lines (which are to enable a high-level feature of args) right after `args = parser.parse_args()`:
```
from smilelogging.utils import update_args
args = update_args(args)
```

- Step 3: when running experiments, append `--hacksmile.ON --hacksmile.config <path_to_config_in_Step_1>` to your script.

## More Explanantions about the Arguments and TIPs

- If you are debugging code, you may not want to create an experiment folder under `Experiments`. Then use `--debug`, for example:
```console
python main.py --debug
```
This will save all the logs in `Debug_Dir`, instead of `Experiments` (`Experiments` is expected to store the *formal* experiment results).


## TODO
- [ ] Add training and testing metric (like accuracy, PSNR) plots.

## Mission of this project
We target **100% open** scientific experimenting: 
- Every number or data point in the paper (either in tables or figures) is traceable with a log/checkpoint.
- Releasing the reviewing comments and communication process.

## Collaboration / Suggestions
Currently, this is still an alpha project. Any collaboration or suggestions are welcome to Huan Wang (Email: `wang.huan@northeastern.edu`).


