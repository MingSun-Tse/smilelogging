# smilelogging
Python logging package for easy reproducible experimenting in research.


## Why you may need this package
This project is meant to provide an easy-to-use (as easy as possible) package to enable *reproducible* experimenting in research. Here is an awkward situation you may also encountered:
> I am doing some project. I got a fatanstic idea some time (one week, one month, or even one year) ago. Now I am looking at the results of that experiment, but I just cannot reproduce them anymore. I cannot remember which script and what hyper-prarameters I used. Even worse, since then I've modified the code (a lot). I don't know where I messed it up...

If you do not use this package, usually, what you can do may be:
- First, use Github to manage your code. Always run experiments after `git commit`. 
- Second, before each experiment, set up a *unique* experiment folder (with a unique ID to label that experiment -- we call it `ExpID`). 
- Third, when running an experiment, print your git commit ID (we call it `CodeID`) and `arguments` in the log.

Every result is uniquely binded with an `ExpID`, corresponding to a unique experiment folder. In that folder, `CodeID` and `arguments` are saved. So ideally, as long as we know the `ExpID`, we should be able to rerun the experiment under the same condition.

These steps are pretty simple, but if you write them over and over again in each project, it can still be quite annoying. This package is meant to **save you with basically 3~4 lines of code change**.


## Usage

**Step 0: Install the package (>= python3.4)**
```
# --upgrade to make sure you install the latest version
pip install smilelogging --upgrade
```

**Step 1: Modify your code**

Here we use the official [PyTorch ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) to give an example.

```
# add this at the head of code
from smilelogging import Logger 

# replace argument parser
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  
==> 
from smilelogging import argparser as parser

# add logger using this pacakge
args = parser.parse_args()
==> 
args = parser.parse_args()
logger = Logger(args)
global print; print = logger.log_printer.logprint # change print function so that logs can be printed to a txt file
```
> TIPS: overwriting the default python print func may not be a good practice, a better way may be `logprint = logger.log_printer.logprint`, and use it like `logprint('Test accuracy: %.4f' % test_acc)`. This will print the log to a txt file at path `log/log.txt`.

**Step 2: Run experiments**

The original ImageNet training snippet is:
```
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet18 [imagenet-folder with train and val folders]
```

Now, try this:
```
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet18 [imagenet-folder with train and val folders] --project_name Scratch__resnet18__imagenet --screen_print
```
> This snippet will set up an experiment folder under path `Experiments/Scratch__resnet18__imagenet_XXX`. That `XXX` thing is an `ExpID` automatically assigned by the time running this snippet. Below is an example on my PC:
```
Experiments/
└── Scratch__resnet18__imagenet_SERVER138-20211021-145936
    ├── gen_img
    ├── log
    │   ├── git_status.txt
    │   ├── gpu_info.txt
    │   ├── log.txt
    │   ├── params.yaml
    │   └── plot
    └── weights
```
<h4 align="center">:sparkles: Congrats:exclamation: You're (almost) all set!</h4>


As seen, there will be 3 folders automatically created: `gen_img`, `weights`, `log`. Log text will be saved in `log/log.txt`, arguments saved in `log/params.yaml` and in the head of `log/log.txt`. Below is an example of the first few lines of `log/log.txt`:
``` 
cd /home/wanghuan/Projects/TestProject
CUDA_VISIBLE_DEVICES=1 python main.py -a resnet18 /home/wanghuan/Dataset/ILSVRC/Data/CLS-LOC/ --project Scracth_resnet18_imagenet --screen_print

('arch': resnet18) ('batch_size': 256) ('cache_ignore': ) ('CodeID': f30e6078) ('data': /home/wanghuan/Dataset/ILSVRC/Data/CLS-LOC/) ('debug': False) ('dist_backend': nccl) ('dist_url': tcp://224.66.41.62:23456) ('epochs': 90) ('evaluate': False) ('gpu': None) ('lr': 0.1) ('momentum': 0.9) ('multiprocessing_distributed': False) ('note': ) ('pretrained': False) ('print_freq': 10) ('project_name': Scracth_resnet18_imagenet) ('rank': -1) ('resume': ) ('screen_print': True) ('seed': None) ('start_epoch': 0) ('weight_decay': 0.0001) ('workers': 4) ('world_size': -1)

[180853 22509 2021/10/21-18:08:54] ==> Caching various config files to 'Experiments/Scracth_resnet18_imagenet_SERVER138-20211021-180853/.caches'
```
Note, it tells us 
- (1) where is the code
- (2) what snippet is used when running this experiment
- (3) what arguments are used
- (4) what is the CodeID -- useful when rolling back to prior code versions with `git reset --hard <CodeID>`,
- (5) where the code files (*.py, *.json, *.yaml etc) are backuped -- note the log line "==> Caching various config files to ...". Ideally, CodeID is already enough to get previous code. Caching code files is a double insurance.
- (6) At the begining of each log line, the prefix "[180853 22509 2021/10/21-18:08:54]" is automatically added if the `logprint` func is used for print, where `180853` is short for the full ExpID `SERVER138-20211021-180853`, `22509` is the program pid (useful if you want to kill the job, e.g., `kill -9 22509`).


**More explanantions about the folder setting**

The `weights` folder is supposed to store the checkpoints during training; and `gen_img` is supposed to store the generated images during training (like in a generative model project). To use them in the code:
```
weights_path = logger.weights_path
gen_img_path = logger.gen_img_path
```


**More explanantions about the arguments and more tips**
- `--screen_print` means the logs will also be print to the console (namely, your screen). If it is not used, the log will only be saved to `log/log.txt`, not printed to screen. 
- If you are debugging code, you may not want to create an experiment folder under `Experiments`. Then use `--debug`, for example:
```
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet18 [imagenet-folder with train and val folders] --debug
```
This will save all the logs in `Debug_Dir`, instead of `Experiments` (`Experiments` is expected to store the *formal* experiment results).


## Collaboration / Suggestions
Currently, this is still a baby project. Any collaboration or suggestions are welcome to Huan Wang (Email: `wang.huan@northeastern.edu`).


## TODO
- Add training and testing metric (like accuracy, PSNR) plots.