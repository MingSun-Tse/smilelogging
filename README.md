# researchlogging
Python logging package for easy reproducible experimenting in research.


## Why you may need this package
This project is meant to provide an easy-to-use (as easy as possible) package to enable *reproducible* experimenting in research.
By "reproducible", I mean such an awkward situation you may also encounter:
```
I am doing some project. I got a fatanstic idea some time (one week, one month, or even one year) ago. Now I am looking at the results of that experiment, but I just cannot reproduce them anymore. I cannot remember what hyper-prarameters I used. Even worse, since then I've modified the code (a lot). I don't know where I messed up...
```

If you do not use this package, usually, what you can do may be:
- First, use Github to manage your code. Always run experiments after `git commit`. 
- Second, before each experiment, set up a *unique* experiment folder (with a unique ID to label that experiment -- we call it `ExpID`). 
- Third, when running an experiment, print your git commit ID (we call it `CodeID`) and `arguments` in the log.

Every result is uniquely binded with an `ExpID`, corresponding to a unique experiment folder. In that folder, `CodeID` and `arguments` are saved. So ideally, as long as I know the ExpID, I should be able to rerun the experiment in exactly the same condition.

These steps are pretty simple, but if you write them over and over again on each project, it can still be quite annoying. This package is meant to save you from this with basically 3~4 lines of code change.


## Usage

**Step 0: Install the package**
```
pip install researchlogging --upgrade # --upgrade to make sure you install the latest version
```

**Step 1: Modify your code**
I use the official [PyTorch ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) to give an example.

```
# add this in your main function, somewhere proper.
from researchlogging import Logger 

# replace argument parser
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  
==> 
from researchlogging import argparser as parser

# add logger using this pacakge
args = parser.parse_args()
==> 
args = parser.parse_args()
logger = Logger(args)
```

**Step 1: Run experiment**
The original ImageNet training snippet is: 
```
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet18 [imagenet-folder with train and val folders]
```

Now, try this:
```
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet18 [imagenet-folder with train and val folders] --project_name Scratch__resnet18__imagenet --screen_print
```
> This snippet will set up an experiment folder under path `Experiments/Scratch__resnet18__imagenet_XXX`. That `XXX` is an ExpID automatically assigned by the time running this snippet. Here is an example on my PC:
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
As seen, there will be 3 folders automatically created: `gen_img`, `weights`, `log`. Log text will be saved in `log/log.txt`, arguments saved in `log/params.yaml` as well as the head of `log/log.txt`. Below is an example of the first few lines of a log.txt. It tells us what snippet is used when running this experiment; also, all the arguments are saved.
```
cd /home2/wanghuan/Projects/TestProject
CUDA_VISIBLE_DEVICES=1 python main.py -a resnet18 /home/wanghuan/Dataset/ILSVRC/Data/CLS-LOC/ --project_name Scratch__resnet18__imagenet --screen_print

('arch': resnet18) ('batch_size': 256) ('cache_ignore': ) ('CodeID': ) ('data': /home/wanghuan/Dataset/ILSVRC/Data/CLS-LOC/) ('debug': False) ('dist_backend': nccl) ('dist_url': tcp://224.66.41.62:23456) ('epochs': 90) ('evaluate': False) ('gpu': None) ('lr': 0.1) ('momentum': 0.9) ('multiprocessing_distributed': False) ('note': ) ('pretrained': False) ('print_freq': 10) ('project_name': Scratch__resnet18__imagenet) ('rank': -1) ('resume': ) ('screen_print': True) ('seed': None) ('start_epoch': 0) ('weight_decay': 0.0001) ('workers': 4) ('world_size': -1)

```

The `weights` folder is supposedly to store the checkpoints during training; and `gen_img` is supposedly to store the generated images during training (like in a generative model project). To use them in the code:
```
weights_path = logger.weights_path
gen_img_path = logger.gen_img_path
```



`--screen_print` means the logs will also be print to the console (namely, your screen). If it not used, the log will only be saved to 




## Collaboration / Suggestions
Currently, this is still a baby project. Any collaboration or suggestions are welcome to Huan Wang (Email: `wang.huan@northeastern.edu`).