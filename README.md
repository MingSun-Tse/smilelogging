# smilelogging
Python logging package for easy reproducible experimenting in research. Happily developed by the members of [SMILE Lab](https://web.northeastern.edu/smilelab/).


## Why this package may help you
This project is meant to provide an easy-to-use (as easy as possible) package to enable *reproducible* experimenting in research. Here is a struggling situation you may also encountered:
> I am doing some project. I got a fatanstic idea some time (one week, one month, or even one year) ago. Now I am looking at the results of that experiment, but I just cannot reproduce them anymore. I cannot remember which script and what hyper-prarameters I used. Even worse, since then I've modified the code (a lot). I don't know where I messed it up ...:cold_sweat:

Usually, what you can do may be:
- First, use Github to manage your code. Always run experiments after `git commit`. 
- Second, before each experiment, set up a *unique* experiment folder (with a unique ID to label that experiment -- we call it `ExpID`). 
- Third, when running an experiment, print your git commit ID (we call it `CodeID`) and `arguments` in the log.

Every result is uniquely binded with an `ExpID`, corresponding to a unique experiment folder. In that folder, `CodeID` and `arguments` are saved. So ideally, as long as we know the `ExpID`, we should be able to rerun the experiment under the same condition.

These steps are pretty simple, but if you implement them over and over again in each project, it can still be quite annoying:anger:. **This package is meant to save you with basically 3~4 lines of code change**:yum:.


## Usage

**Step 0: Install the package (>= python3.4)**
```python
pip install smilelogging

# next we will use PyTorch code as an example, so please also install pytorch and torchvision
pip install torch torchvision
```

**Step 1: Modify your code**

Here we use the [PyTorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist) to give a step-by-step example. In total, you only need to **add 3 lines of code and replace 1 line**.

```python
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from smilelogging import Logger # @mst: add this line

# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
from smilelogging import argparser as parser # @mst: replace above with this line

args = parser.parse_args() # @mst: add the following 2 lines
logger = Logger(args)
global print; print = logger.log_printer.logprint
```

We already put the modified code at `test_example/main.py`, so you do not need to edit any file now. Simply `cd test_example` and continue to next step.

**Step 2: Run experiments**

The original MNIST training snippet is:
```s
python main.py
```

Now, try this:
```s
python main.py --project_name lenet_mnist --screen_print
```
> This snippet will set up an experiment folder under path `Experiments/lenet_mnist_XXX`. That `XXX` thing is an `ExpID` automatically assigned by the time running this snippet. Below is an example on my PC:
```

```
<h4 align="center">:sparkles:Congrats:beers:You're all set:exclamation:</h4>


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
- (4) what is the CodeID -- useful when rolling back to prior code versions (`git reset --hard <CodeID>`)
- (5) where the code files (*.py, *.json, *.yaml etc) are backuped -- note the log line "==> Caching various config files to ...". Ideally, CodeID is already enough to get previous code. Caching code files is a double insurance
- (6) At the begining of each log line, the prefix "[180853 22509 2021/10/21-18:08:54]" is automatically added if the `logprint` func is used for print, where `180853` is short for the full ExpID `SERVER138-20211021-180853`, `22509` is the program pid (useful if you want to kill the job, e.g., `kill -9 22509`)


**More explanantions about the folder setting:**

The `weights` folder is supposed to store the checkpoints during training; and `gen_img` is supposed to store the generated images during training (like in a generative model project). To use them in the code:
```python
weights_path = logger.weights_path
gen_img_path = logger.gen_img_path
```


**More explanantions about the arguments and tips:**
- `--screen_print` means the logs will also be print to the console (namely, your screen). If it is not used, the log will only be saved to `log/log.txt`, not printed to screen. 
- If you are debugging code, you may not want to create an experiment folder under `Experiments`. Then use `--debug`, for example:
```
python main.py --debug
```
This will save all the logs in `Debug_Dir`, instead of `Experiments` (`Experiments` is expected to store the *formal* experiment results).
- In the above, we use `print = logger.log_printer.logprint`. Overwriting the default python print func may not be a good practice, a better way may be `logprint = logger.log_printer.logprint`, and use it like `logprint('Test accuracy: %.4f' % test_acc)`. This will print the log to a txt file at path `log/log.txt`.


## TODO
- [ ] Add training and testing metric (like accuracy, PSNR) plots.


## Collaboration / Suggestions
Currently, this is still a baby project. Any collaboration or suggestions are welcome to Huan Wang (Email: `wang.huan@northeastern.edu`).


