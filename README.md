# smilelogging
[![PyPI version](https://badge.fury.io/py/smilelogging.svg)](https://badge.fury.io/py/smilelogging)

Python logging package for easy reproducible experimenting in research. Developed by the members of [SMILE Lab](https://web.northeastern.edu/smilelab/).

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

**Step 0: Install the package**
```bash
# We will use PyTorch code as an example, so please also install PyTorch here
pip install torch torchvision

# Clone this repo and install from source (pypi may not be the lastest!)
git clone https://github.com/MingSun-Tse/smilelogging.git
cd smilelogging
pip install .
```

**Step 1: Modify your code**

Here we use the [PyTorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist) to give a step-by-step example. In total, you only need to **add 2 lines of code and replace 1 line**.

```python
from torch.optim.lr_scheduler import StepLR
from smilelogging import Logger  # ==> Add this line

# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
from smilelogging import argparser as parser  # ==> Replace above with this line

args = parser.parse_args()

# ==> Add this line. This will overwrite the system print function.
logger = Logger(args, overwrite_print=True)  

# ==> Or, if you do not want to overwrite the system print function, add this line. Then use `logger.info` to print.
logger = Logger(args)
```

We already put the modified code at `test_example/main.py`, so you do not need to edit any file now. Simply `cd test_example` and continue to next step.

**Step 2: Run experiments**

The original MNIST training snippet is:
```bash
python main.py
```

Now, try this:
```bash
python main.py --experiment_name lenet_mnist
```


## Configure the logger

Starting `v0.5`, we introduce a config file to customize the experiment folder structure. The configure file, a hidden YAML file named `.smilelogging_config.yaml`, is supposed to be located at the _root of each project_. When you run your code for the first time, the config file will be created _automatically_ for you. For example, in the above MNIST example, before running the code, there is only `main.py` in the `test_example` folder
<p align="center">
    <img src="images/before_running.png"  width="400px" >
</p>

After running the code, a hidden file `.smilelogging_config.yaml` will be created in the `test_example` folder.
<p align="center">
    <img src="images/after_running.png"  width="400px" >
</p>

You can customize the experiment folder strcuture by editing the config file. 


### Explanations about the config file
<details>
<summary>Default configs</summary>

```yaml
# Path of the folder to store ALL the experiments.
experiments_path: ./Experiments


#  Path of the folder to store experiments when debugging.
debug_path: ./Debug_Dir


# Folder structure of each experiment. 
# <experiment_folder> means this is a placeholder - Do NOT change it. You may customize the others not wrapped by <>.
experiment_folder_structure:
<experiment_folder>:
    weights:
    log: 
    log.txt
    system_info:
    .caches:


# Customize the format of experiment folder name.
experiment_folder:
format: "<experiment_name>--SERVER<nodeid>.<timeid>"
format_ddp: "<experiment_name>--SERVER<nodeid>.<timeid>.<rank>"  

# Customize the logging prefix format.
logging_prefix: 
format: "<expid> R<rank> <time>"
format_debug: "<expid> R<rank> <time> <callinfo>"
time_format: "%m%d %H:%M:%S"
color: blue


# Backup code.
cache_code:
is_open: True
script: ~/Projects/encode_lab_research_tools/experimenting_tools/cache_code.sh
```
</details>


<details>
<summary>Config explanation: experiment_folder_structure</summary>

The indented structure under the `experiment_folder_structure` describes the folder structure of each experiment. The name with a colon means it is a folder. The name without a colon means it is a file. The default folder and file structure is must-have, so do not change them. You can add more folders, e.g., you may want to have a folder to save the generated images for image generation tasks. Below, we add an `generated_images` folder to save the generated images:
```YAML
# Folder structure of each experiment. 
# <experiment_folder> means this is a placeholder - Do NOT change it. You may customize the others not wrapped by <>.
experiment_folder_structure:
  <experiment_folder>:
    weights:
    log: 
      log.txt
    system_info:
    .caches:
    generated_images:
```
Then, in your code, you can access the `generated_images` folder by using `logger.experiment_folder.generated_images.path`.
</details>





