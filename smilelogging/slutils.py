import os, sys
import numpy as np
import socket

def get_exp_name_id(exp_path):
    r"""arg example: Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318/weights/ckpt.pth
    """
    seps = exp_path.split(os.sep)
    try:
        for s in seps:
            if '_SERVER' in s:
                exp_id = s.split('-')[-1]
                assert exp_id.isdigit()
                ExpID = 'SERVER' + s.split('_SERVER')[1]
                exp_name = s.split('_SERVER')[0]
                date = s.split('-')[-2]
    except:
        print(f'Failed to parse "{exp_path}", please check')
        exit(0)
    return ExpID, exp_id, exp_name, date

def standardize_metricline(line):
    r"""Make metric line in standard form.
    """
    for m in ['(', ')', '[', ']', '<', '>', '|', ',', '.', ';', '!', '?',]: # Some non-numerical, no-meaning marks
        if m in line:
            line = line.replace(m, f' {m} ')
    if ':' in line:
        line = line.replace(':', ' ')
    line = ' '.join(line.split())
    return line

def get_value(line, key, type_func=float):
    r"""Get the value of a <key> in <line> in a log txt.
    """
    # Preprocessing to deal with some unstandard log format
    line = standardize_metricline(line)
    # print(line)
        
    value = line.split(f' {key} ')[1].strip().split()[0]
    # print(value)
    
    if value.endswith('%'):
        value = type_func(value[:-1]) / 100.
    else:
        value = type_func(value)
    return value

def replace_value(line, key, new_value):
    line = standardize_metricline(line)
    value = line.split(key)[1].strip().split()[0]
    line = line.replace(f' {key} {value} ', f' {key} {new_value} ')
    return line

def get_project_name():
    cwd = os.getcwd()
    # assert '/Projects/' in cwd
    return cwd.split(os.sep)[-1]

# acc line example: Acc1 71.1200 Acc5 90.3800 Epoch 840 (after update) lr 5.0000000000000016e-05 (Best_Acc1 71.3500 @ Epoch 817)
# acc line example: Acc1 0.9195 @ Step 46600 (Best = 0.9208 @ Step 38200) lr 0.0001
# acc line example: ==> test acc = 0.7156 @ step 80000 (best = 0.7240 @ step 21300)
def is_metric_line(line, mark=''):
    r"""This function determines if a line is an accuracy line. Of course the accuracy line should meet some 
    format features which @mst used. So if these format features are changed, this func may not work.
    """
    line = standardize_metricline(line)
    if mark:
        return mark in line
    else:
        line = line.lower()
        return "acc" in line and "best" in line and '@' in line and 'lr' in line and 'resume' not in line and 'finetune' not in line


def parse_metric(line, metric, scale=1.):
    r"""Parse out the metric value of interest.
    """
    line = line.strip()
    # Get the last metric
    try:
        metric_l = get_value(line, metric)
    except:
        print(f'Parsing last metric failed; please check! The line is "{line}"')
        exit(1)

    # Get the best metric
    try:
        if f'Best {metric}' in line: # previous impl.
            metric_b = get_value(line, f'Best {metric}')
        elif f'Best_{metric}' in line:
            metric_b = get_value(line, f'Best_{metric}')
        elif f'Best{metric}' in line:
            metric_b = get_value(line, f'Best{metric}')
        else:
            metric_b = -1 # Not found the best metric value (not written in log)
    except:
        print(f'Parsing best metric failed; please check! The line is "{line}"')
        exit(1)
    return metric_l * scale, metric_b * scale


def parse_time(line):
    r"""Parse the time (e.g., epochs or steps) in a metric line.
    """
    line = standardize_metricline(line)
    if ' Epoch ' in line:
        time = get_value(line, 'Epoch', type_func=int)
    elif ' Step ' in line:
        time = get_value(line, 'Step', type_func=int)
    elif ' step ' in line:
        time = get_value(line, 'step', type_func=int)
    else:
        print(f'Fn "parse_time" failed. Please check')
        raise NotImplementedError
    return time


def parse_finish_time(log_f):
    lines = open(log_f, 'r').readlines()
    for k in range(1, min(1000, len(lines))):
        if 'predicted finish time' in lines[-k].lower():
            finish_time = lines[-k].split('time:')[1].split('(')[0].strip() # example: predicted finish time: 2020/10/25-08:21 (speed: 314.98s per timing)
            return finish_time

def get_ip():
    # Get IP address. Refer to: https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip