import argparse
from fnmatch import fnmatch
import glob
import os, sys
import numpy as np
from .slutils import get_exp_name_id

parser = argparse.ArgumentParser()
parser.add_argument('--kw', type=str, required=True, help='keyword for filtering expriment folders')
parser.add_argument('--exact_kw', action='store_true', help='if true, not filter by exp_name but exactly the kw')
parser.add_argument('--metricline_mark', type=str, default='')
parser.add_argument('--metric', type=str, default='Acc1')
parser.add_argument('--lastline_mark', type=str, default='last') # 'Epoch 240' or 'Step 11200', which is used to pin down the line that prints the best accuracy
parser.add_argument('--remove_outlier', action='store_true')
parser.add_argument('--outlier_thresh', type=float, default=0.5, help='if |value - mean| > outlier_thresh, we take this value as an outlier')
parser.add_argument('--ignore', type=str, default='', help='seperated by comma')
parser.add_argument('--exps_folder', type=str, default='Experiments')
parser.add_argument('--n_decimals', type=int, default=4)
parser.add_argument('--scale', type=float, default=1.)

parser.add_argument('--acc_analysis', action='store_true')
parser.add_argument('--corr_analysis', action='store_true')
parser.add_argument('--corr_stats', type=str, default='spearman', choices=['pearson', 'spearman', 'kendall'])
parser.add_argument('--out_plot_path', type=str, default='plot.jpg')
args = parser.parse_args()

# 1st filtering: get all the exps with the keyword
all_exps_ = [x for x in glob.glob(f'{args.exps_folder}/{args.kw}') if os.path.isdir(x) and '_SERVER' in x]
if len(all_exps_) == 0:
    print(f'!! [Warning] Found NO experiments with the given keyword. Please check')

# 2nd filtering: remove all exps in args.ignore
if args.ignore:
    ignores = args.ignore.split(',')
    all_exps_ = [e for e in all_exps_ if True not in [fnmatch(e, i) for i in ignores]]

# 3rd filtering: add all the exps with the same name, even it is not included by the 1st filtering by kw
if args.exact_kw:
    all_exps = all_exps_
    exp_groups = [] # Get group exps, because each group is made up of multiple times
    for exp in all_exps:
        _, _, exp_name, _ = get_exp_name_id(exp)
        if exp_name not in exp_groups:
            exp_groups.append(exp_name)
else:
    all_exps, exp_groups = [], []
    for exp in all_exps_:
        _, _, exp_name, _ = get_exp_name_id(exp)
        if exp_name not in exp_groups:
            exp_groups.append(exp_name)
        all_exps_with_the_same_name = glob.glob(f'{args.exps_folder}/{exp_name}_SERVER*')
        for x in all_exps_with_the_same_name:
            if x not in all_exps:
                all_exps.append(x)
all_exps.sort()
exp_groups.sort()