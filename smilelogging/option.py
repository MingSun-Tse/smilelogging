import argparse
from fnmatch import fnmatch
import glob
import os
from .slutils import get_exp_name_id

parser = argparse.ArgumentParser()
parser.add_argument('--kw', type=str, required=True, help='keyword for filtering expriment folders')
parser.add_argument('--exact_kw', action='store_true', help='if true, not filter by expname but exactly the kw')
parser.add_argument('--metricline_mark', type=str, default='')
parser.add_argument('--metric', type=str, default='Acc1')
parser.add_argument('--lastline_mark', type=str,
                    default='last')  # 'Epoch 240' or 'Step 11200', which is used to pin down the line that prints the best accuracy
parser.add_argument('--remove_outlier', action='store_true')
parser.add_argument('--outlier_thresh', type=float, default=0.5,
                    help='if |value - mean| > outlier_thresh, we take this value as an outlier')
parser.add_argument('--ignore', type=str, default='', help='seperated by comma')
parser.add_argument('--exps_folder', type=str, default='Experiments')
parser.add_argument('--n_decimals', type=int, default=4)
parser.add_argument('--scale', type=float, default=1.)

parser.add_argument('--acc_analysis', action='store_true')
parser.add_argument('--corr_analysis', action='store_true')
parser.add_argument('--corr_stats', type=str, default='spearman', choices=['pearson', 'spearman', 'kendall'])
parser.add_argument('--out_plot_path', type=str, default='plot.jpg')
parser.add_argument('--tte', type=int, default=-1)
parser.add_argument('--hhe', type=int, default=-1)
parser.add_argument('--acc', action='store_true')
args = parser.parse_args()

if args.tte > 0:
    os.system(f'tail -n {args.tte} {args.exps_folder}/{args.kw}/log/log.txt')
    exit(0)
if args.hhe > 0:
    os.system(f'head -n {args.hhe} {args.exps_folder}/{args.kw}/log/log.txt')
    exit(0)
if args.acc:
    os.system(f'cat {args.exps_folder}/{args.kw}/log/log.txt | grep Acc1')
    exit(0)

# 1st filtering: get all the exps with the keyword
all_exps = [x for x in glob.glob(f'{args.exps_folder}/{args.kw}') if os.path.isdir(x) and 'SERVER' in x]
if len(all_exps) == 0:
    print(f'!! [Warning] Found NO experiments with the given keyword. Please check')

# 2nd filtering: remove all exps in args.ignore
if args.ignore:
    ignores = args.ignore.split(',')
    all_exps = [e for e in all_exps if True not in [fnmatch(e, i) for i in ignores]]

# 3rd filtering: add all the exps with the same name, even it is not included by the 1st filtering by kw
if args.exact_kw:
    exp_groups = {}  # Get group exps, because each group is made up of multiple times
    for exp in all_exps:
        _, _, expname, _ = get_exp_name_id(exp)
        if expname not in exp_groups:
            exp_groups.append(expname)
else:
    exp_groups = {}
    for exp in all_exps:
        _, _, expname, _ = get_exp_name_id(exp)
        if expname not in exp_groups:
            search_pattern = f'{args.exps_folder}/{expname}-SERVER*' if 'RANK' in expname \
                else f'{args.exps_folder}/{expname}_SERVER*'
            exps_with_same_expname = glob.glob(search_pattern)
            exps_with_same_expname.sort()
            exp_groups[expname] = exps_with_same_expname