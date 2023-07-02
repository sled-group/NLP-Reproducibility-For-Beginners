from ast import Str
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure
# colors = cm.rainbow(np.linspace(0, 1, len(ys)))
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import math
from scipy.stats import spearmanr, pearsonr, sem, linregress
import pickle
import os 
import statsmodels.api as sm
import sys
import json

sys.path.append('.')
from constants import SUBJECT_DATA, GOLD_RESULTS_DATA, \
                      PAPERS, PAPER_TITLES, PAPER_TITLES_SHORT, \
                      RESULT_IDS, RESULT_TITLES, RESULT_TITLES_CONDENSED, \
                      SKILL_LEVELS, SKILL_LEVEL_TITLES, SKILL_LEVEL_TITLES_SHORT, \
                      SKILL_LEVEL_COLORS, PAPER_COLORS, GRAPH_OUTPUT_DIR
from graph import GraphData, scatter_plot_2d                    


x_points = []
y_points = []
point_categories = []
for student in SUBJECT_DATA:    
    setup_time_training = student.postsurvey["experiment_time_setup_training_hours"] * 60 + student.postsurvey["experiment_time_setup_training_minutes"]
    setup_time_evaluation = student.postsurvey["experiment_time_setup_evaluation_hours"] * 60 + student.postsurvey["experiment_time_setup_evaluation_minutes"]
    gpu_mins = student.gpu_mins if student.gpu_mins is not None else 0

    if gpu_mins > 0:
        x_points.append(setup_time_training + setup_time_evaluation)
        y_points.append(gpu_mins)
        point_categories.append(SKILL_LEVEL_TITLES[student.skill_level])

print("%s records with GPU time" % len(x_points))

corr, p = pearsonr(x_points, y_points)
print('Setup Time vs. Runtime (%s) Pearson correlation: %.3f (p=%.5f)' % ('all', corr, p))

for group in SKILL_LEVELS:
    x_group = [x for x, g in zip(x_points, point_categories) if g == SKILL_LEVEL_TITLES[group]]
    y_group = [y for y, g in zip(y_points, point_categories) if g == SKILL_LEVEL_TITLES[group]]

    graph_data = GraphData(
        x=x_group,
        y=y_group,
        x_label='Setup Time (min.)',
        y_label='Runtime (min.)',
        x_range=(0, 1800),
        y_range=(0, 2500),
        x_ticks=list(np.arange(0, 1801, 300)),
        y_ticks=list(np.arange(0, 2501, 500)),
        colors=[SKILL_LEVEL_COLORS[group] for _ in x_group],
        add_fitline=True,

    )
    scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "setup_time_vs_runtime_%s.pdf" % group))    

    corr_aux_spearman, p_aux_spearman = spearmanr(x_group, y_group)
    print('Setup Time vs. Runtime (%s) Spearman correlation: %.3f (p=%.5f)' % (group, corr_aux_spearman, p_aux_spearman))

    corr_aux_pearson, p_aux_pearson = pearsonr(x_group, y_group)
    print('Setup Time vs. Runtime (%s) Pearson correlation: %.3f (p=%.5f)' % (group, corr_aux_pearson, p_aux_pearson))    

    slope, intercept, corr_aux_linreg, p_aux_linreg, slope_err = linregress(x_group, y_group)
    print('Setup Time vs. Runtime (%s) linear regression: slope=%.5f (+- %.5f), intercept=%.5f, r=%.5f (p=%.5f)' % (group, slope, slope_err, intercept, corr_aux_linreg, p_aux_linreg))        
