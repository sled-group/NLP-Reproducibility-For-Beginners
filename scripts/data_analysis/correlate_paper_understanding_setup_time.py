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
y_points2 = []
point_categories = []
point_categories2 = []
for group in PAPERS:
    for student in SUBJECT_DATA:
        if student.paper == group:
            setup_time_training = student.postsurvey["experiment_time_setup_training_hours"] * 60 + student.postsurvey["experiment_time_setup_training_minutes"]
            setup_time_evaluation = student.postsurvey["experiment_time_setup_evaluation_hours"] * 60 + student.postsurvey["experiment_time_setup_evaluation_minutes"]
            setup_mins = setup_time_training + setup_time_evaluation
            paper_understanding_accuracy = student.postsurvey['understanding_cq_accuracy']

            code_setup_rating = student.postsurvey['experiment_rating_setup']

            x_points.append(paper_understanding_accuracy)
            y_points.append(setup_mins)
            y_points2.append(code_setup_rating)
            point_categories.append(student.skill_level)
            point_categories2.append(student.paper)

corr, p = pearsonr(x_points, y_points)
print('Comprehension Q accuracy vs. setup time (%s) Pearson correlation: %.3f (p=%.5f)' % ('all', corr, p))

for group in SKILL_LEVELS:
    x_group = [x for x, g in zip(x_points, point_categories) if g == group]
    y_group = [y for y, g in zip(y_points, point_categories) if g == group]

    corr_aux_spearman, p_aux_spearman = spearmanr(x_group, y_group)
    print('Comprehension Q accuracy vs. setup time (%s) Spearman correlation: %.3f (p=%.5f)' % (group, corr_aux_spearman, p_aux_spearman))

    corr_aux_pearson, p_aux_pearson = pearsonr(x_group, y_group)
    print('Comprehension Q accuracy vs. setup time (%s) Pearson correlation: %.3f (p=%.5f)' % (group, corr_aux_pearson, p_aux_pearson))    

    slope, intercept, corr_aux_linreg, p_aux_linreg, slope_err = linregress(x_group, y_group)
    print('Comprehension Q accuracy vs. setup time (%s) linear regression: slope=%.5f (+- %.5f), intercept=%.5f, r=%.5f (p=%.5f)' % (group, slope, slope_err, intercept, corr_aux_linreg, p_aux_linreg))        

graph_data = GraphData(
    x=x_points,
    y=y_points,
    x_label='Paper Comprehension Accuracy (%)',
    y_label='Setup Time (min.)',
    x_range=(0.0, 1.05),
    y_range=(0, 1000),
    y_ticks=list(np.arange(0, 1001, 200)),
    colors=[SKILL_LEVEL_COLORS[group] for group in point_categories],
    point_categories=[SKILL_LEVEL_TITLES[group] for group in point_categories],
)
scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_time_by_skill_level.pdf"))    

graph_data = GraphData(
    x=x_points,
    y=y_points,
    x_label='Paper Comprehension Accuracy (%)',
    y_label='Setup Time (min.)',
    x_range=(0.0, 1.05),
    y_range=(0, 1000),
    y_ticks=list(np.arange(0, 1001, 200)),
    colors=[PAPER_COLORS[group] for group in point_categories2],
    point_categories=[PAPER_TITLES[group] for group in point_categories2],
)
scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_time_by_paper.pdf"))    


graph_data = GraphData(
    x=x_points,
    y=y_points2,
    x_label='Paper Comprehension Accuracy (%)',
    y_label='Setup Difficulty (1-5)',
    x_range=(0.0, 1.05),
    y_range=(0, 5.5),
    y_ticks=list(np.arange(0, 5.5, 1.0)),
    colors=[SKILL_LEVEL_COLORS[group] for group in point_categories],
    point_categories=[SKILL_LEVEL_TITLES[group] for group in point_categories],
)
scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_rating_by_skill_level.pdf"))    
