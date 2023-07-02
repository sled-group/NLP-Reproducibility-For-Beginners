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

# Hypotheses to test wrt understanding the work:
#
# Understanding the "understanding" phase
# - Students' Python and PyTorch experience are connected to their understanding of the work
# -- no
# - Students' learning from the course is connected to their understanding of the work 
# -- kind of (seems homework experience improvement is negatively correlated for advanced students???)
#
# Connecting understanding to execution
# - Students' understanding is connected to their accuracy of reproducibility
# -- no
# - Students' understanding is connected to their experience in reproducing the results (either based on time or self report)
# -- no

x_points = []
y_points = []
point_categories = []
for student in SUBJECT_DATA:    
    python_pytorch_years = math.floor(student.presurvey['python_pytorch_years'])
    improved_pytorch_experience = student.presurvey['exp_pytorch_now'] - student.presurvey['exp_pytorch_before']
    improved_hw_experience = student.presurvey['hw3_score'] - student.presurvey['hw2_score']
    average_learning = (improved_pytorch_experience + improved_hw_experience)/2.
    
    paper_understanding_accuracy = student.postsurvey['understanding_cq_accuracy']
    
    result_errors = []
    for result_id in RESULT_IDS:
        if result_id in student.postsurvey:
            result = student.postsurvey[result_id]
            reported_result = GOLD_RESULTS_DATA[result_id]['reported']
            result_errors.append(result - reported_result)
    assert len(result_errors) > 0
    reproduced_accuracy = np.mean(result_errors)

    setup_time_training = student.postsurvey["experiment_time_setup_training_hours"] * 60 + student.postsurvey["experiment_time_setup_training_minutes"]
    setup_time_evaluation = student.postsurvey["experiment_time_setup_evaluation_hours"] * 60 + student.postsurvey["experiment_time_setup_evaluation_minutes"]
    setup_mins = setup_time_training + setup_time_evaluation
    reproducibility_rating = student.postsurvey['experiment_rating_reproducibility']
    reproducibility_rating2 = np.mean([student.postsurvey['experiment_rating_download_code'],
                                       student.postsurvey['experiment_rating_download_data'],
                                       student.postsurvey['experiment_rating_setup'],
                                       student.postsurvey['experiment_rating_preprocessing'],
                                       student.postsurvey['experiment_rating_training'],
                                       student.postsurvey['experiment_rating_evaluation']])

    x_points.append(improved_hw_experience)
    y_points.append(reproducibility_rating)
    point_categories.append(student.skill_level)
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

# graph_data = GraphData(
#     x=x_points,
#     y=y_points,
#     x_label='Paper Comprehension Accuracy (%)',
#     y_label='Setup Time (min.)',
#     x_range=(0.0, 1.05),
#     y_range=(0, 2500),
#     y_ticks=list(np.arange(0, 2501, 500)),
#     colors=[SKILL_LEVEL_COLORS[group] for group in point_categories],
#     point_categories=[SKILL_LEVEL_TITLES[group] for group in point_categories],
# )
# scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_time_all.pdf"))    

