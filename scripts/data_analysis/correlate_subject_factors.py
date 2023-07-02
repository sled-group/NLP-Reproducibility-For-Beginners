from ast import Str
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure
# colors = cm.rainbow(np.linspace(0, 1, len(ys)))
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import math
from scipy.stats import spearmanr, pearsonr, sem, linregress, pointbiserialr
import pickle
import os 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

import sys
import json

sys.path.append('.')
from constants import SUBJECT_DATA, GOLD_RESULTS_DATA, \
                      PAPERS, PAPER_TITLES, PAPER_TITLES_SHORT, \
                      RESULT_IDS, RESULT_TITLES, RESULT_TITLES_CONDENSED, \
                      SKILL_LEVELS, SKILL_LEVEL_TITLES, SKILL_LEVEL_TITLES_SHORT, \
                      SKILL_LEVEL_COLORS, PAPER_COLORS, GRAPH_OUTPUT_DIR
from graph import GraphData, scatter_plot_2d

x_points_skill = []
x_points_comprehension = []
x_points_comprehension_simple = []
y_points = []
y_points_rating = []
point_categories = []
point_categories2 = []
for student in SUBJECT_DATA:    
    python_experience = math.floor(student.presurvey['exp_python_now'])
    pytorch_experience = math.floor(student.presurvey['exp_pytorch_now'])
    improved_pytorch_experience = pytorch_experience - student.presurvey['exp_pytorch_before']
    improved_hw_experience = student.presurvey['hw3_score'] - student.presurvey['hw2_score']
    hw2_score = student.presurvey['hw2_score']
    hw3_score = student.presurvey['hw3_score']

    paper_understanding_accuracy = student.postsurvey['understanding_cq_accuracy']
    paper_understanding_accuracy_by_question = [student.postsurvey['understanding_cq%s_correct' % str(cqi + 1)] for cqi in range(7)]

    setup_time_training = student.postsurvey["experiment_time_setup_training_hours"] * 60 + student.postsurvey["experiment_time_setup_training_minutes"]
    setup_time_evaluation = student.postsurvey["experiment_time_setup_evaluation_hours"] * 60 + student.postsurvey["experiment_time_setup_evaluation_minutes"]
    setup_mins = setup_time_training + setup_time_evaluation

    reproducibility_rating = student.postsurvey['experiment_rating_reproducibility']
    code_setup_rating = student.postsurvey['experiment_rating_setup']

    x_points_skill.append([python_experience, pytorch_experience, hw2_score, hw3_score])
    # x_points_skill.append([python_experience, hw3_score])
    x_points_comprehension.append(paper_understanding_accuracy_by_question)
    x_points_comprehension_simple.append(paper_understanding_accuracy)
    y_points.append(setup_mins)
    y_points_rating.append(code_setup_rating)
    point_categories.append(student.skill_level)
    point_categories2.append(student.paper)

x_points_skill = np.array(x_points_skill)
x_points_comprehension = np.array(x_points_comprehension)
f = open(os.path.join(GRAPH_OUTPUT_DIR, "subject_factors_regression.txt"), "w")

# # Multiple linear regression for skill factors vs. code setup rating regression
# variable_names = ['python_experience', 'pytorch_experience', 'hw2_score', 'hw3_score']
# # variable_names = None
# vif = {}
# x_points_skill_ = sm.add_constant(x_points_skill, prepend=False)
# model = sm.OLS(y_points_rating, x_points_skill_)
# res = model.fit()
# f.write('\n\nSkill Factors vs. Code Setup Rating Regression')
# f.write(str(res.summary()))
# f.write('\n')

# VIF for skill factors and comprehension factors
vif_skill = {}
for variable_idx in range(x_points_skill.shape[1]):
    vif_skill[variable_idx] = variance_inflation_factor(x_points_skill, variable_idx)

vif_comprehension = {}
for variable_idx in range(x_points_comprehension.shape[1]):
    vif_comprehension[variable_idx] = variance_inflation_factor(x_points_comprehension, variable_idx)

f.write('\n\nVIF (skill level):\n')
f.write(str(vif_skill))

f.write('\n\nVIF (comprehension):\n')
f.write(str(vif_comprehension))

def get_r_squared(x, y):
    linear_regression = LinearRegression()
    linear_regression.fit(x, y)
    r_squared = linear_regression.score(x, y)
    r_squared_adjusted = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - x.shape[1] - 1)
    return r_squared, r_squared_adjusted

# # Linear regression r-squared for skill factors and comprehension factors vs. setup time, 
# r_squared, r_squared_adjusted = get_r_squared(x_points_skill, y_points)
# f.write('\n\nr-squared from linear regression (skill level factors vs. setup time):\n')
# f.write(str(r_squared))
# f.write('\nadjusted: ' + str(r_squared_adjusted) + '\n')

# r_squared, r_squared_adjusted = get_r_squared(x_points_comprehension, y_points)
# f.write('\n\nr-squared from linear regression (comprehension factors vs. setup time):\n')
# f.write(str(r_squared))
# f.write('\nadjusted: ' + str(r_squared_adjusted) + '\n')

# # Linear regression r-squared for skill factors vs. code setup rating
# r_squared, r_squared_adjusted = get_r_squared(x_points_skill, y_points_rating)
# f.write('\n\nr-squared from linear regression (skill level vs. code setup rating):\n')
# f.write(str(r_squared))
# f.write('\nadjusted: ' + str(r_squared_adjusted) + '\n')

# r_squared, r_squared_adjusted = get_r_squared(x_points_comprehension, y_points_rating)
# f.write('\n\nr-squared from linear regression (comprehension factors vs. code setup rating):\n')
# f.write(str(r_squared))
# f.write('\nadjusted: ' + str(r_squared_adjusted) + '\n')



# Spearman and Pearson correlations for skill level
f.write('\n\nSpearman and Pearson correlations:')
for i in range(x_points_skill.shape[1]):
    f.write("\n\nSkill level variable %s:" % str(i))

    corr_aux_spearman, p_aux_spearman = spearmanr(x_points_skill[:, i], y_points)
    f.write('\n\n\tSkill Level vs. Code Setup Time Spearman correlation: %.3f (p=%.5f)' % (corr_aux_spearman, p_aux_spearman))

    corr_aux_spearman, p_aux_spearman = spearmanr(x_points_skill[:, i], y_points_rating)
    f.write('\n\n\tSkill Level vs. Code Setup Rating Spearman correlation: %.3f (p=%.5f)' % (corr_aux_spearman, p_aux_spearman))

    # slope, intercept, corr_aux_linreg, p_aux_linreg, slope_err = linregress(x_points_skill[:, i], y_points_rating)
    # r_squared = corr_aux_linreg ** 2
    # r_squared_adjusted = 1 - (1 - r_squared) * (len(y_points_rating) - 1) / (len(y_points_rating) - x_points_comprehension.shape[1] - 1)

    # f.write('\n\n\tSkill Level vs. Code Setup Rating linear regression/Pearson correlation: slope=%.5f (+- %.5f), intercept=%.5f, r=%.5f (p=%.5f), r^2=%.5f, r^2 adj.=%.5f' % ( slope, slope_err, intercept, corr_aux_linreg, p_aux_linreg, r_squared, r_squared_adjusted))        


# Spearman and Pearson correlations for comprehension factors vs. code setup rating
f.write("\n\nComprehension accuracy vs. code setup rating:")

corr_aux_spearman, p_aux_spearman = spearmanr(x_points_comprehension_simple, y_points)
f.write('\n\n\tPaper Comprehension vs. Code Setup Time Spearman correlation: %.3f (p=%.5f)' % (corr_aux_spearman, p_aux_spearman))

corr_aux_spearman, p_aux_spearman = spearmanr(x_points_comprehension_simple, y_points_rating)
f.write('\n\n\tPaper Comprehension vs. Code Setup Rating Spearman correlation: %.3f (p=%.5f)' % (corr_aux_spearman, p_aux_spearman))

slope, intercept, corr_aux_linreg, p_aux_linreg, slope_err = linregress(x_points_comprehension_simple, y_points_rating)
f.write('\n\n\tPaper Comprehension vs. Code Setup Rating linear regression/Pearson correlation: slope=%.5f (+- %.5f), intercept=%.5f, r=%.5f (p=%.5f)' % ( slope, slope_err, intercept, corr_aux_linreg, p_aux_linreg))        

# Point-biserial correlation between correctness on individual questions and experience factors
f.write("\n\nComprehension questions vs. code setup time and rating:")

for group in SKILL_LEVELS:
    this_x = np.array([x for x, c in zip(x_points_comprehension, point_categories) if group == c])
    this_y = np.array([x for x, c in zip(y_points, point_categories) if group == c])

    f.write('\n\nSkill Level %s:' % group)
    for i in range(this_x.shape[1]):

        r_pointbiserial, p_pointbiserial = pointbiserialr(this_x[:, i], this_y)
        f.write('\n\n\tQ%s: Paper Comprehension vs. Code Setup Time point biserial correlation: %.3f (p=%.5f)' % (str(i+1), r_pointbiserial, p_pointbiserial))

        # r_pointbiserial, p_pointbiserial = pointbiserialr(this_x[:, i], this_y_rating)
        # f.write('\n\tQ%s: Paper Comprehension vs. Code Setup Rating point biserial correlation: %.3f (p=%.5f)' % (str(i+1), r_pointbiserial, p_pointbiserial))

for group in PAPERS:
    this_x = np.array([x for x, c in zip(x_points_comprehension, point_categories2) if group == c])
    this_y = np.array([x for x, c in zip(y_points, point_categories2) if group == c])

    f.write('\n\nPaper %s:' % group)
    for i in range(this_x.shape[1]):

        r_pointbiserial, p_pointbiserial = pointbiserialr(this_x[:, i], this_y)
        f.write('\n\n\tQ%s: Paper Comprehension vs. Code Setup Time point biserial correlation: %.3f (p=%.5f)' % (str(i+1), r_pointbiserial, p_pointbiserial))


f.write('\n\nOverall:')
x_points_comprehension = np.array(x_points_comprehension)
for i in range(x_points_comprehension.shape[1]):
    r_pointbiserial, p_pointbiserial = pointbiserialr(x_points_comprehension[:, i], y_points)
    f.write('\n\n\tQ%s: Paper Comprehension vs. Code Setup Time point biserial correlation: %.3f (p=%.5f)' % (str(i+1), r_pointbiserial, p_pointbiserial))

# r_pointbiserial, p_pointbiserial = pointbiserialr(x_points_comprehension[:, i], y_points_rating)
# f.write('\n\tQ%s: Paper Comprehension vs. Code Setup Rating point biserial correlation: %.3f (p=%.5f)' % (str(i+1), r_pointbiserial, p_pointbiserial))




graph_data = GraphData(
    x=x_points_comprehension_simple,
    y=y_points,
    x_label='Paper Comprehension Accuracy (%)',
    y_label='Setup Time (min.)',
    x_range=(0.0, 1.05),
    y_range=(0, 1000),
    y_ticks=list(np.arange(0, 1001, 200)),
    colors=[SKILL_LEVEL_COLORS[group] for group in point_categories],
    point_categories=[SKILL_LEVEL_TITLES[group] for group in point_categories]
)
scatter_plot_2d(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "understanding_vs_setup_time_all.pdf"))    


f.close()
