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
from statsmodels.miscmodels.ordinal_model import OrderedModel
import sys
from sklearn.linear_model import LinearRegression
import json

sys.path.append('.')
from constants import SUBJECT_DATA, GOLD_RESULTS_DATA, \
                      PAPERS, PAPER_TITLES, PAPER_TITLES_SHORT, \
                      RESULT_IDS, RESULT_TITLES, RESULT_TITLES_CONDENSED, \
                      SKILL_LEVELS, SKILL_LEVEL_TITLES, SKILL_LEVEL_TITLES_SHORT, \
                      SKILL_LEVEL_COLORS, PAPER_COLORS, GRAPH_OUTPUT_DIR, \
                      RUNTIME_BY_PAPER
from graph import GraphData, grouped_mean_bar_chart       
from data import ACLRC_ITEMS             


point_categories1 = []
point_categories2 = []
X_points = []
X_points_runtime = []
papers_runtime = []
step_data = {}
checklist_features_l = {item: [] for item in ACLRC_ITEMS}
item_freq = defaultdict(int)
item_freq_by_paper = {p: defaultdict(int) for p in PAPERS}
for student in SUBJECT_DATA:
    setup_time_training = student.postsurvey["experiment_time_setup_training_hours"] * 60 + student.postsurvey["experiment_time_setup_training_minutes"]
    setup_time_evaluation = student.postsurvey["experiment_time_setup_evaluation_hours"] * 60 + student.postsurvey["experiment_time_setup_evaluation_minutes"]
    
    setup_mins = setup_time_training + setup_time_evaluation
    student.postsurvey['setup_hours'] = setup_mins #/ 60.

    runtime_mins = student.gpu_mins if not math.isnan(student.gpu_mins) and student.gpu_mins is not None else None
    runtime_pct_error = ((runtime_mins - RUNTIME_BY_PAPER[student.paper]) / RUNTIME_BY_PAPER[student.paper]) if runtime_mins else None
    student.postsurvey['runtime_pct_error'] = runtime_pct_error

    reproducibility_steps = {    
        'setup_time': 'setup_hours',
        'setup_ease': 'experiment_rating_setup',
        'run_time': 'runtime_pct_error',
    }

    for step, question in reproducibility_steps.items():
        if student.postsurvey[question] is not None:
            if step not in step_data:
                step_data[step] = []
            step_data[step].append(student.postsurvey[question])
    
    checked_items = student.postsurvey['aclrc_helpful']
    for item in checked_items:            
        assert item in ACLRC_ITEMS, "Checklist item not found: %s" % item

    checklist_features = {}
    for ii, item in enumerate(ACLRC_ITEMS):
        if item in checked_items:
            checklist_features['I%s' % int(ii)] = 1
            checklist_features_l[item].append(1)
            if item not in [ACLRC_ITEMS[1], ACLRC_ITEMS[16]]:
                item_freq['I%s' % int(ii)] += 1
                item_freq_by_paper[student.paper]['I%s' % int(ii)] += 1
        else:
            checklist_features['I%s' % int(ii)] = 0
            checklist_features_l[item].append(0)

    X_points.append(checklist_features)
    point_categories1.append(student.skill_level)
    point_categories2.append(student.paper)
    if runtime_mins:
        X_points_runtime.append(checklist_features)
        papers_runtime.append(student.paper)

f = open(os.path.join(GRAPH_OUTPUT_DIR, "aclrc_regression_results.txt"), "w")
X_points_mapping = {'setup_time': X_points, 
                    'setup_ease': X_points,
                    'run_time': X_points_runtime,
                    }
categories_mapping = {'setup_time': point_categories2, 
                    'setup_ease': point_categories2,
                    'run_time': papers_runtime,
                    }                    

# Measure VIF of all features (only items 1 and 16 have a high VIF, because basically everyone chose them out of necessity)
X_points_df = pd.DataFrame(X_points)
vif_all = {}
for variable_idx in range(len(ACLRC_ITEMS)):
    vif_all[variable_idx] = variance_inflation_factor(X_points_df, variable_idx)

f.write('VIF of all items:')
for variable, vif in vif_all.items():
    f.write('\n' + str(variable) + ': ' + str(vif))
f.write('\n\n')

top_items = [item for item, _ in Counter(item_freq).most_common()]

X_points_df = pd.DataFrame(X_points)
vif_all = {}
for variable_name in top_items:
    variable_idx = int(variable_name.replace('I',''))
    vif_all[variable_idx] = variance_inflation_factor(X_points_df, variable_idx)

f.write('\n\nVIF of considered items:')
for variable, vif in vif_all.items():
    f.write('\n' + str(variable) + ': ' + str(vif))
f.write('\n\n')

# Point-biserial correlation (appropriate for comparing binary variables with continuous variables)
X_points = np.array(X_points)

f.write('\n\nOVERALL ANALYSIS:')
for step in step_data:
    if step != 'setup_ease':
        this_X_points_df = pd.DataFrame(X_points_mapping[step])
        this_X_points = this_X_points_df.to_numpy()
        f.write('\t\n\n%s:' % step.upper())
        for i in range(this_X_points.shape[1]):
            if i not in [1, 16]:
                r_pointbiserial, p_pointbiserial = pointbiserialr(this_X_points[:, i], step_data[step])
                f.write('\n\t\tACL Checklist Item %s vs. %s point biserial correlation: %.3f (p=%.5f)' % (str(i +1), step, r_pointbiserial, p_pointbiserial) + ('***' if p_pointbiserial < 0.05 else ''))

        # Ordinal logistic regression for code setup difficulty
        f.write('\n\nLOGISTIC REGRESSION FOR CODE SETUP:\n')
        this_X_points_df.drop(columns=['I1', 'I16'])
        mod_prob = OrderedModel(step_data[step],
                                this_X_points_df,
                                distr='probit')
        res_prob = mod_prob.fit(method='bfgs')
        f.write(str(res_prob.summary()))

for pi, paper in enumerate(PAPERS):
    f.write('\n\nPAPER %s ANALYSIS:' % paper)
    top_items = [item for item, _ in Counter(item_freq_by_paper[paper]).most_common()]

    for step in step_data:
        if step != 'setup_ease':
            this_X_points_df = pd.DataFrame(X_points_mapping[step])
            this_X_points = this_X_points_df.to_numpy()
            this_X_points = np.array([X for X, p in zip(this_X_points, categories_mapping[step]) if p == paper])
            y_paper_step = np.array([y for y, p in zip(step_data[step], categories_mapping[step]) if p == paper])

            # Point-biserial correlation (appropriate for comparing binary variables with continuous variables)
            f.write('\t\n\n%s:' % step.upper())
            for i in range(this_X_points.shape[1]):
                if i not in [1, 16]:
                    r_pointbiserial, p_pointbiserial = pointbiserialr(this_X_points[:, i], y_paper_step)
                    f.write('\n\t\tACL Checklist Item %s vs. %s point biserial correlation: %.3f (p=%.5f)' % (str(i+1), step, r_pointbiserial, p_pointbiserial) + ('***' if p_pointbiserial < 0.05 else ''))



all_data_by_skill_level = {k: [] for k in ACLRC_ITEMS}

all_graph_data_by_paper = [[[X['I%s' % int(ai)] for X, p in zip(X_points, point_categories2) if p == paper] for paper in PAPERS] for ai, aclrc_item in enumerate(ACLRC_ITEMS)]
all_graph_data_by_skill_level = [[[X['I%s' % int(ai)] for X, p in zip(X_points, point_categories1) if p == skill_level] for skill_level in SKILL_LEVELS] for ai, aclrc_item in enumerate(ACLRC_ITEMS)]

all_graph_data_by_paper = GraphData(
    x=np.arange(len(ACLRC_ITEMS)),
    y=all_graph_data_by_paper,
    x_label=[str(i+1) for i in range(len(ACLRC_ITEMS))],
    y_label="% Chosen",
    y_range=(0,1.05),
    y_ticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    category_names=[PAPER_TITLES[p] for p in PAPERS],
    figure_size=(20,6),
    colors=[PAPER_COLORS[p] for p in PAPERS],
    font_size=14,
    add_error_bars=True,
)
grouped_mean_bar_chart(all_graph_data_by_paper, os.path.join(GRAPH_OUTPUT_DIR, "aclrc_items_by_paper.pdf"))


all_graph_data_by_skill_level = GraphData(
    x=np.arange(len(ACLRC_ITEMS)),
    y=all_graph_data_by_skill_level,
    x_label=[str(i+1) for i in range(len(ACLRC_ITEMS))],
    y_label="% Chosen",
    y_range=(0,1.05),
    y_ticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    category_names=[SKILL_LEVEL_TITLES[p] for p in SKILL_LEVELS],
    figure_size=(20,6),
    colors=[SKILL_LEVEL_COLORS[p] for p in SKILL_LEVEL_TITLES],
    font_size=14,
    add_error_bars=True,
)
grouped_mean_bar_chart(all_graph_data_by_skill_level, os.path.join(GRAPH_OUTPUT_DIR, "aclrc_items_by_skill_level.pdf"))
