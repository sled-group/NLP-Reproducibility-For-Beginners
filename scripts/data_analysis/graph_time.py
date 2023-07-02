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
                      SKILL_LEVEL_COLORS, PAPER_COLORS, GRAPH_OUTPUT_DIR, \
                      RUNTIME_BY_PAPER, SETUPTIME_BY_PAPER
from graph import GraphData, box_plot                    

y_setup_time = []
y_runtime = []
for student in SUBJECT_DATA:    
    setup_time_training = student.postsurvey["experiment_time_setup_training_hours"] * 60 + student.postsurvey["experiment_time_setup_training_minutes"]
    setup_time_evaluation = student.postsurvey["experiment_time_setup_evaluation_hours"] * 60 + student.postsurvey["experiment_time_setup_evaluation_minutes"]
    gpu_mins = student.gpu_mins if student.gpu_mins is not None else 0

    y_setup_time.append((student, setup_time_training + setup_time_evaluation))
    if gpu_mins > 0:
        y_runtime.append((student, gpu_mins))

FONT_SIZE = 18
FIGURE_SIZE = (3,5)

# Graph 1: Setup time by skill level
labels = [SKILL_LEVEL_TITLES_SHORT[group] for group in SKILL_LEVELS]
x = np.arange(len(labels))  # the label locations
group_y = [[rating for s, rating in y_setup_time if s.skill_level == group and not math.isnan(rating)] for group in SKILL_LEVELS]
print([len(gy) for gy in group_y])
for sublist in group_y:
    assert len(sublist) > 0, "Missing some data..."
graph_data = GraphData(
    x=x,
    y=group_y,
    x_label=labels,
    y_label="Setup Time (min.)",
    y_range=(0,800),
    colors=["cyan"],
    font_size=FONT_SIZE,
    figure_size=FIGURE_SIZE,
)
box_plot(graph_data, save_path=os.path.join(GRAPH_OUTPUT_DIR, "setup_time_by_skill_level.pdf"))

# Graph 2: Runtime by skill level
labels = [SKILL_LEVEL_TITLES_SHORT[group] for group in SKILL_LEVELS]
x = np.arange(len(labels))  # the label locations
group_y = [[(rating - RUNTIME_BY_PAPER[s.paper]) / RUNTIME_BY_PAPER[s.paper] * 100.0 for s, rating in y_runtime if s.skill_level == group and not math.isnan(rating) and rating > 0] for group in SKILL_LEVELS]
# group_y = [[RUNTIME_BY_PAPER[s.paper] for s, rating in y_runtime if s.skill_level == group and not math.isnan(rating) and rating > 0] for group in SKILL_LEVELS]
print([len(gy) for gy in group_y])
for sublist in group_y:
    assert len(sublist) > 0, "Missing some data..."
graph_data = GraphData(
    x=x,
    y=group_y,
    x_label=labels,
    y_label="Runtime (% error)",
    y_range=(-200.0,1000.0),
    # y_range=(0,2000),
    colors=["red"],
    font_size=FONT_SIZE,
    figure_size=FIGURE_SIZE,
)
box_plot(graph_data, save_path=os.path.join(GRAPH_OUTPUT_DIR, "runtime_by_skill_level.pdf"))

# Graph 3: Setup time by paper
labels = [PAPER_TITLES_SHORT[paper] for paper in PAPERS]
x = np.arange(len(labels))  # the label locations
group_y = [[rating for s, rating in y_setup_time if s.paper == group and not math.isnan(rating) and rating > 0] for group in PAPERS]
print([len(gy) for gy in group_y])
for sublist in group_y:
    assert len(sublist) > 0, "Missing some data..."
graph_data = GraphData(
    x=x,
    y=group_y,
    x_label=labels,
    y_label="Setup Time (min.)",
    y_range=(0,800),
    colors=["cyan"],
    font_size=FONT_SIZE,
    figure_size=FIGURE_SIZE,
)
box_plot(graph_data, save_path=os.path.join(GRAPH_OUTPUT_DIR, "setup_time_by_paper.pdf"))

# Graph 4: Runtime by paper
labels = [PAPER_TITLES_SHORT[paper] for paper in PAPERS]
x = np.arange(len(labels))  # the label locations
group_y = [[(rating - RUNTIME_BY_PAPER[s.paper]) / RUNTIME_BY_PAPER[s.paper] * 100.0 for s, rating in y_runtime if s.paper == group and not math.isnan(rating) and rating > 0] for group in PAPERS]
# group_y = [[rating for s, rating in y_runtime if s.paper == group and not math.isnan(rating)] for group in PAPERS]
print([len(gy) for gy in group_y])
for sublist in group_y:
    assert len(sublist) > 0, "Missing some data..."
graph_data = GraphData(
    x=x,
    y=group_y,
    x_label=labels,
    y_label="Runtime (% error)",
    y_range=(-200.0,1000.0),
    # y_range=(0,2000),
    colors=["red"],
    font_size=FONT_SIZE,
    figure_size=FIGURE_SIZE,
)
box_plot(graph_data, save_path=os.path.join(GRAPH_OUTPUT_DIR, "runtime_by_paper.pdf"))
