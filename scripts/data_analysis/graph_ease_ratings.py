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
from graph import GraphData, grouped_mean_bar_chart

STEP_TITLES = {
    "experiment_rating_download_code": "code\ndownl.",
    "experiment_rating_download_data": "data\ndownl.",
    "experiment_rating_setup": "code\nsetup",
    "experiment_rating_preprocessing": "data\npreproc.",
    "experiment_rating_training": "system\ntraining",
    "experiment_rating_evaluation": "system\neval.",
}
rating_data = {k: {k2: [] for k2 in PAPERS} for k in STEP_TITLES}
rating_data2 = {k: {k2: [] for k2 in SKILL_LEVELS} for k in STEP_TITLES}
for subject in SUBJECT_DATA:
    for k in rating_data:
        rating_data[k][subject.paper].append(subject.postsurvey[k])
        rating_data2[k][subject.skill_level].append(subject.postsurvey[k])

graph_data = GraphData(
    x=np.arange(len(STEP_TITLES)),
    y=[[rating_data2[k][k2] for k2 in rating_data2[k]] for k in rating_data2],
    x_label=list(STEP_TITLES.values()),
    y_label="Difficulty Rating (1-5)",
    y_range=(0,5.5),
    category_names=[SKILL_LEVEL_TITLES[p] for p in SKILL_LEVELS],
    figure_size=(135,6),
    colors=[SKILL_LEVEL_COLORS[p] for p in SKILL_LEVELS],
    add_error_bars=True,
    font_size=26,
)
grouped_mean_bar_chart(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "step_ease_ratings_by_skill_level.pdf"))

graph_data = GraphData(
    x=np.arange(len(STEP_TITLES)),
    y=[[rating_data[k][k2] for k2 in rating_data[k]] for k in rating_data],
    x_label=list(STEP_TITLES.values()),
    y_label="Difficulty Rating (1-5)",
    y_range=(0,5.5),
    category_names=[PAPER_TITLES[p] for p in PAPERS],
    figure_size=(42,6),
    colors=[PAPER_COLORS[p] for p in PAPERS],
    add_error_bars=True,
    font_size=26,
)
grouped_mean_bar_chart(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "step_ease_ratings_by_paper.pdf"))

