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


# Gather subject-reported results
y_points = []
for subject in SUBJECT_DATA:        
    result = None
    for result_id in RESULT_IDS:
        if result_id in subject.postsurvey:
            result = subject.postsurvey[result_id]
            if result is None:
                raise ValueError("Subject %s is missing reproduced result." % str(subject.subject_id))
    
            if result >= 0.0 and result <= 1.0:
                result *= 100.
            y_points.append((subject, result_id, result))

# Graph 1: Accuracy by skill level
width = 0.75
font_size = 14
fig, ax = plt.subplots()
labels = [SKILL_LEVEL_TITLES[group] for group in SKILL_LEVELS]
x = np.arange(len(labels))  # the label locations
group_y = [[rating - GOLD_RESULTS_DATA[result_id]['reported'] for s, result_id, rating in y_points if s.skill_level == group and not math.isnan(rating)] for group in SKILL_LEVELS]

for sublist in group_y:
    assert len(sublist) > 0, "Missing some data..."
group_err = [sem([rating for s, result_id, rating in y_points if s.skill_level == group]) for group in SKILL_LEVELS]
for gi, group in enumerate(SKILL_LEVELS):
    group_rects = ax.bar(gi, np.mean(group_y[gi]), width, yerr=group_err[gi], color=SKILL_LEVEL_COLORS[group], label=group[0].upper() + group[1:])  
    ax.text(gi + 0.05, np.mean(group_y[gi])+(0.75 if np.mean(group_y[gi]) > 0.0 else -0.75), '%.2f' % np.mean(group_y[gi]), color=SKILL_LEVEL_COLORS[group], fontsize=font_size, fontweight='bold')
ax.set_ylabel('Relative Error (%)', fontsize=font_size, fontweight='bold')
ax.set_xticks([])
ax.spines["bottom"].set_position(("data", 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)    
plt.ylim(-3,3)
plt.legend(fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)    
fig.tight_layout()
fig.set_figheight(3)    
fig.set_figwidth(7)
plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, "accuracy_by_skill_level.pdf"))

# Graph 2: accuracy by paper result
plt.clf()
fig, ax = plt.subplots()
labels = RESULT_TITLES_CONDENSED
x = np.arange(len(labels))  # the label locations
group_y = [[rating - GOLD_RESULTS_DATA[result_id]['reported'] for s, result_id, rating in y_points if result_id == group and not math.isnan(rating)] for group in RESULT_IDS]
for sublist in group_y:
    assert len(sublist) > 0, "Missing some data..."
group_err = [sem([rating for s, result_id, rating in y_points if result_id == group]) for group in RESULT_IDS]
for gi, group in enumerate(RESULT_IDS):
    group_rects = ax.bar(gi, np.mean(group_y[gi]), width, yerr=group_err[gi], color=PAPER_COLORS[group.split('_')[-1][0]], label=labels[gi] if gi == 0 or labels[gi] != labels[gi-1] else None)  
ax.set_ylabel('Relative Error (%)', fontsize=font_size, fontweight='bold')
ax.set_xticklabels(labels)
ax.plot(x, [GOLD_RESULTS_DATA[l]['expert'] - GOLD_RESULTS_DATA[l]['reported'] for l in RESULT_IDS], color='gray', marker='o', linestyle='dashed', label='Expert')
ax.set_xticks([])
ax.spines["bottom"].set_position(("data", 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)    
plt.ylim(-3,3)
plt.legend(fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)    
fig.tight_layout()
fig.set_figheight(3)    
fig.set_figwidth(7)
plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, "accuracy_by_paper.pdf"))
