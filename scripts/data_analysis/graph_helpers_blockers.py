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
from pprint import pprint

sys.path.append('.')
from constants import SUBJECT_DATA, GOLD_RESULTS_DATA, \
                      PAPERS, PAPER_TITLES, PAPER_TITLES_SHORT, \
                      RESULT_IDS, RESULT_TITLES, RESULT_TITLES_CONDENSED, \
                      SKILL_LEVELS, SKILL_LEVEL_TITLES, SKILL_LEVEL_TITLES_SHORT, \
                      SKILL_LEVEL_COLORS, PAPER_COLORS, GRAPH_OUTPUT_DIR
from graph import GraphData, stacked_count_bar_chart
from data import FREETEXT_ANNOTATION_TITLES

print('%s subjects total' % len(SUBJECT_DATA))
print("%s subjects reported ACLRC additions" % len([subject for subject in SUBJECT_DATA if 'aclrc_additions' in subject.postsurvey]))

# Graph all freetext responses by assigned papers
for freetext_key in FREETEXT_ANNOTATION_TITLES:
    for GROUPS, TITLES, COLORS, GROUP_TYPE in [ (SKILL_LEVELS, SKILL_LEVEL_TITLES, SKILL_LEVEL_COLORS, "skill_level"),
                                                (PAPERS, PAPER_TITLES, PAPER_COLORS, "paper")]:
        STEP_TITLES = FREETEXT_ANNOTATION_TITLES[freetext_key]
        all_data = {st: {p: 0 for p in GROUPS} for st in set(STEP_TITLES.values())}
        for subject in SUBJECT_DATA:
            if freetext_key not in subject.postsurvey:
                continue
            already_counted = defaultdict(bool)
            for response in subject.postsurvey[freetext_key]:
                if not already_counted[STEP_TITLES[response]]:
                    all_data[STEP_TITLES[response]][getattr(subject, GROUP_TYPE)] += 1
                    already_counted[STEP_TITLES[response]] = True
        
        def sum_counts_over_groups(all_data_step_title, groups):
            return sum([all_data_step_title[g] for g in groups])

        # Any rare suggestions or ones that aren't interesting can be relabeled as "other"
        regroup_titles = {}
        for step, step_title in STEP_TITLES.items():
            if sum_counts_over_groups(all_data[step_title], GROUPS) < 3 or step in ["paper_specific", "platform_specific"]:
                regroup_titles[step_title] = "Other" # Map old title to new title of response
                STEP_TITLES[step] = step_title
            else:
                regroup_titles[step_title] = STEP_TITLES[step]

        # Regroup data by titles
        all_data_temp = {st: {p: 0 for p in GROUPS} for st in set(regroup_titles.values())}
        for step_title in all_data:
            for paper in all_data[step_title]:
                all_data_temp[regroup_titles[step_title]][paper] += all_data[step_title][paper]
        all_data = all_data_temp

        top_steps_final = [st for st, _ in Counter({step_title: sum_counts_over_groups(all_data[step_title], GROUPS) for step_title in all_data if step_title not in ["Other", "Already\nIncluded"]}).most_common(5)]
        for step_title in all_data:
            if step_title not in top_steps_final and step_title not in ["Other", "Already\nIncluded"]:
                for group in GROUPS:
                    all_data["Other"][group] += all_data[step_title][group]
        for step_title in list(all_data.keys()):
            if step_title not in top_steps_final and step_title not in ["Other", "Already\nIncluded"]:
                del all_data[step_title]

        # Sort by frequency of suggestions
        all_data = {k: v for k, v in sorted(all_data.items(), key=lambda x:sum([x[1][p] for p in GROUPS]), reverse=True)}
        json.dump(all_data, open(os.path.join(GRAPH_OUTPUT_DIR, "counts_%s_by_%s.json" % (freetext_key, GROUP_TYPE)), 'w'), indent=4)

        # Move "Other" and "Already\nIncluded" to end regardless of frequency
        all_data = {k: v for k, v in all_data.items() if k not in ["Other", "Already\nIncluded"]} | \
                   {k: v for k, v in all_data.items() if k == "Other"} | \
                    {k: v for k, v in all_data.items() if k == "Already\nIncluded"}
        x_labels = list(all_data.keys())


        max_val = max([sum_counts_over_groups(all_data[step_title], GROUPS) for step_title in all_data])
        upper_lim = min([n for n in range(0,100,10) if n > max_val])

        all_data = np.array([[all_data[k1][k2] for k2 in all_data[k1]] for k1 in all_data])


        # Generate version of graph broken down by grouping (skill level or papers)
        graph_data = GraphData(
            x=np.arange(len(x_labels)),
            y=all_data,
            x_label=x_labels,
            y_label="Frequency",
            y_range=(0,upper_lim+10),
            category_names=[TITLES[p] for p in GROUPS],
            figure_size=(17,6),
            colors=[COLORS[p] for p in GROUPS],
            font_size=29,
        )
        stacked_count_bar_chart(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "counts_%s_by_%s.pdf" % (freetext_key, GROUP_TYPE)))

        # Generate version of graph not broken down
        graph_data = GraphData(
            x=np.arange(len(x_labels)),
            y=all_data,
            x_label=x_labels,
            y_label="Frequency",
            y_range=(0,upper_lim+10),
            category_names=[TITLES[p] for p in GROUPS],
            figure_size=(17,6),
            colors=["gray"],
            font_size=29,
        )
        stacked_count_bar_chart(graph_data, os.path.join(GRAPH_OUTPUT_DIR, "counts_%s.pdf" % (freetext_key)), ungroup=True)

