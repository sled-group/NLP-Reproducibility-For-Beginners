import os
import json
import pandas as pd
from data import Subject

PAPERS = ['A', 'B', 'C']
PAPER_TITLES = {'A': 'Paper A',
                'B': 'Paper B',
                'C': 'Paper C'}
PAPER_TITLES_SHORT = {
                'A': 'A',
                'B': 'B',
                'C': 'C'}                

DATA_PATH = "data_files/sample_data.json"
SUBJECT_DATA = [Subject.from_dict(subject) for subject in json.load(open(DATA_PATH, 'r'))]
for subject in SUBJECT_DATA:
    for key in subject.postsurvey:
        # Reverse difficulty ratings to be 5 for most difficult
        if key.startswith('experiment_rating'):
            subject.postsurvey[key] = 6 - subject.postsurvey[key]

GOLD_RESULTS_PATH = './data_files/expert_reported_results.csv'
GOLD_RESULTS_DATA = {row['result_id']: row for ri, row in pd.read_csv(GOLD_RESULTS_PATH).iterrows()}

RUNTIME_BY_PAPER_SUCCESSFUL_ONLY = {"A": 33, "B": 167, "C": 107}
RUNTIME_BY_PAPER = {"A": 34, "B": 171, "C": 116}

SETUPTIME_BY_PAPER = {"A": 120, "B": 120, "C": 120}

RESULT_IDS = list(GOLD_RESULTS_DATA.keys())
RESULT_TITLES = ["Paper %s" % rid.split('_')[-1][0] + ", Result %s" % rid.split('_')[-1][1] if len(rid.split('_')[-1]) == 2 else "" for rid in RESULT_IDS]
RESULT_TITLES_CONDENSED = ["Paper " + result_id.split('_')[-1][0] for result_id in RESULT_IDS]

SKILL_LEVELS = ['novice', 'intermediate', 'advanced']
SKILL_LEVEL_TITLES = {'novice': 'Novice',
                      'intermediate': 'Intermediate',
                      'advanced': 'Advanced'}
SKILL_LEVEL_TITLES_SHORT = {'novice': 'Nov.',
                            'intermediate': 'Int.',
                            'advanced': 'Adv.'}                      

SKILL_LEVEL_COLORS = {SKILL_LEVELS[0]: (204/255, 121/255, 167/255),
                        SKILL_LEVELS[1]: (230/255, 159/255, 0/255),
                        SKILL_LEVELS[2]: (0/255, 158/255, 115/255)}
PAPER_COLORS = {'A': (213/255, 94/255, 0/255),
                 'B':  (240/255, 228/255, 66/255),
                 'C': (86/255, 180/255, 233/255)}                 

GRAPH_OUTPUT_DIR = './generated_graphs'
if not os.path.exists(GRAPH_OUTPUT_DIR):
    os.makedirs(GRAPH_OUTPUT_DIR)