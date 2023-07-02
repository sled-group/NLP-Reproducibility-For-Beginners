from pprint import pprint
import sys

sys.path.append('.')
from constants import SUBJECT_DATA, GOLD_RESULTS_DATA, \
                      PAPERS, PAPER_TITLES, PAPER_TITLES_SHORT, \
                      RESULT_IDS, RESULT_TITLES, RESULT_TITLES_CONDENSED, \
                      SKILL_LEVELS, SKILL_LEVEL_TITLES, SKILL_LEVEL_TITLES_SHORT, \
                      SKILL_LEVEL_COLORS, PAPER_COLORS, GRAPH_OUTPUT_DIR
from graph import GraphData, grouped_mean_bar_chart       
from data import ACLRC_ITEMS             

dist = {p: {s: 0 for s in SKILL_LEVELS} for p in PAPERS}
for student in SUBJECT_DATA:
    dist[student.paper][student.skill_level] += 1

print('%s subjects in the data' % len(SUBJECT_DATA))
pprint(dist)
