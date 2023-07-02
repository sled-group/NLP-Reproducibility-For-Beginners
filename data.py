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
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Any
from pprint import pprint


@dataclass_json
@dataclass
class Subject:
    subject_id: int
    skill_level: str
    paper: str

    presurvey: dict[str, Any] = None
    postsurvey: dict[str, Any] = None

    gpu_mins: int = None


# Correct answers for paper comprehension questions
COMPREHENSION_QUESTIONS_ANSWER_KEY = {"A": {
                                        "understanding_cq1": "At the time of writing, there were no benchmark datasets evaluating temporal reasoning for NLP.",
                                        "understanding_cq2": "Textual entailment with a focus on ordering of events.",
                                        "understanding_cq3": "A story written in text, and a hypothesis about the temporal ordering of events in the story.",
                                        "understanding_cq4": "An inference label of entailment or contradiction.",
                                        "understanding_cq5": "tracie/code/models/ptntime/train_t5.py",
                                        "understanding_cq6": "The percentage of stories for which the system made correct predictions on all hypotheses.",
                                        "understanding_cq7": "When used in a zero-shot setting, the proposed models consistently outperform baselines from prior work fine-tuned on the task."
                                      },
                                      "B": {
                                        "understanding_cq1": "Aligning recipes based only on actions ignores rich information about the structure of recipes and relationships between sentences.",
                                        "understanding_cq2": "Alignment of actions in multiple recipes for the same dish.",
                                        "understanding_cq3": "Two recipes for a single dish, and a source action selected from one of the recipes.",
                                        "understanding_cq4": "Either one action from the target recipe which best aligns to the given source action, or no matching actions.",
                                        "understanding_cq5": "ara/Alignment_Model/model.py",
                                        "understanding_cq6": "Ten instances of the alignment model are trained using cross validation, where each fold holds out the recipes from one of the ten dishes as validation data, and from another dish as testing data. The testing results on each of the ten dishes are combined from the ten model instances.",
                                        "understanding_cq7": "The alignment models struggle to generalize, as they perform better on recipes for dishes seen in training than those not seen in training."
                                      },
                                      "C": {
                                        "understanding_cq1": "Tables are uniquely challenging to understand because they convey explicit information that unstructured text does not.",
                                        "understanding_cq2": "Textual entailment based on a semi-structured context.",
                                        "understanding_cq3": "A table converted to a set of key-value pairs, and a proposed fact about the table.",
                                        "understanding_cq4": "An inference label of entailment, neutral, or contradiction.",
                                        "understanding_cq5": "infotabs-code/scripts/preprocess/json_to_struct.py",
                                        "understanding_cq6": "Models that overfit to superficial lexical cues will struggle with the α2 test set, while models that overfit to domain-specific statistical cues will struggle with the α3 test set.",
                                        "understanding_cq7": "A support vector machine performs better than transformer-based language models on InfoTabs when representing tables as paragraphs."
                                      }
                                    }


# Categories that subjects' freetext survey responses were labeled with, and explanations of them (written by paper authors, not subjects). 
# Some freetext responses may not fit any categories (i.e., subject gave no relevant feedback), and categories may not be mutually exclusive
FREETEXT_ANNOTATION_KEY = {
    "Did the authors provide anything in the code release that helped make the results easier for you to reproduce?": {
        "clear_documentation": "Clarity of instructions to run the code (not necessarily related to giving exact commands or explaining arguments of the code). A common compliment is when step-by-step instructions say what to expect before and after a certain step.",
        "example_commands_scripts": "This includes efforts to make running the code very user-friendly by providing example commands or scripts for running the code.",
        "dependencies": "Specification of exact dependencies the authors used to run the experiments.",
        "code_clarity": "Includes code comments, variable names, cleanliness of code, and intuitive organization of files.",
        "argument_documentation": "Documentation specifically for command-line arguments for the code. This may be combined into a 'code clarity' issue.",
        "hyperparameter_configuration": "Provision and/or explanation of correct hyperparameter configuration to reproduce the results in the paper, including random seeds.",
        "downloading_preprocessing_ease": "Efforts to make access and preprocessing of data and model files easier.",
        "commit_history": "Availability of past commits/versions of a code base.",
        "other_unclear": "Something positive mentioned but it's unclear what it is."
    },
    "What could the authors have provided or done better for you to reproduce the results faster or with less frustration?": {
        "better_data_access": "Correcting some issue or limitation in the data format, availability, preprocessing, or usage, including broken links and text encoding problems.",
        "dependencies": "Clearer specification of dependencies to run the code.",
        "complete_evaluation": "Code does not output the final result reported in the paper, and needs manual intervention to get there.",
        "better_documentation": "Better documentation for running the code, including how each part corresponds to results in the paper, how to handle expected errors, and correction of typos.",
        "debug_code": "The code has bugs that need to be fixed.",
        "hyperparameter_configuration": "Better documentation of the hyperparameter configuration needed to reproduce results.",
        "code_clarity": "Better code comments, structure, etc.",
        "hardware_requirements": "Specification of required compute, memory, etc.",
        "example_commands": "Wanting more efforts to make running the code very user-friendly by providing example commands or scripts for running the code.",
        "expected_runtime": "Expected time for the code to run, including progress tracking during running. May be lumped in with hardware_requirements.",
        "argument_documentation": "Documentation specifically for command-line arguments for the code. This may be combined into a 'code clarity' issue.",
        "commit_history": "Clearer commit messages to better track changes of code.",
        "other_unclear": "An improvement suggested, but it's unclear what it is.",
    },
    "Is there anything you would add to this checklist that you wish authors would provide to help you reproduce results faster or with less frustration?\n\nSuggest up to 5 additional items for the checklist.":
    {
        "example_commands": "Provide an example command or script to run the preprocessing, training, and evaluation from start to finish without user intervention.",
        "detailed_instructions": "Provide step-by-step instructions to reproduce the results, including the purpose of each file and what to expect before and after running it. May also include emphasis on more important instructions.",
        "changelog": "Provide a changelog of changes made after/beyond publication.",
        "faq": "Provide a list of frequent/expected issues in the code, and solutions.",
        "dataset_description": "Documentation of the dataset's format and how to interpret it.",
        "full_dependencies": "Fully specify dependencies by providing all working and tested versions of packages, or providing a container, e.g., Docker, or requirements file to install the required dependencies of the environment in one easy step (without any extra dependencies that may cause conflicts). Provide instructions to create the environment.",
        "interactive_demo": "Provide an interactive demo, notebook, or GUI for the proposed system.",
        "issue_forum": "Require that authors actively respond to issues with reproducing results, or provide a forum to discuss issues. Maybe combine with faq later.",
        "output_files": "Provide the actual system logs/output files that were obtained to get the results reported in the paper. Maybe combine this with detailed_instructions later.",
        "easy_configuration": "Provide an easy way to adjust all the constants in the code, including file paths and hyperparameters. Command-line arguments is a common way to do this.",
        "code_to_paper": "Documentation of how parts of the code correspond to approaches or results in the paper. Maybe combine with detailed_instructions or code_clarity later.",
        "sanity_check": "Require authors to demonstrate reproducibility from the released code base, e.g., to check for bugs, and document the date of the last successful test.",
        "backup_links": "Provide a secondary or backup link to linked resources, e.g., data or models.",
        "paper_specific": "Gave a suggestion that was too specific to the assigned paper.",
        "tensor_sizes": "Provide details about the dimensions of tensor inputs and outputs to systems. This may be combined with detailed_instructions.",
        "code_freeze": "Restrict modification of released code by authors.",
        "code_clarity": "Require code to have detailed comments and reasonable filenames, variable names, etc. as well as better explanation of errors that may be thrown by the code, e.g., assertion errors.",
        "cross_platform": "Provide instructions and/or results for code setup on multiple platforms, hardware, or dependency versions, including those the code wasn't originally run with.",
        "evaluation_code": "Code must include methods to generate the final evaluation metrics presented in the paper. This may be combined with example_commands.",
        "graph_results": "Require that all results are plotted on a graph.",
        "platform_specific": "Subject's suggestion was overly specific to the platform they ran the results on, e.g., providing the CUDA version.",
        "file_size": "Dataset or model size should be restricted to a certain size.",
        "video_tutorial": "Provide a video tutorial to reproduce the results.",
        "already_included": "Subject suggested something that's already on the ACLRC.",
    },
}

FREETEXT_ANNOTATION_TITLES = {
    "reproducibility_helpers": {
        "clear_documentation": "Docum.\nClarity",
        "example_commands_scripts": "Example\nScripts",
        "dependencies": "Depen-\ndencies",
        "code_clarity": "Code\nClarity",
        "argument_documentation": "Code\nClarity",
        "hyperparameter_configuration": "Hyper-\nparams",
        "downloading_preprocessing_ease": "Resource\nAccess",
        "commit_history": "Commit\nHistory",
        "other_unclear": "Other"
    },
    "reproducibility_blockers": {
        "better_data_access": "Resource\nAccess",
        "dependencies": "Depen-\ndencies",
        "complete_evaluation": "Unclear\nOutput",
        "better_documentation": "Docum.\nClarity",
        "debug_code": "Buggy\nCode",
        "hyperparameter_configuration": "Hyper-\nparams",
        "code_clarity": "Code\nClarity",
        "hardware_requirements": "Req.\nCompute",
        "example_commands": "Example\nScripts",
        "expected_runtime": "Req.\nCompute",
        "argument_documentation": "Code\nClarity",
        "commit_history": "Commit\nHistory",
        "other_unclear": "Other",
    },
    "aclrc_additions":
    {
        "example_commands": "Code\nDemo",
        "detailed_instructions": "Docum.\nClarity",
        "changelog": "Changelog",
        "faq": "Issue\nSupport",
        "dataset_description": "Data\nDocum.",
        "full_dependencies": "Dep.\nDetails",
        "interactive_demo": "Code\nDemo",
        "issue_forum": "Issue\nSupport",
        "output_files": "Docum.\nClarity",
        "easy_configuration": "Code\nClarity",
        "code_to_paper": "Docum.\nClarity",
        "sanity_check": "Sanity\nCheck",
        "backup_links": "Backup\nLinks",
        "paper_specific": "Paper\nSpecific",
        "tensor_sizes": "Docum.\nClarity",
        "code_freeze": "Changelog",
        "code_clarity": "Code\nClarity",
        "cross_platform": "Cross-\nPlatform",
        "evaluation_code": "Docum.\nClarity",
        "graph_results": "Graph\nResults",
        "platform_specific": "Platform\nSpecific",
        "file_size": "File\nSize",
        "video_tutorial": "Code\nDemo",
        "already_included": "Already\nIncluded",
    },
}
# Other: Paper specific, platform specific, appeared less than 3 times


ACLRC_ITEMS = ['A clear description of the mathematical setting, algorithm, and/or model', 
                   'A link to a downloadable source code, with specification of all dependencies, including external libraries (recommended for camera ready)',
                   'A description of computing infrastructure used', 
                   'The average runtime for each model or algorithm, or estimated energy cost', 
                   'The number of parameters in each model', 
                   'Corresponding validation performance for each reported test result', 
                   'A clear definition of the specific evaluation measure or statistics used to report results.', 
                   'The exact number of training and evaluation runs', 
                   'The bounds for each hyperparameter', 
                   'The hyperparameter configurations for best-performing models', 
                   'The method of choosing hyperparameter values (e.g., manual tuning, uniform sampling, etc.) and the criterion used to select among them (e.g., accuracy)', 
                   'Summary statistics of the results (e.g., mean, variance, error bars, etc.)', 
                   'Relevant statistics such as number of examples and label distributions', 
                   'Details of train/validation/test splits', 
                   'An explanation of any data that were excluded, and all pre-processing steps', 
                   'For natural language data, the name of the language(s)', 
                   'A link to a downloadable version of the dataset or simulation environment', 
                   'For new data collected, a complete description of the data collection process, such as instructions to annotators and methods for quality control'] 
ACLRC_QUESTION = "Which of the following items from the ACL Reproducibility Checklist were especially \nhelpful for reproducing this paper's results?\n\n\n\nSelect all items that apply. You can read more about the checklist here."