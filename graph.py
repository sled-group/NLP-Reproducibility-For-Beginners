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
from typing import Any, Union

@dataclass
class GraphData:
    x: list[float]
    y: Union[list[float], list[list[float]], list[list[list[float]]]]
    x_label: str = None
    y_label: Union[str, list[str]] = None

    x_range: tuple[float, float] = None
    y_range: tuple[float, float] = None
    x_ticks: list[float] = None
    y_ticks: list[float] = None

    title: str = None

    point_categories: Union[list[str], list[list[str]]] = None
    category_names: list[str] = None
    
    add_fitline: bool = False
    add_error_bars: bool = False
    add_std_bars: bool = False

    font_size: int = 18
    figure_size: tuple[int,int] = (1,1)
    colors: list[str] = None

def scatter_plot_2d(graph_data: GraphData, save_path: str):
    x, y = graph_data.x, graph_data.y
    xlabel, ylabel = graph_data.x_label, graph_data.y_label
    point_categories = graph_data.point_categories
    title = graph_data.title
    add_line = graph_data.add_fitline
    xrange, yrange = graph_data.x_range, graph_data.y_range
    colors = graph_data.colors
    xticks, yticks = graph_data.x_ticks, graph_data.y_ticks

    plt.clf()
    figure(figsize=(8, 4))
    if point_categories is not None:
        for category in set(point_categories):
            plt.scatter([xp for xp, c in zip(x, point_categories) if c == category],
                        [yp for yp, c in zip(y, point_categories) if c == category],
                        label=category, c=[cl for cl, c in zip(colors, point_categories) if c == category])
    else:
        plt.scatter(x, y, label=point_categories, c=colors)

    if add_line:
        x = np.array(x)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, color='gray')

    font_size = graph_data.font_size
    plt.xlabel(xlabel, fontsize=font_size, fontweight='bold')
    plt.ylabel(ylabel, fontsize=font_size, fontweight='bold')
    if xrange is not None:
        plt.xlim(xrange[0], xrange[1])
    if yrange is not None:
        plt.ylim(yrange[0], yrange[1])
    if title is not None:
        plt.title(title, fontsize=font_size, fontweight='bold')
    if point_categories is not None:
        plt.legend(fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9) # NOTE: this is not used for all plots in the paper - just hacked for some camera ready figures. May need to remove in future debugging
    plt.savefig(save_path)            

def grouped_mean_bar_chart(graph_data: GraphData,  save_path: str):
    plt.clf()
    fig, ax = plt.subplots()
    x = graph_data.x
    y = graph_data.y

    means = np.array([[np.mean(sl) for sl in l] for l in y])
    for gi, group in enumerate(graph_data.category_names):
        width = 0.2  # the width of the bars
        group_y = means[:, gi]
        if graph_data.add_std_bars:
            group_err = [np.std(sub_data[gi]) for sub_data in y]
        elif graph_data.add_error_bars:
            group_err = [sem(sub_data[gi]) for sub_data in y]
        else:
            group_err = None
        group_rects = ax.bar(x-width + gi * width, group_y, width, yerr=group_err, label=graph_data.category_names[gi], color=graph_data.colors[gi])

    ax.set_ylabel(graph_data.y_label, fontsize=graph_data.font_size, fontweight='bold')
    if graph_data.title is not None:
        ax.set_title(graph_data.title)
    ax.set_xticks(graph_data.x)
    ax.set_xticklabels(graph_data.x_label, fontsize=graph_data.font_size, fontweight='bold')
    if graph_data.y_ticks is not None:
        ax.set_yticks(graph_data.y_ticks)
    plt.yticks(fontsize=graph_data.font_size)
    ax.legend(ncol=3, fontsize=graph_data.font_size)
    if graph_data.y_range is not None:
        plt.ylim(*graph_data.y_range)

    fig.tight_layout()
    if (graph_data.figure_size) != (1,1):
        width, height = graph_data.figure_size
        fig.set_figwidth(width)
        fig.set_figheight(height)

    plt.savefig(save_path, bbox_inches="tight")            

PYPLOT_COLOR_MAPPINGS = {
    "blue": "b",
    "green": "g",
    "red": "r",
    "cyan": "c",
    "magenta": "m",
    "yellow": "y",
    "black": "k",
    "white": "w"
}
def box_plot(graph_data: GraphData, save_path: str):
    fig, ax = plt.subplots()
    labels = graph_data.x_label
    x = graph_data.x  # the label locations
    group_y = graph_data.y # y should be grouped into len(x_label) groups

    group_rects = ax.boxplot(group_y, sym=PYPLOT_COLOR_MAPPINGS[graph_data.colors[0]] + 'o', whis=1.5, widths=0.85, medianprops={ 'linestyle': '--', 'color': graph_data.colors[0]})

    ax.set_ylabel(graph_data.y_label, fontsize=graph_data.font_size, fontweight='bold')

    ax.set_xticklabels(labels, fontsize=graph_data.font_size, fontweight='bold')
    plt.yticks(fontsize=graph_data.font_size)
    if graph_data.y_range is not None:
        plt.ylim(*graph_data.y_range)

    if (graph_data.figure_size) != (1,1):
        width, height = graph_data.figure_size
        fig.set_figwidth(width)
        fig.set_figheight(height)
        # plt.subplots_adjust(left=0.07*width, bottom=0.01*height)
    plt.savefig(save_path, bbox_inches="tight")

def stacked_count_bar_chart(graph_data: GraphData,  save_path: str, ungroup=False):
    """Generate and save a bar chart of some grouped counts.
    :param ungroup: Set to true to undo groupings and present results in aggregate."""
    plt.clf()
    fig, ax = plt.subplots()
    x = graph_data.x
    y = np.array(graph_data.y)
    labels = graph_data.x_label

    if not ungroup:
        # For each grouping, stack another segment on top of all bars
        group_last = None
        for gi in range(y.shape[1]): 
            group_y = y[:, gi]
            width = 0.35  # the width of the bars
            group_rects = ax.bar(x, group_y, width, label=graph_data.category_names[gi], bottom=group_last, color=graph_data.colors[gi])
            if group_last is None:
                group_last = group_y
            else:
                group_last += group_y
    else:
        group_y = np.sum(y, axis=1)
        width = 0.35  # the width of the bars
        group_rects = ax.bar(x, group_y, width,  color=graph_data.colors[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    font_size = graph_data.font_size
    ax.bar_label(group_rects, padding=3, fontsize=font_size, fontweight='bold')
    ax.set_ylabel(graph_data.y_label, fontsize=font_size, fontweight='bold')
    

    plt.ylim(*graph_data.y_range)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=font_size, fontweight='bold')
    plt.yticks(fontsize = font_size)
    if not ungroup:
        ax.legend(ncol=3, fontsize=font_size)
    if (graph_data.figure_size) != (1,1):
        width, height = graph_data.figure_size
        fig.set_figwidth(width)
        fig.set_figheight(height)    
    plt.savefig(save_path, bbox_inches="tight")            
