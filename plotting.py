import pickle
import time

# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
#a,b,c,d
#base, optonly, colonly, opt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10


#############################################
### Example Usage(ablation study of beta) ###
#############################################
#load the results from the pickle files
# TODO: change the file names to the correct ones

# with open('xxx.pkl', 'rb') as f:
#     basecifaralpha05 = pickle.load(f)

# with open('xxx.pkl', 'rb') as f:
#     b01finpcifarbest = pickle.load(f)

# with open('xxx.pkl', 'rb') as f:
#     b005finpcifar = pickle.load(f)

# with open('xxx.pkl', 'rb') as f:
#     b03finpcifar = pickle.load(f)

# with open('xxx.pkl', 'rb') as f:
#     b05finpcifar = pickle.load(f)






def calculate_cv(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    cv = (std_dev / mean) # * 100
    return cv


def get_eod(data):

    grouped_by_index = list(zip(*data.values()))
    max_differences = [round(max(values) - min(values),3) for values in grouped_by_index]

    return max_differences


def line_eod(*series, labels=None):

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    rounds = np.arange(1, len(list(series[0].values())[0]) + 1)
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    print('EOD')

    for i, (s, label) in enumerate(zip(series, labels)):
        print(f'{labels[i]} max: {max(get_eod(series[i]))} mean: {round(float(np.mean(get_eod(series[i]))), 2)} '
              f'last 5 rounds mean: {round(float(np.mean(get_eod(series[i])[-5:])), 2)}')

        ax.plot(rounds, get_eod(series[i]), label=label)
        ax.axhline(y=np.mean(get_eod(series[i])), color=colors[i % len(colors)], linestyle=':')

    ax.set_xlabel('Rounds')
    ax.set_ylabel('EOD')
    # ax.set_ylim(0, 80)
    ax.set_xticks(np.arange(0, len(list(series[0].values())[0]) + 1))
    ax.set_title('EOD across Rounds')


    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.savefig('line_EOD.png')

    print('\n')
    plt.close(fig)


def line_average_train_accuracy(*series, labels=None):
    """
    Plot multiple training accuracy curves.

    Parameters:
    - series: any number of lists/arrays of values (e.g., a, d, b, c, ...)
    - labels: list of label names (must match number of series)
    """

    rounds = np.arange(1, len(series[0]) + 1)  # assume all series are the same length

    fig, ax = plt.subplots()

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    for i, (data, label) in enumerate(zip(series, labels)):
        linestyle = '--' if i == 0 else '-'
        ax.plot(rounds, data, label=label, linestyle=linestyle)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Train Accuracy (%)')
    ax.set_ylim(0, 85)
    ax.set_xticks(np.arange(0, len(series[0])+1, 1))

    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')

    fig.savefig('line_average_train_accuracy.png')
    print('train accuracy')
    print(f'baseline accuracy: {np.mean(series[0][-5:]):.2f}')
    for i, (data, label) in enumerate(zip(series, labels)):
        print(f'{labels[0]} -> {labels[i]} = {np.mean(series[0][-5:]):.2f} -> {np.mean(series[i][-5:]):.2f}')
        print(f'delta = {(np.mean(series[i][-5:])- np.mean(series[0][-5:])):.2f} '
              f'percentage: {((np.mean(series[i][-5:])- np.mean(series[0][-5:]))/np.mean(series[0][-5:])):.2f}')
    print('\n')
    plt.close(fig)



def line_average_test_accuracy(*series, labels=None):
    """
    Plot multiple training accuracy curves.

    Parameters:
    - series: any number of lists/arrays of values (e.g., a, d, b, c, ...)
    - labels: list of label names (must match number of series)
    """

    rounds = np.arange(1, len(series[0]) + 1)  # assume all series are the same length

    fig, ax = plt.subplots()

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    for i, (data, label) in enumerate(zip(series, labels)):
        linestyle = '--' if i == 0 else '-'
        ax.plot(rounds, data, label=label, linestyle=linestyle)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 85)
    ax.set_xticks(np.arange(0, len(series[0])+1, 1))
    ax.legend(loc='lower right')

    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')

    fig.savefig('line_average_test_accuracy.png')
    print('Test accuracy')
    print(f'baseline accuracy: {np.mean(series[0][-5:]):.2f}')
    for i, (data, label) in enumerate(zip(series, labels)):
        print(f'{labels[0]} -> {labels[i]} = {np.mean(series[0][-5:]):.2f} -> {np.mean(series[i][-5:]):.2f}')
        print(f'delta = {(np.mean(series[i][-5:])- np.mean(series[0][-5:])):.2f} '
              f'percentage: {((np.mean(series[i][-5:])- np.mean(series[0][-5:]))/np.mean(series[0][-5:])):.2f}')
    print('\n')
    plt.close(fig)


def loss_CoV(*series, labels=None):
    """
    Plot Coefficient of Variation (CoV) of loss across rounds for multiple methods.

    Parameters:
    - *series: list of 2D arrays/lists (shape: [rounds][clients])
    - labels: list of label names (must match number of series)
    """

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    rounds = np.arange(1, len(series[0]) + 1)  # assume all series same length
    cov_results = [[] for _ in series]  # one list per series

    for i in range(len(series[0])):  # for each round
        for idx, method in enumerate(series):
            total_cv = 0
            for j in range(len(method[0])):  # for each client
                total_cv += calculate_cv(method[i][j])
            avg_cv = total_cv / len(method[0])
            cov_results[idx].append(avg_cv)

    # Plotting
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (cov, label) in enumerate(zip(cov_results, labels)):
        ax.plot(rounds, cov, label=label, color=colors[i % len(colors)])
        ax.axhline(y=np.mean(cov), color=colors[i % len(colors)], linestyle=':')

    ax.set_xlabel('Rounds')
    ax.set_ylabel('CoV(Loss)')
    ax.legend(loc='lower right')
    ax.set_xticks(np.arange(0, len(rounds)+1, 1))
    ax.set_ylim(0.08, 0.58)

    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')

    fig.savefig('line_loss_CV.png')

    # Output CoV stats
    print('loss cv')
    print(', '.join([f'{np.mean(c):.2f}' for c in cov_results]))
    print('\n')
    plt.close(fig)


def loss_fi(*series, labels=None):
    """
    Plot Fairness Index (FI) of loss across rounds for multiple methods.

    Parameters:
    - *series: any number of 2D arrays/lists (shape: [rounds][clients])
    - labels: list of label names (must match number of series)
    """

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    rounds = np.arange(1, len(series[0]) + 1)
    fi_results = [[] for _ in series]  # one list per method

    for i in range(len(series[0])):  # for each round
        for idx, method in enumerate(series):
            total_cv = 0
            for j in range(len(method[0])):  # each client
                total_cv += calculate_cv(method[i][j])
            avg_cv = total_cv / len(method[0])
            fi = 1 / (1 + np.square(avg_cv))
            fi_results[idx].append(fi)

    # Plotting
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (fi, label) in enumerate(zip(fi_results, labels)):
        ax.plot(rounds, fi, label=label, color=colors[i % len(colors)])
        ax.axhline(y=np.mean(fi), color=colors[i % len(colors)], linestyle=':')

    ax.set_xlabel('Rounds')
    ax.set_ylabel('FI(Loss)')
    ax.set_xticks(np.arange(0, len(rounds)+1, 1))
    ax.set_ylim(0.74, 1.01)

    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')

    fig.savefig('line_loss_FI.png')

    # Print mean FI values
    print('loss fi')
    print(', '.join([f'{np.mean(f):.2f}' for f in fi_results]))
    print('\n')
    plt.close(fig)


def siacvplot(*series, labels=None, mode='CoV(SIA)'):
    """
    Plot Coefficient of Variation (CoV) across rounds.

    Parameters:
    - *series: accuracy/loss values over rounds, each a list or dict (if mode == 'CoV(SIA)')
    - labels: list of label names (must match number of series)
    - mode: 'CoV(SIA)' or 'CoV(Client)' to control input structure
    """

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    # Input transformation based on mode
    if mode == 'CoV(SIA)':
        rounds = np.arange(1, len(list(series[0].values())[0])+1)
        transformed_series = [np.array(list(s.values())).T for s in series]
        # print('EOD')
        # print(f'baseline max: {max(get_eod(series[0]))} mean: {round(float(np.mean(get_eod(series[0]))), 3)}')
        # print(f'finp max: {max(get_eod(series[-1]))} mean: {round(float(np.mean(get_eod(series[-1]))), 3)}')
    else:
        rounds = np.arange(1, len(series[0])+1)
        transformed_series = series

    # Calculate CoV for each round
    cov_results = []
    for data in transformed_series:
        cov = [calculate_cv(row) for row in data]
        cov_results.append(cov)

    # Plotting
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (cov, label) in enumerate(zip(cov_results, labels)):
        ax.plot(rounds, cov, label=label, color=colors[i % len(colors)])
        ax.axhline(y=np.mean(cov), color=colors[i % len(colors)], linestyle=':')

    ax.set_xlabel('Rounds')
    ax.set_ylabel(mode)
    ax.legend(loc='upper right')
    ax.set_xticks(np.arange(0, len(rounds)+1, 1))
    ax.set_ylim(0.08, 0.58)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')

    fig.savefig(f'line_{mode}.png')

    # Print summary
    print(mode)
    print(', '.join([f'{np.mean(c):.2f}' for c in cov_results]))
    print('\n')
    plt.close(fig)


def siafiplot(*series, labels=None, mode='FI(SIA)'):
    """
    Plot FI(SIA) or FI(Client) across rounds for multiple methods.

    Parameters:
    - *series: each can be a list or a dict (if mode == 'FI(SIA)')
    - labels: list of label names (must match number of series)
    - mode: 'FI(SIA)' or 'FI(Client)' to determine data format
    """

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    if mode == 'FI(SIA)':
        rounds = np.arange(1, len(list(series[0].values())[0])+1)
        transformed_series = [np.array(list(s.values())).T for s in series]
    else:
        rounds = np.arange(1, len(series[0]) + 1)
        transformed_series = series

    fi_results = []
    for data in transformed_series:
        fi = [1 / (1 + np.square(calculate_cv(row))) for row in data]
        fi_results.append(fi)

    # Plotting
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (fi, label) in enumerate(zip(fi_results, labels)):
        ax.plot(rounds, fi, label=label, color=colors[i % len(colors)])
        ax.axhline(y=np.mean(fi), color=colors[i % len(colors)], linestyle=':')

    ax.set_xlabel('Rounds')
    ax.set_ylabel(mode)

    # Dynamic Y-axis range
    # means = [np.mean(fi) for fi in fi_results]
    # if all(m > 0.5 for m in means):
    #     ax.set_ylim(0.4, 1)
    # else:
    #     ax.set_ylim(0, 1)

    ax.legend(loc='upper right')
    ax.set_xticks(np.arange(0, len(rounds)+1, 1))
    ax.legend(loc='upper right')

    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')

    fig.savefig(f'line_{mode}.png')

    # Print summary
    print(mode)
    print(', '.join([f'{np.mean(fi):.2f}' for fi in fi_results]))
    print('\n')
    plt.close(fig)


def sia_rounds(*series, labels=None):
    """
    Plot SIA accuracy over rounds for multiple methods.

    Parameters:
    - *series: any number of accuracy series (e.g., a, b, c, d)
    - labels: list of label names (same length as series)
    """

    if labels is None or len(labels) != len(series):
        raise ValueError("You must provide a label for each series.")

    rounds = np.arange(1, len(series[0]) + 1)
    fig, ax = plt.subplots()

    for i, (s, label) in enumerate(zip(series, labels)):
        ax.plot(rounds, s, label=label)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Average SIA Accuracy (%)')
    ax.set_ylim(5, 55)
    ax.set_title('SIA Accuracy across Rounds')
    ax.legend(loc='upper right')
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, len(series[0]) + 1))

    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')

    fig.savefig('line_SIA_Accuracy.png')

    # Print stats
    print('SIA PERFORMANCE')
    print('Baseline', r'$\beta$=0.05', r'$\beta$=0.1', r'$\beta$=0.3', r'$\beta$=0.5')
    mean_str = ', '.join([f'{np.mean(s):.2f}' for s in series])
    max_str = ', '.join([f'{np.max(s):.2f}' for s in series])
    print(f'mean sia: {mean_str}')
    print(f'max sia: {max_str}')
    print('\n')

    plt.close(fig)



#############################################
### Example Usage(ablation study of beta) ###
#############################################
# data structure:
# inside pickle: [sia_per_client, trainacc, testacc, loss, sia_average, time_per_round]

# set labels to all results pickle files
# LABEL = ['Baseline', r'$\beta$=0.05', r'$\beta$=0.1', r'$\beta$=0.3', r'$\beta$=0.5']
# LABEL1 = ['Baseline', r'$\beta$=0.05', r'$\beta$=0.1', r'$\beta$=0.3']

# line_average_train_accuracy(), line_average_test_accuracy() for training/testing accuracy figure.
# line_average_train_accuracy(basecifaralpha05[1], b005finpcifar[1], b01finpcifarbest[1], b03finpcifar[1],
#                              b05finpcifar[1], labels=LABEL)
# line_average_test_accuracy(basecifaralpha05[2], b005finpcifar[2], b01finpcifarbest[2], b03finpcifar[2],
#                               b05finpcifar[2], labels=LABEL)

# loss_CoV(), loss_fi() for CoV and FI of Loss figure.
# loss_CoV(basecifaralpha05[3], b005finpcifar[3], b01finpcifarbest[3], b03finpcifar[3],
#                               labels=LABEL1)
# loss_fi(basecifaralpha05[3], b005finpcifar[3], b01finpcifarbest[3], b03finpcifar[3],
#                               labels=LABEL1)

# siacvplot(), siafiplot() for CoV and FI of SIA figure.
# siacvplot(basecifaralpha05[0], b005finpcifar[0], b01finpcifarbest[0], b03finpcifar[0],
#                              labels=LABEL1, mode='CoV(SIA)')
# siafiplot(basecifaralpha05[0], b005finpcifar[0], b01finpcifarbest[0], b03finpcifar[0],
#                             labels=LABEL1, mode='FI(SIA)')

# sia_rounds() for average SIA accuracy across round figure.
# sia_rounds(basecifaralpha05[4], b005finpcifar[4], b01finpcifarbest[4], b03finpcifar[4],
#                              b05finpcifar[4], labels=LABEL)

# line_eod() for EOD figure.
# line_eod(basecifaralpha05[0], b005finpcifar[0], b01finpcifarbest[0], b03finpcifar[0],
#                              b05finpcifar[0], labels=LABEL)



