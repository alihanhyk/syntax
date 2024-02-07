import dill
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

rcParams['text.usetex'] = True

plt_markers = ['v', '^', 'o']
plt_colors = ['#4e79a7', '#59a14f', '#e15759']
plt_colorslight = ['#d1dde9', '#d4e8d1', '#f7d5d5']

lambdas = np.logspace(-3, 2, 10)

N = 25
for tag, H in zip(['h200', 'h400'], [200 - 2 * N, 400 - 2 * N]):

    fprs_m = list()
    fprs_s = list()
    tprs_m = list()
    tprs_s = list()

    for i in range(10):

        with open(f'res/syntax-lambda{i}.obj', 'rb') as file:
            res, _, truth = dill.load(file)
            res = (res[:,:,H-1,:] > 0.)
            truth = truth.astype(bool)


        FPR = np.mean((res & ~truth).sum(-1) / (~truth).sum(-1), axis=1)
        TPR = np.mean((res & truth).sum(-1) / (truth).sum(-1), axis=1)

        fprs_m.append(FPR.mean())
        fprs_s.append(FPR.std())
        tprs_m.append(TPR.mean())
        tprs_s.append(TPR.std())

    fprs_m, fprs_s = np.array(fprs_m), np.array(fprs_s)
    tprs_m, tprs_s = np.array(tprs_m), np.array(tprs_s)

    fig0 = plt.figure(figsize=(4, 4*3/4))
    fig1 = plt.figure(figsize=(4, 4*3/4))

    if tag == 'h200':
        fig0.gca().fill_between(lambdas, (0.195 - 0.002) * np.ones(10), (0.195 + 0.002) * np.ones(1), color=plt_colorslight[0])
        fig0.gca().plot(lambdas, 0.195 * np.ones(10), color=plt_colors[0], marker=plt_markers[0], label='Conventional study')
        fig0.gca().fill_between(lambdas, (0.176 - 0.004) * np.ones(10), (0.176 + 0.004) * np.ones(1), color=plt_colorslight[1])
        fig0.gca().plot(lambdas, 0.176 * np.ones(10), color=plt_colors[1], marker=plt_markers[1], label='Thresholding bandits')

    if tag == 'h400':
        fig0.gca().fill_between(lambdas, (0.149 - 0.003) * np.ones(10), (0.149 + 0.003) * np.ones(1), color=plt_colorslight[0])
        fig0.gca().plot(lambdas, 0.149 * np.ones(10), color=plt_colors[0], marker=plt_markers[0], label='Conventional study')
        fig0.gca().fill_between(lambdas, (0.137 - 0.004) * np.ones(10), (0.137 + 0.004) * np.ones(1), color=plt_colorslight[1])
        fig0.gca().plot(lambdas, 0.137 * np.ones(10), color=plt_colors[1], marker=plt_markers[1], label='Thresholding bandits')

    fig0.gca().fill_between(lambdas, fprs_m-fprs_s, fprs_m+fprs_s, color=plt_colorslight[2])
    fig0.gca().plot(lambdas, fprs_m, color=plt_colors[2], marker=plt_markers[2], label='\\textsc{Syntax}')

    if tag == 'h200':
        fig1.gca().fill_between(lambdas, (0.807 - 0.003) * np.ones(10), (0.807 + 0.003) * np.ones(1), color=plt_colorslight[0])
        fig1.gca().plot(lambdas, 0.807 * np.ones(10), color=plt_colors[0], marker=plt_markers[0], label='Conventional study')
        fig1.gca().fill_between(lambdas, (0.826 - 0.004) * np.ones(10), (0.826 + 0.004) * np.ones(1), color=plt_colorslight[1])
        fig1.gca().plot(lambdas, 0.826 * np.ones(10), color=plt_colors[1], marker=plt_markers[1], label='Thresholding bandits')

    if tag == 'h400':
        fig1.gca().fill_between(lambdas, (0.854 - 0.003) * np.ones(10), (0.854 + 0.003) * np.ones(1), color=plt_colorslight[0])
        fig1.gca().plot(lambdas, 0.854 * np.ones(10), color=plt_colors[0], marker=plt_markers[0], label='Conventional study')
        fig1.gca().fill_between(lambdas, (0.864 - 0.002) * np.ones(10), (0.864 + 0.002) * np.ones(1), color=plt_colorslight[1])
        fig1.gca().plot(lambdas, 0.864 * np.ones(10), color=plt_colors[1], marker=plt_markers[1], label='Thresholding bandits')

    fig1.gca().fill_between(lambdas, tprs_m-tprs_s, tprs_m+tprs_s, color=plt_colorslight[2])
    fig1.gca().plot(lambdas, tprs_m, color=plt_colors[2], marker=plt_markers[2], label='\\textsc{Syntax}')

    fig0.gca().set_xlabel('Factor Effect Parameter ($\\lambda$)')
    fig0.gca().set_ylabel('False Positive Rate (FPR)')
    fig0.gca().set_xscale('log')
    fig0.gca().legend(loc='upper left')

    fig1.gca().set_xlabel('Factor Effect Parameter ($\\lambda$)')
    fig1.gca().set_ylabel('True Positive Rate (TPR)')
    fig1.gca().set_xscale('log')
    fig1.gca().legend(loc='lower left')

    fig0.tight_layout()
    fig0.savefig(f'fig/sensitivity-{tag}-fpr.pdf')

    fig1.tight_layout()
    fig1.savefig(f'fig/sensitivity-{tag}-tpr.pdf')
