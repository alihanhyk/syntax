import dill
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

rcParams['text.usetex'] = True

plt_markers = ['v', '^', 'o']
plt_colors = ['#4e79a7', '#59a14f', '#e15759']
plt_colorslight = ['#d1dde9', '#d4e8d1', '#f7d5d5']

N = 25
H = 400
h0, h1 = 0, H - 2 * N
hs = np.arange(h0, h1) + 2 * N + 1

###
hs = np.concatenate(([hs[0]], hs))
h1 = h1 + 1

###

fig0 = plt.figure(figsize=(4, 4*3/4))
fig1 = plt.figure(figsize=(4, 4*3/4))

algs = ['banada', None, 'synada']
labels = ['Thresholding bandits', None, '\\textsc{Syntax}']

for alg, label, marker, color, colorlight in zip(algs, labels, plt_markers, plt_colors, plt_colorslight):
    if alg == None:
        continue

    with open(f'res/dim-{alg}.obj', 'rb') as file:
        res, _, truth = dill.load(file)
        res = (res > 0.)
        truth = truth[:,:,None,:].astype(bool)

    FPR = np.mean((res & ~truth).sum(-1) / (~truth).sum(-1), axis=1)
    TPR = np.mean((res & truth).sum(-1) / (truth).sum(-1), axis=1)

    FPR_m, FPR_s = FPR.mean(axis=0), FPR.std(axis=0)
    TPR_m, TPR_s = TPR.mean(axis=0), TPR.std(axis=0)
    FPR_m, FPR_s = np.concatenate(([FPR_m[0]], FPR_m)), np.concatenate(([FPR_s[0]], FPR_s))
    TPR_m, TPR_s = np.concatenate(([TPR_m[0]], TPR_m)), np.concatenate(([TPR_s[0]], TPR_s))

    fig0.gca().fill_between(hs, FPR_m[h0:h1]-FPR_s[h0:h1], FPR_m[h0:h1]+FPR_s[h0:h1], color=colorlight)
    fig0.gca().plot(hs, FPR_m[h0:h1], color=color, label=label, marker=marker, markevery=50)

    fig1.gca().fill_between(hs, TPR_m[h0:h1]-TPR_s[h0:h1], TPR_m[h0:h1]+TPR_s[h0:h1], color=colorlight)
    fig1.gca().plot(hs, TPR_m[h0:h1], color=color, label=label, marker=marker, markevery=50)

fig0.gca().set_xlabel('Number of Samples ($H$)')
fig0.gca().set_ylabel('False Positive Rate (FPR)')
fig0.gca().legend(loc='upper right')

fig1.gca().set_xlabel('Number of Samples ($H$)')
fig1.gca().set_ylabel('True Positive Rate (TPR)')
fig1.gca().legend(loc='lower right')

fig0.tight_layout()
fig0.savefig('fig/adaptive-fpr.pdf')

fig1.tight_layout()
fig1.savefig('fig/adaptive-tpr.pdf')

###

fig0 = plt.figure(figsize=(4, 4*3/4))
fig1 = plt.figure(figsize=(4, 4*3/4))

algs = ['synuni', 'synsta', 'synada']
labels = ['Synthetic study', 'Synthetic design', '\\textsc{Syntax}']

for alg, label, marker, color, colorlight in zip(algs, labels, plt_markers, plt_colors, plt_colorslight):
    if alg == None:
        continue

    with open(f'res/dim-{alg}.obj', 'rb') as file:
        res, _, truth = dill.load(file)
        res = (res > 0.)
        truth = truth[:,:,None,:].astype(bool)

    FPR = np.mean((res & ~truth).sum(-1) / (~truth).sum(-1), axis=1)
    TPR = np.mean((res & truth).sum(-1) / (truth).sum(-1), axis=1)

    FPR_m, FPR_s = FPR.mean(axis=0), FPR.std(axis=0)
    TPR_m, TPR_s = TPR.mean(axis=0), TPR.std(axis=0)
    FPR_m, FPR_s = np.concatenate(([FPR_m[0]], FPR_m)), np.concatenate(([FPR_s[0]], FPR_s))
    TPR_m, TPR_s = np.concatenate(([TPR_m[0]], TPR_m)), np.concatenate(([TPR_s[0]], TPR_s))

    fig0.gca().fill_between(hs, FPR_m[h0:h1]-FPR_s[h0:h1], FPR_m[h0:h1]+FPR_s[h0:h1], color=colorlight)
    fig0.gca().plot(hs, FPR_m[h0:h1], color=color, label=label, marker=marker, markevery=50)

    fig1.gca().fill_between(hs, TPR_m[h0:h1]-TPR_s[h0:h1], TPR_m[h0:h1]+TPR_s[h0:h1], color=colorlight)
    fig1.gca().plot(hs, TPR_m[h0:h1], color=color, label=label, marker=marker, markevery=50)

fig0.gca().set_xlabel('Number of Samples ($H$)')
fig0.gca().set_ylabel('False Positive Rate (FPR)')
fig0.gca().legend(loc='upper right')

fig1.gca().set_xlabel('Number of Samples ($H$)')
fig1.gca().set_ylabel('True Positive Rate (TPR)')
fig1.gca().legend(loc='lower right')

fig0.tight_layout()
fig0.savefig('fig/synthetic-fpr.pdf')

fig1.tight_layout()
fig1.savefig('fig/synthetic-tpr.pdf')

###

plt.figure(figsize=(4,4*3/4))

algs = ['banuni', 'banada', 'synada']
labels = ['Conventional study', 'Thresholding bandits', '\\textsc{Syntax}']

for alg, label, marker, color, colorlight in zip(algs, labels, plt_markers, plt_colors, plt_colorslight):
    if alg == None:
        continue

    with open(f'res/dim-{alg}.obj', 'rb') as file:
        _, res_ns, _ = dill.load(file)
        res_ns = res_ns.sum(axis=-2)

    res = np.mean(res_ns[...,1] / (res_ns[...,0]+res_ns[...,1]), axis=1)
    res_m = res.mean(axis=0)
    res_s = res.std(axis=0)

    if alg == 'banuni':
        res_m = np.ones((H,)) * .5
        res_s = np.zeros((H,))

    res_m = np.concatenate(([res_m[0]], res_m))
    res_s = np.concatenate(([res_s[0]], res_s))

    plt.fill_between(hs, res_m[h0:h1]-res_s[h0:h1], res_m[h0:h1]+res_s[h0:h1], color=colorlight)
    plt.plot(hs, res_m[h0:h1], color=color, label=label, marker=marker, markevery=50)

plt.xlabel('Number of Samples ($H$)')
plt.ylabel('Proportion of Samples Allocated\nto the Treatment Group ($n^{(1)}/n$)')
plt.legend(loc='right')

plt.tight_layout()
plt.savefig('fig/ratio.pdf')
