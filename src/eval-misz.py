import dill
import numpy as np
import pandas as pd

N = 25
H200 = 200 - 2 * N
H400 = 400 - 2 * N

algs = list()
algs.append(('banuni', 'Conventional study'))
algs.append(('banada', 'Thresholding bandits'))
algs.append(('synuni', 'Synthetic study'))
algs.append(('synsta', 'Synthetic design'))
algs.append(('synada', 'SYNTAX'))

metrics = pd.MultiIndex.from_product([
    ['H = 200', 'H = 400'],
    ['FPR', 'TPR']])

df = pd.DataFrame(index=[alg_name for _, alg_name in algs], columns=metrics)

for alg_tag, alg in algs:

    with open(f'res/misz-{alg_tag}.obj', 'rb') as file:
        res, _, truth = dill.load(file)
        h200 = (res[:,:,H200-1,:] > 0.)
        h400 = (res[:,:,H400-1,:] > 0.)
        truth = truth.astype(bool)

    FPR_h200 = np.mean((h200 & ~truth).sum(-1) / (~truth).sum(-1), axis=-1)
    FPR_h400 = np.mean((h400 & ~truth).sum(-1) / (~truth).sum(-1), axis=-1)
    TPR_h200 = np.mean((h200 & truth).sum(-1) / (truth).sum(-1), axis=-1)
    TPR_h400 = np.mean((h400 & truth).sum(-1) / (truth).sum(-1), axis=-1)

    df.loc[alg, ("H = 200", "FPR")] = f"{FPR_h200.mean()*100:.1f}% ({FPR_h200.std()*100:.1f}%)"
    df.loc[alg, ("H = 200", "TPR")] = f"{TPR_h200.mean()*100:.1f}% ({TPR_h200.std()*100:.1f}%)"
    df.loc[alg, ("H = 400", "FPR")] = f"{FPR_h400.mean()*100:.1f}% ({FPR_h400.std()*100:.1f}%)"
    df.loc[alg, ("H = 400", "TPR")] = f"{TPR_h400.mean()*100:.1f}% ({TPR_h400.std()*100:.1f}%)"

print(df)
