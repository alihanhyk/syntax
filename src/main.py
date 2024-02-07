import dill
import jax
import jax.numpy as np
import tqdm

from algs import infer_bandit, infer_synthetic
from algs import recruit_uniform, recruit_static, recruit_adaptive

from envs import make, simulate
from envs import env_params_diminishing, env_params_increasing
from envs import env_params_mismatchx, env_params_mismatchz

jax.config.update('jax_platform_name', 'cpu')

envs = list()
envs.append(('dim', env_params_diminishing))
envs.append(('inc', env_params_increasing))
envs.append(('misx', env_params_mismatchx))
envs.append(('misz', env_params_mismatchz))

algs = list()
algs.append(('banuni', {'infer': infer_bandit, 'recruit': recruit_uniform}))
algs.append(('banada', {'infer': infer_bandit, 'recruit': recruit_adaptive}))
algs.append(('synuni', {'infer': infer_synthetic, 'recruit': recruit_uniform}))
algs.append(('synsta', {'infer': infer_synthetic, 'recruit': recruit_static}))
algs.append(('synada', {'infer': infer_synthetic, 'recruit': recruit_adaptive}))

N = 25
H = 400 - 2 * N
iter0 = 10
iter1 = 1000

for env_tag, env_params in envs:
    for alg_tag, alg in algs:
    
        print(f'simulating {alg_tag} in environment {env_tag} ...')

        key = jax.random.PRNGKey(0)
        res = np.zeros((iter0, iter1, H, N))
        res_ns = np.zeros((iter0, iter1, H, N, 2))
        truth = np.zeros((iter0, iter1, N))

        for i in tqdm.tqdm(range(iter0)):
            key, subkey = jax.random.split(key)

            for j in tqdm.tqdm(range(iter1), leave=False):
                
                subkey, subsubkey = jax.random.split(subkey)
                env, env_info = make(subsubkey, env_params)

                _, _, zs_bar, _ = env
                truth = truth.at[i,j].set(zs_bar[:,1] - zs_bar[:,0] > 0.)
                
                ###
                alg['infer_params'] = dict()
                alg['infer_params']['lambda'] = env_info['lambda']

                subkey, subsubkey = jax.random.split(subkey)
                _res, _res_ns = simulate(subsubkey, env, alg, H)
                
                res = res.at[i,j].set(_res)
                res_ns = res_ns.at[i,j].set(_res_ns)

        ###
        with open(f'res/{env_tag}-{alg_tag}.obj', 'wb') as file:
            dill.dump((res, res_ns, truth), file)
