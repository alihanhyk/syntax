import dill
import jax
import jax.numpy as np
import tqdm

from algs import infer_synthetic
from algs import recruit_adaptive

from envs import make, simulate
from envs import env_params_diminishing

jax.config.update('jax_platform_name', 'cpu')

alg_syntax = {'infer': infer_synthetic, 'recruit': recruit_adaptive}

N = 25
H = 400 - 2 * N
iter0 = 10
iter1 = 1000
    
lambdas = np.logspace(-3, 2, 10)
for lambda_i, lambda_val in enumerate(lambdas):

        key = jax.random.PRNGKey(0)
        res = np.zeros((iter0, iter1, H, N))
        res_ns = np.zeros((iter0, iter1, H, N, 2))
        truth = np.zeros((iter0, iter1, N))

        for i in tqdm.tqdm(range(iter0)):
            key, subkey = jax.random.split(key)

            for j in tqdm.tqdm(range(iter1), leave=False):
                
                subkey, subsubkey = jax.random.split(subkey)
                env, env_info = make(subsubkey, env_params_diminishing)

                _, _, zs_bar, _ = env
                truth = truth.at[i,j].set(zs_bar[:,1] - zs_bar[:,0] > 0.)
                
                ###
                alg_syntax['infer_params'] = dict()
                alg_syntax['infer_params']['lambda'] = lambda_val

                subkey, subsubkey = jax.random.split(subkey)
                _res, _res_ns = simulate(subsubkey, env, alg_syntax, H)
                
                res = res.at[i,j].set(_res)
                res_ns = res_ns.at[i,j].set(_res_ns)

        ###
        with open(f'res/sens-lambda{lambda_i}.obj', 'wb') as file:
            dill.dump((res, res_ns, truth), file)
