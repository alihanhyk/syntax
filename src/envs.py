from functools import partial
import jax
import jax.numpy as np
from jax.tree_util import Partial

###

env_params_diminishing = dict()
env_params_diminishing['factor_scaling'] = Partial(lambda T: 2.-10.**(np.arange(T+1)-T))
env_params_diminishing['mismatch_covariates'] = False
env_params_diminishing['mismatch_factors'] = False

env_params_increasing = dict()
env_params_increasing['factor_scaling'] = Partial(lambda T: 10.**(np.arange(T+1)-T))
env_params_increasing['mismatch_covariates'] = False
env_params_increasing['mismatch_factors'] = False

env_params_mismatchx = dict()
env_params_mismatchx['factor_scaling'] = env_params_diminishing['factor_scaling']
env_params_mismatchx['mismatch_covariates'] = True
env_params_mismatchx['mismatch_factors'] = False

env_params_mismatchz = dict()
env_params_mismatchz['factor_scaling'] = env_params_diminishing['factor_scaling']
env_params_mismatchz['mismatch_covariates'] = False
env_params_mismatchz['mismatch_factors'] = True

def make(key, env_params):
    N, T, K, R = 25, 5-1, 2, 2
    var = 1.

    if env_params['mismatch_factors']:
        N, T, K, R = 25, 5-1, 2, 5

    key, *subkeys = jax.random.split(key, 3)
    _auxX = jax.random.normal(subkeys[0], shape=(N,K))
    _auxZ = jax.random.normal(subkeys[1], shape=(N,R))

    xs = _auxX

    if env_params['mismatch_covariates']:
        _auxX = _auxX**2
    
    key, *subkeys = jax.random.split(key, 4)
    _auxD = jax.random.normal(subkeys[0], shape=(T+1,))
    _auxW = jax.random.ball(subkeys[1], K, shape=(T+1,))
    _auxM = jax.random.ball(subkeys[2], R, shape=(T+1,))
    
    _scale = env_params['factor_scaling'](T)
    _auxW = _auxW * _scale[:,None]
    _auxM = _auxM * _scale[:,None]

    ys_bar = _auxD[:-1][None,...] + _auxX @ _auxW[:-1].T + _auxZ @ _auxM[:-1].T

    key, subkey = jax.random.split(key)
    _auxR = jax.random.normal(subkey, shape=(N,))

    z0_bar = (_auxD[-1] + _auxX @ _auxW[-1] + _auxZ @ _auxM[-1])
    z1_bar = z0_bar + _auxR
    zs_bar = np.stack((z0_bar, z1_bar), axis=-1)

    info = dict()
    info['lambda'] = np.sum((_auxM[-1] @ (np.linalg.inv(_auxM[:-1].T @ _auxM[:-1]) @ _auxM[:-1].T))**2)
    return (xs, ys_bar, zs_bar, var), info

###

def _simulate(arg0, arg1):
    (ns, xs, ys, zs, 
        ys_bar, zs_bar, var, alg), (t, key) = arg0, arg1

    i, v = alg['recruit'](t, ns, xs, ys, zs, alg['infer'], alg['infer_params'])

    key, *subkeys = jax.random.split(key, 3)
    _y = ys_bar[i] + np.sqrt(var) * jax.random.normal(subkeys[0], shape=ys_bar.shape[1:])
    _z = zs_bar[i,v] + np.sqrt(var) * jax.random.normal(subkeys[1])

    ns = ns.at[i,v].add(1)
    ys = ys.at[i].add((_y - ys[i]) / ns[i].sum())
    zs = zs.at[i,v].add((_z - zs[i,v]) / ns[i,v])

    zs_hat, _ = alg['infer']['_is'](ns, xs, ys, zs, alg['infer_params'])
    return (ns, xs, ys, zs,
         ys_bar, zs_bar, var, alg), (zs_hat, ns)

@partial(jax.jit, static_argnames='H')
def simulate(key, env, alg, H):
    xs, ys_bar, zs_bar, var = env

    ns = np.ones((xs.shape[0], 2))
    key, *subkeys = jax.random.split(key, 3)
    ys = ys_bar + np.sqrt(var / 2) * jax.random.normal(subkeys[0], shape=ys_bar.shape)
    zs = zs_bar + np.sqrt(var) * jax.random.normal(subkeys[1], shape=zs_bar.shape)

    key, subkey = jax.random.split(key)
    _, res = jax.lax.scan(_simulate,
        (ns, xs, ys, zs, ys_bar, zs_bar, var, alg),
        (np.arange(H), jax.random.split(subkey, H)))

    return res
