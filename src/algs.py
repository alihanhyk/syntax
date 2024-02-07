import jax
import jax.numpy as np
from jax.tree_util import Partial

### INFERENCE

def _infer_bandit(i, ns, xs, ys, zs, infer_params):
    return zs[i,1] - zs[i,0], 1 / ns[i,0] + 1 / ns[i,1]

def _infer_synthetic(i, ns, xs, ys, zs, infer_params):
    _i = np.zeros(ns.shape[0]).at[i].set(1.)

    _qq = np.diag(1 / ns[:,0] + infer_params['lambda'] / ns.sum(axis=-1))
    _qc = -infer_params['lambda'] * _i / ns.sum(axis=-1)
    _qa = np.concatenate((xs, ys, np.ones((ns.shape[0],1))), axis=1).T
    _qb = np.concatenate((xs[i], ys[i], np.ones(1)))
    _a0 = np.concatenate((
        np.concatenate((_qq, _qa.T), axis=1),
        np.concatenate((_qa, np.zeros((_qa.shape[0], _qa.shape[0]))), axis=1)), axis=0)
    _a1 = np.concatenate((-_qc, _qb))
    w = np.linalg.solve(_a0, _a1)[:_qa.shape[1]]
    
    return zs[i,1] - zs[:,0] @ w, \
        1 / ns[i,1] + np.sum(w**2 / ns[:,0]) \
            + infer_params['lambda'] * np.sum((_i - w)**2 / ns.sum(axis=-1))

###

infer_bandit = dict()
infer_bandit['_is'] = Partial(
    jax.jit(lambda ns, xs, ys, zs, infer_params:
        jax.vmap(_infer_bandit,
            in_axes=(0,None,None,None,None,None))
            (np.arange(ns.shape[0]), ns, xs, ys, zs, infer_params)))

infer_bandit['i_nss'] = Partial(
    jax.jit(lambda i, ns, xs, ys, zs, infer_params:
        jax.vmap(jax.vmap(_infer_bandit,
            in_axes=(None,0,None,None,None,None)),
            in_axes=(None,0,None,None,None,None))
            (i, ns[None,None,...] \
                + np.zeros((ns.shape[0],2,ns.shape[0],2)) \
                    .at[:,0,:,0].set(np.eye(ns.shape[0])) \
                    .at[:,1,:,1].set(np.eye(ns.shape[0])), xs, ys, zs, infer_params)))

infer_synthetic = dict()
infer_synthetic['_is'] = Partial(
    jax.jit(lambda ns, xs, ys, zs, infer_params:
        jax.vmap(_infer_synthetic,
            in_axes=(0,None,None,None,None,None))
            (np.arange(ns.shape[0]), ns, xs, ys, zs, infer_params)))

infer_synthetic['i_nss'] = Partial(
    jax.jit(lambda i, ns, xs, ys, zs, infer_params:
        jax.vmap(jax.vmap(_infer_synthetic,
            in_axes=(None,0,None,None,None,None)),
            in_axes=(None,0,None,None,None,None))
            (i, ns[None,None,...] \
                + np.zeros((ns.shape[0],2,ns.shape[0],2)) \
                    .at[:,0,:,0].set(np.eye(ns.shape[0])) \
                    .at[:,1,:,1].set(np.eye(ns.shape[0])), xs, ys, zs, infer_params)))

### RECRUITMENT

@Partial
@jax.jit
def recruit_uniform(t, ns, xs, ys, zs, infer, infer_params):
    return t % ns.shape[0], (t // ns.shape[0]) % ns.shape[1]

@Partial
@jax.jit
def recruit_static(t, ns, xs, ys, zs, infer, infer_params):
    _, vars = infer['_is'](ns, xs, ys, zs, infer_params)
    i = np.argmax(vars)
    _, _vars = infer['i_nss'](i, ns, xs, ys, zs, infer_params)
    return np.unravel_index(np.argmin(_vars), _vars.shape)

@Partial
@jax.jit
def recruit_adaptive(t, ns, xs, ys, zs, infer, infer_params):
    bars, vars = infer['_is'](ns, xs, ys, zs, infer_params)
    i = np.argmin(np.abs(bars) / np.sqrt(vars))
    _, _vars = infer['i_nss'](i, ns, xs, ys, zs, infer_params)
    return np.unravel_index(np.argmin(_vars), _vars.shape)
