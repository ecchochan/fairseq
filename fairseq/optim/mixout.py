#!/usr/bin/env python3

"""
Example of a generic Mixout implementation. (Lee et al., 2019).
https://arxiv.org/abs/1909.11299
Implementation by Stephen Roller (https://stephenroller.com).
Updated 2020-02-10 to include 1/(1 - p) correction term. Thanks to
Cheolhyoung Lee for making this correction.
Example output:
$ python mixout.py
parameter: 0.weight   Vanilla distance: 0.00239  Mixout distance: 0.00128
parameter: 0.bias     Vanilla distance: 0.000191  Mixout distance: 5.8e-05
parameter: 2.weight   Vanilla distance: 0.000494  Mixout distance: 0.000258
parameter: 2.bias     Vanilla distance: 1.75e-05  Mixout distance: 1.01e-05

# https://gist.github.com/stephenroller/f45a372e231825f9f5578e9e705f4e95

"""

import torch
import torch.nn as nn


def MixoutWrapper(module: nn.Module, p: float = 0.7, exclude: str = 'layer_norm'):
    """
    Implementation of Mixout (https://arxiv.org/abs/1909.11299).
    Use with:
    >>> mixout_model = model.apply(MixoutWrapper).
    """
    # duplicate all the parameters, making copies of them and freezing them
    module._names = []
    module._params_orig = dict()
    _params_learned = nn.ParameterDict()
    exclude = exclude.split(',')
    for n, q in list(module.named_parameters(recurse=False)):
        if any(k in n for k in exclude):
            continue
        c = q.clone().detach()
        c.requires_grad = False
        module._params_orig[n] = c
        _params_learned[n] = q
        module._names.append(n)
        delattr(module, n)
        setattr(module, n, c)
    if module._names:
        module._params_learned = _params_learned

    def mixout(module, n):
        if module.training:
            o = module._params_orig[n]
            mask = (torch.rand_like(o) < p).type_as(o)
            # update 2020-02-
            return (
                mask * module._params_orig[n]
                + (1 - mask) * module._params_learned[n]
                - p * module._params_orig[n]
            ) / (1 - p)
        else:
            return module._params_learned[n].data

    def hook(module, input):
        for n in module._names:
            v = mixout(module, n)
            setattr(module, n, v)

    module.register_forward_pre_hook(hook)
    return module
