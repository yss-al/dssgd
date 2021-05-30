#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def DSSGD(w, w_glob, theta_upload=0.1):
    if theta_upload == 1.:
        return copy.deepcopy(w)
    w_row = torch.cat([v.flatten() for _, v in w.items()])
    w_glob_row = torch.cat([v.flatten() for _, v in w_glob.items()])
    delta_grad = w_glob_row - w_row
    _, indexes = torch.topk(delta_grad, int(len(delta_grad) * 0.1)) 
    coeff = torch.zeros_like(delta_grad)
    coeff[indexes] = 1.
    delta_grad = delta_grad * coeff 
    w_glob_row = w_glob_row - delta_grad
    upload_w = copy.deepcopy(w_glob)
    start = 0
    for k, v in upload_w.items():
        upload_w[k] = w_glob_row[start:start + v.nelement()].view(v.size())
        start += v.nelement()
    return upload_w


