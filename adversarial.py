import torch
import numpy as np

class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 12500

    @staticmethod
    def forward(ctx, input):
        AdversarialLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 10
        low = 0.0
        high = 1.0
        lamb = 2.0
        iter_num, max_iter = AdversarialLayer.iter_num, AdversarialLayer.max_iter 
        coeff = np.float(lamb * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
        return -coeff * gradOutput