import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import numpy as np
def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))
def normalize_infnorm(x, eps=1e-8):
    assert type(x) == np.ndarray
    return x / (abs(x).max(axis = 0) + 1e-8)   

class LinearWeightNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1 
    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim = 1, keepdim = True))
        return F.linear(x, W, self.bias)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) +')'

def pull_away_term(x):
    '''pull-away loss

    Args:
        x: type=> torch Tensor or Variable, size=>[batch_size * feature_dim], generated samples

    Return:
        scalar Loss
    '''
    x = F.normalize(x)
    pt = x.matmul(x.t()) ** 2
    return (pt.sum() - pt.diag().sum()) / (len(x) * (len(x) - 1))

   
