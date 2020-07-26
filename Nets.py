import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
from functional import  LinearWeightNorm
# Discriminator architecture: 5 FC WN layers with units (500,500,250,250,250)
class Discriminator(nn.Module):
    def __init__(self, input_dim = 28 ** 2, output_dim = 10):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([
            LinearWeightNorm(input_dim, 500), #Parameter in FC layer constrained by WN for regularization
            LinearWeightNorm(500, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)] 
        )
        self.final = LinearWeightNorm(250, output_dim, weight_scale=1)
    def forward(self, x, feature = False, cuda = False, first = False):
#        pdb.set_trace()
        x = x.view(-1, self.input_dim) # x.size = (64,1561)
        noise = torch.randn(x.size()) * 0.05 if self.training else torch.Tensor([0]) #Gaussian Noise are added after FC and Input layer for smoothing purpose
        if cuda:
            noise = noise.cuda()
        x = x + Variable(noise, requires_grad = False)
        if first:
            return self.layers[0](x)
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.elu(m(x))
            noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0])
            if cuda:
                noise = noise.cuda()
            x = (x_f + Variable(noise, requires_grad = False))
        if feature: # if in non-prediction mode
            return x_f, self.final(x) #x_f = features extracted from input x, for feature matching in Generator
        return self.final(x) #if in prediction mode, noise is not included
#where is the output softmax layer???

class Generator(nn.Module):
    def __init__(self, z_dim, output_dim = 28 ** 2):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 500, bias = False)
        self.bn1 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.fc2 = nn.Linear(500, 500, bias = False)
        self.bn2 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.fc3 = LinearWeightNorm(500, output_dim, weight_scale = 1)
        self.bn1_b = Parameter(torch.zeros(500))
        self.bn2_b = Parameter(torch.zeros(500))
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
    def forward(self, batch_size, cuda = False, seed = -1):
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad = False, volatile = not self.training)
        if cuda:
            x = x.cuda()
        x = F.elu(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.elu(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.tanh(self.fc3(x)) #Generator output layer is constrained by weight normalization trick with a trainable weight scale??
        return x

