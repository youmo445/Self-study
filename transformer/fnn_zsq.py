import torch
from torch import Tensor

def manual_sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def single_neuron_forward(x, weight, bias):
    weighted_sum = (x * weight).sum()
    z = weighted_sum + bias 
    y = manual_sigmoid(z)
    return z, y

def manual_forward(x, weight, bias):
    z = x @ weight.transpose(0,1) + bias
    y = manual_sigmoid(z)
    return z, y
    
def xor_forward(x):
    hidden_weight_0 = torch.tensor([3.0, 4.0])
    hidden_bias_0 = torch.tensor(-2.0)
    hidden_weight_1 = torch.tensor([2.0, 3.0])
    hidden_bias_1 = torch.tensor(-4.0)
    z_0, h_0 = single_neuron_forward(x, hidden_weight_0, hidden_bias_0)
    z_1, h_1 = single_neuron_forward(x, hidden_weight_1, hidden_bias_1)
    z = torch.stack([z_0, z_1])
    h = torch.stack([h_0, h_1])
    output_weight = torch.tensor([5.0, -5.0])
    output_bias = torch.tensor(-2.0)
    z_out, y = single_neuron_forward(h, output_weight, output_bias)
    return z, h, z_out, y

def xor_forward_manual(x):
    hidden_weight = torch.tensor([[3.0, 4.0], [2.0, 3.0]])
    hidden_bias = torch.tensor([-2.0, -4.0])
    z, h = manual_forward(x, hidden_weight, hidden_bias)
    output_weight = torch.tensor([[5.0, -5.0]])
    output_bias = torch.tensor([-2.0])
    z_out, y = manual_forward(h, output_weight, output_bias)
    return z, h, z_out, y

class XORNetwork(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.hidden_weight = torch.nn.Parameter([[3.0, 4.0], [2.0, 3.0]])
        self.hidden_bias = 