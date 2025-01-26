import torch

def kl_div(p1, p2):
    #p1: teacher
    #p2: student
    eps = 10**-12
    tmp1 = p1 + eps
    tmp2 = p2 + eps
    return torch.sum(tmp1*(tmp1.log()-tmp2.log()))/tmp1.size()[0]