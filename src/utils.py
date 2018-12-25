import torch

def torch_device():
    if torch.cuda.device_count() > 0:
        return 'cuda:0'
    else:
        return 'cpu'
