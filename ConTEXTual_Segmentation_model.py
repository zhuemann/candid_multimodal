import torch

class Context(torch.nn.Module):
    def __init__(self, model, n_class, n_nodes):
        print("hi")