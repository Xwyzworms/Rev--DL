import torch

def printDimShape(tensor : torch.Tensor):
    print(f" Dims : {(tensor.ndim)}")
    print(f" Shape : {(tensor.shape)}")