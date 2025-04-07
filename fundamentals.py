import torch
import numpy as np
import torch.nn as nn

scalar=torch.tensor(3)
print(scalar)
print(scalar.ndim)
print(scalar.shape)
vector=torch.tensor([1,2])
print(vector)
print(vector.ndim) # gives number of pairs of brackets
print(vector.shape) # gives rows and columns 
matrix=torch.tensor([[1,2],[3,4]])
print(matrix)
print(matrix.ndim)
print(matrix.shape)
