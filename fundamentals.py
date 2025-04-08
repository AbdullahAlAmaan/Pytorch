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
random_tensor=torch.rand(2,2,3,4) # this is the size of the tensor, should give the matrix 3 rows 4 columns
print(random_tensor)
zero=torch.zeros(3,3)
print(zero)
ones=torch.ones(size=(3,3)) # could put size=() argument also but it is already there bu default
print(ones)
range=torch.arange(0,11,1) # gives the range of numbers from 0 to 10 with the step(difference between numbers) of 1 
print(range)
similar=torch.ones_like(range) # gives the same shape as the range tensor
print(similar)
float32=torch.tensor([1,2,3],dtype=torch.half , device=None # cpu gpu tpu where the tensor is located
                     , requires_grad=False)
float16=float32.type(torch.float64) #type conversion
print(float16)