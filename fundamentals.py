import torch
import numpy as np
import torch.nn as nn

# scalar=torch.tensor(3)
# print(scalar)
# print(scalar.ndim)
# print(scalar.shape)
# vector=torch.tensor([1,2])
# print(vector)
# print(vector.ndim) # gives number of pairs of brackets
# print(vector.shape) # gives rows and columns 
# matrix=torch.tensor([[1,2],[3,4]])
# print(matrix)
# print(matrix.ndim)
# print(matrix.shape)
# random_tensor=torch.rand(2,2,3,4) # this is the size of the tensor, should give the matrix 3 rows 4 columns
# print(random_tensor)
# zero=torch.zeros(3,3)
# print(zero)
# ones=torch.ones(size=(3,3)) # could put size=() argument also but it is already there bu default
# print(ones)
# range=torch.arange(0,11,1) # gives the range of numbers from 0 to 10 with the step(difference between numbers) of 1 
# print(range)
# similar=torch.ones_like(range) # gives the same shape as the range tensor
# print(similar)
# float32=torch.tensor([1,2,3],dtype=torch.half , device=None # cpu gpu tpu where the tensor is located
#                      , requires_grad=False)
# float16=float32.type(torch.float64) #type conversion
# print(float16)
# print(float16.dtype)
# print(float16.device)
# print(float16.size())
# some_tensor=torch.rand(3,2,dtype=torch.float64)
# tranpose=some_tensor.T # for mm we can use transpose function, that changes columns into rows and rows into columns
# print(tranpose)
# some_tensor1=torch.rand(2,2,dtype=torch.float64)
# print(some_tensor)
# print (some_tensor.dtype)
# matmul=torch.matmul(some_tensor,some_tensor1) 
# #matrix multiplication can also be done using torch.mm(a,b)
# print(some_tensor)
# print(some_tensor1)
# print(matmul)
# some_tensor1=torch.rand(2,2,dtype=torch.float64)
# print(some_tensor1)
# print(some_tensor1.max())
# print(torch.max(some_tensor1)) # can be done both ways for sum, mean,max,min
# print(some_tensor1.argmax()) # helps to find the position where max or min is
tensor=torch.rand(3,4)
print(tensor)
print (tensor.shape)
tensor2=tensor.reshape(4,3) # number of rows and columns, it has to match the total elements to be compatible (multiply both and see if same as number of elements )
print(tensor2)
print(tensor2.size())
# .view function is by reference, so if i initialize another variable with my tensor and make any changes there, it will change my original variable too
stacked=torch.stack([tensor,tensor])
print(stacked)