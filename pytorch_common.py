import torch
import numpy as np 

"""
1. torch.ones / zeros 
2. torch.rand / randn /
3. isnan isinf
4. unsqueeze & squeeze & view 


variable  
	requires_grad is boolean 
sum / mean / max / min 
norm ?




nn.Linear 
activation: relu tanh sigmoid softmax 


advanced
	1. class + nn.Module + __init__  + forward(__call__) 
	
"""

### 1. torch.ones zeros
'''
a = torch.zeros(2,3)
b = torch.ones(2,3)
print(torch.eye(3))  ### identity-matrix
print(a)
print(b)
'''

### 2. torch.randn rand 
'''
a = torch.rand(3,5)
b = torch.randn(3,5)
'''

### 3. isnan isinf
'''
a = torch.rand(3,5)
b = torch.zeros(3,5)
c = a / b
print(c)
print(torch.isinf(c)) 
assert torch.isinf(c).any()
assert torch.isinf(c).all()
'''

### 4. unsqueeze & squeeze & view 

### 4.1 view
'''
a = torch.Tensor(2,3)
b = a.view(1,-1)
print(a)
print(b)
'''

### 4.2 squeeze & unsqueeze
'''
b=torch.Tensor(1,3)
c = b.squeeze(0)
d = c.unsqueeze(1)
assert b.shape == (1,3)
assert c.shape == (3,)
assert d.shape == (3,1)
'''








