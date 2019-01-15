import torch
import numpy as np 
from torch import nn 
from torch.autograd import Variable


"""
1. torch.ones / zeros 
2. torch.rand / randn /
3. isnan isinf
4. unsqueeze & squeeze & view    squeeze 降维。 unsqueeze 升维
5. contiguous()    contiguous().view() == reshape  
6. embedding 
7. nn.init 

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



####  5  luong general

batch_size = 11
encoder_hidden_size = 9
encoder_len = 13
decoder_hidden_size = 7

attn_W = nn.Linear(
			decoder_hidden_size,
			encoder_hidden_size,
			bias = False)
### input is 
encoder_output = Variable(torch.rand(batch_size, encoder_len, encoder_hidden_size))   #### B, T, d1
decoder_state = Variable(torch.rand(batch_size, decoder_hidden_size))				#### B, d2
### output is  B, T

### attention
Wh = attn_W(decoder_state)   #### B,d1
assert Wh.shape == (batch_size, encoder_hidden_size)
Wh = Wh.unsqueeze(1)
Wh_ext = Wh.repeat(1,encoder_len,1)   #### B,T,d1
assert Wh_ext.shape == (batch_size, encoder_len, encoder_hidden_size)
eWh = Wh_ext * encoder_output   #### B,T,d1
assert eWh.shape == (batch_size, encoder_len, encoder_hidden_size)
eWh = eWh.sum(2)
assert eWh.shape == (batch_size, encoder_len)











