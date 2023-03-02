import torch

a = torch.LongTensor([[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
print(a)
print('shape a:', a.shape)
embedding = torch.nn.Embedding(4, 64)
b = embedding(a)
print('shape b:', b.shape)
print(b)
