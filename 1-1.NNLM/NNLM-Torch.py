'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
m = dic_len = len(word_dict) # V

# NNLM Parameter
n_step = 2 # n-1
n_hidden = 2 # h
total_epoch = 10000

# make batch function
# Pytorch : X one-hot, Y number label
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch

# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()

        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, dic_len).type(dtype))
        self.b = nn.Parameter(torch.randn(dic_len).type(dtype))

    def forward(self, X):
        input = X.view(-1, n_step * m) # get one batch
        tanh = nn.functional.tanh(self.d + torch.mm(input, self.H))
        output = torch.mm(tanh, self.U) + self.b
        return output

model = NNLM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.Tensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# Training
for epoch in range(total_epoch):

    optimizer.zero_grad()
    output = model(input_batch)

    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
input = [sen.split()[:2] for sen in sentences]
print(input)

output = []
for pre in predict:
    for key, value in word_dict.items():
        if value == pre:
            output.append(key)
print(output)
