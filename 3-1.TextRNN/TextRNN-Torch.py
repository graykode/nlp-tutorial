'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
n_class = dic_len = len(word_dict)

# Parameter
batch_size = len(sentences)
total_epoch = 10000
n_step = 2 # n_step mean
n_hidden = 5
learning_rate = 0.001

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

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.Tensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, batch_first=True)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, hidden, X):
        outputs, hidden = self.rnn(X, hidden)
        outputs = outputs.permute(1,0,2)
        outputs = outputs[-1]
        model = torch.mm(outputs, self.W) + self.b
        return model

model = TextRNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1, total_epoch + 1):
    optimizer.zero_grad()

    # [num_layers * num_directions, batch, hidden_size]
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))
    output = model(hidden, input_batch)

    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

input = [sen.split()[:2] for sen in sentences]
print(input)

# Predict
hidden = Variable(torch.zeros(1, batch_size, n_hidden))
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
output = []
for pre in predict:
    for key, value in word_dict.items():
        if value == pre:
            output.append(key)
print(output)