'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [['man', 'women'], ['black', 'white'],
            ['king', 'queen'], ['girl', 'boy'],
            ['up', 'down'], ['empty', 'full']]

# Parameter
max_len = 5
n_hidden = 128
total_epoch = 3000
n_class = dic_len
batch_size = len(seq_data)

def make_batch(seq_data):
    target_batch = output_batch = input_batch = []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (max_len - len(seq[i]))

        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    # Make Tensor
    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(torch.LongTensor(target_batch))

# Model
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5, batch_first=True)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        outputs, _ = self.dec_cell(dec_input, enc_states)

        return self.fc(outputs)

input_batch, output_batch, target_batch = make_batch(seq_data)

model = Seq2Seq()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(total_epoch):
    # [num_layers * num_directions, batch, hidden_size]
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))
    optimizer.zero_grad()

    output = model(input_batch, hidden, output_batch)
    loss = 0
    for i in range(0, len(target_batch)):
        loss += criterion(output[i], target_batch[i])

    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()
# Test
def translate(word):
    input_batch, output_batch, target_batch = make_batch([[word, 'P' * len(word)]])

    # [num_layers * num_directions, batch, hidden_size]
    hidden = Variable(torch.zeros(1, 1, n_hidden))
    output = model(input_batch, hidden, output_batch)
    predict = output.data.max(2, keepdim=True)[1]

    decoded = [char_arr[i] for i in predict[0]]
    end = decoded.index('E')
    translated = ''.join(decoded[:end])
    
    return translated.replace('P','')

print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('hight ->', translate('hight'))
print('upp ->', translate('upp'))