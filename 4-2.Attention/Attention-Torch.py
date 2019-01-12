# code by Tae Hwan Jung(Jeff Jung) @graykode
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
n_class = len(word_dict)

# Parameter
max_len = 5 # 'S' or 'E' will be added (= n_step,seq_len)
n_hidden = 128
batch_size = 1

def make_batch(sentences):
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]

    # make tensor
    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(torch.LongTensor(target_batch))
  
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        
        # Linear for attention
        self.attn = nn.Linear(n_hidden, n_hidden)
        
    def forward(self, enc_input, hidden, dec_input):
        enc_input = enc_input.transpose(0, 1) # enc_input: [max_len(=n_step, time step), batch_size, n_hidden]
        dec_input = dec_input.transpose(0, 1) # dec_input: [max_len(=n_step, time step), batch_size, n_hidden]

        # enc_outputs : [max_len, batch_size, num_directions(=1) * n_hidden(=1)]
        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        enc_outputs, enc_states = self.enc_cell(enc_input, hidden)
        dec_outputs, _ = self.dec_cell(dec_input, enc_states)

        
        return dec_outputs
      
    def get_att_weight(self, hidden, enc_outputs):
        attn_scores = Variable(torch.zeros(len(enc_outputs)))  # attn_scores : [n_step]

    def get_att_score(self, hidden, encoder_hidden):
        score = self.attn(encoder_hidden)
        return torch.dot(hidden.view(-1), score.view(-1))

input_batch, output_batch, target_batch = make_batch(sentences)

# hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
hidden = Variable(torch.zeros(1, 1, n_hidden))

model = Attention()
output = model(input_batch, hidden, output_batch)