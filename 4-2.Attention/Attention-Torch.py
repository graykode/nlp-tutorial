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

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz ']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# Parameter
max_len = 20
n_hidden = 128
total_epoch = 10000
n_class = dic_len

seq_data = [['Ich mochte ein bier', 'I want a BEER']]

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (max_len - len(seq[i]))

        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])

        target_batch.append(target)

    return input_batch, output_batch, target_batch