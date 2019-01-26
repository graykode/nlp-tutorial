'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Reference : https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
'''
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# Bi-LSTM(Attention) Parameters
embedding_dim = 2
n_hidden = 5 # number of hidden units in one cell
sequence_length = 3 # all sentence is consist of 3 words
num_classes = 2  # 0 or 1

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)

input_batch = []
for sen in sentences:
    input_batch.append(np.asarray([word_dict[n] for n in sen.split()]))

target_batch = []
for out in labels:
    target_batch.append(np.eye(num_classes)[out]) # ONE-HOT : To using Tensor Softmax Loss function

# LSTM Model
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, num_classes])
out = tf.Variable(tf.random_normal([n_hidden * 2, num_classes]))

embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim]))
input = tf.nn.embedding_lookup(embedding, X) # [batch_size, len_seq, embedding_dim]

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

# output : [batch_size, len_seq, n_hidden], states : [batch_size, n_hidden]
output, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, input, dtype=tf.float32)

# Attention
output = tf.concat([output[0], output[1]], 2)                             # output[0] : lstm_fw, output[1] : lstm_bw
final_hidden_state = tf.concat([final_state[1][0], final_state[1][1]], 1) # final_hidden_state : [batch_size, n_hidden * num_directions(=2)]
final_hidden_state = tf.expand_dims(final_hidden_state, 2)                # final_hidden_state : [batch_size, n_hidden * num_directions(=2), 1]

attn_weights = tf.squeeze(tf.matmul(output, final_hidden_state), 2) # attn_weights : [batch_size, len_seq]
soft_attn_weights = tf.nn.softmax(attn_weights, 1)
new_hidden_state = tf.matmul(tf.transpose(output, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2)) # new_hidden_state : [batch_size, n_hidden * num_directions(=2), 1]
new_hidden_state = tf.squeeze(new_hidden_state, 2) # [batch_size, n_hidden * num_directions(=2)]

model = tf.matmul(new_hidden_state, out)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# Model-Predict
hypothesis = tf.nn.softmax(model)
predictions = tf.argmax(hypothesis, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%06d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Test
test_text = 'sorry hate you'
tests = []
tests.append(np.asarray([word_dict[n] for n in test_text.split()]))

predict = sess.run([predictions], feed_dict={X: tests})
result = predict[0][0]
if result == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")