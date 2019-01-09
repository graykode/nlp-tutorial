'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

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
# Tensorflow : X one-hot, Y ont-hot
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(np.eye(dic_len)[target])

    return input_batch, target_batch

# Model
X = tf.placeholder(tf.float32, [None, n_step, m])
Y = tf.placeholder(tf.float32, [None, dic_len])

input = tf.reshape(X, shape=[-1, n_step * m])
H = tf.Variable(tf.random_normal([n_step * m, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, dic_len]))
b = tf.Variable(tf.random_normal([dic_len]))

tanh = tf.nn.tanh(d + tf.matmul(input, H))
model = tf.matmul(tanh, U) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction =tf.argmax(model, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentences)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict =  sess.run([prediction], feed_dict={X: input_batch})

# Test
input = [sen.split()[:2] for sen in sentences]
print(input)

output = []
for pre in [pre for pre in predict[0]]:
    for key, value in word_dict.items():
        if value == pre:
            output.append(key)
print(output)