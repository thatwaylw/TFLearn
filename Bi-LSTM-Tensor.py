'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentence = (
    'The morning had dawned clear and cold with a crispness that hinted at the end of summer They '
	'set forth at daybreak to see a man beheaded twenty in all and Bran rode among them nervous with '
	'excitement This was the first time he had been deemed old enough to go with his lord father and his '
	'brothers to see the king’s justice done It was the ninth year of summer and the seventh of Bran’s life'
) # 就是1句，1个str

word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
n_class = len(word_dict)
n_step = 5 #len(sentence.split())     # 改成5就出错
n_hidden = 5

def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        if i < n_step-1:
            input = [word_dict[n] for n in words[:(i + 1)]]
            input = input + [0] * (n_step - len(input))
        else:
            input = [word_dict[n] for n in words[(i+1-n_step):(i+1)]]
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])
    return input_batch, target_batch

# Bi-LSTM Model
X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])
#print(X.shape)

W = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))
b = tf.Variable(tf.random_normal([n_class]))
#print(W.shape)

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

input_batch, target_batch = make_batch(sentence)
print(len(input_batch), input_batch[0].shape)
print(len(target_batch), target_batch[0].shape)


# outputs : [batch_size, len_seq, n_hidden], states : [batch_size, n_hidden]
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype=tf.float32)

#print(outputs.shape)
#ooo =  sess.run([outputs], feed_dict={X: input_batch})
#print(' ::: ', ooo[0][0].shape, ooo[0][1].shape)
#exit()

outputs = tf.concat([outputs[0], outputs[1]], 2) # output[0] : lstm_fw, output[1] : lstm_bw
outputs = tf.transpose(outputs, [1, 0, 2]) # [n_step, batch_size, n_hidden]
outputs = outputs[-1] # [batch_size, n_hidden]

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training
for epoch in range(10000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

predict =  sess.run([prediction], feed_dict={X: input_batch})
print(sentence)
print([number_dict[n] for n in [pre for pre in predict[0]]])