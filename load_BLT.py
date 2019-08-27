#coding=utf-8
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
sentence = (
	'The morning had dawned clear and cold with a crispness that hinted at the end of summer They '
	'set forth at daybreak to see a man beheaded twenty in all and Bran rode among them nervous with '
	'excitement This was the first time he had been deemed old enough to go with his lord father and his '
	'brothers to see the king’s justice done It was the ninth year of summer and the seventh of Bran’s life'
) # 就是1句，1个str
dict_list = list(set(sentence.split()))
dict_list.sort()
word_dict = {w: i for i, w in enumerate(dict_list)}
number_dict = {i: w for i, w in enumerate(dict_list)}
n_class = len(word_dict)
n_step = 5 #len(sentence.split())
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

#X = tf.placeholder(tf.float32, [None, n_step, n_class])
input_batch, target_batch = make_batch(sentence)
print(len(input_batch), input_batch[0].shape)
print(len(target_batch), target_batch[0].shape)

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('models/ckp-1000.meta')
	saver.restore(sess, tf.train.latest_checkpoint('models'))
	graph = tf.get_default_graph()
	X = graph.get_tensor_by_name('xx:0')
	prediction = graph.get_tensor_by_name('preds:0')
	
	predict =  sess.run([prediction], feed_dict={X: input_batch})
	print(sentence)
	print([number_dict[n] for n in [pre for pre in predict[0]]])