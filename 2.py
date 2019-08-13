import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix2, matrix1)
result =  sess.run(product)
print(result)

wa = tf.constant([[0.0, 3., 0.01]])
wb = tf.constant([[1., 0.1, 2.]])
product = tf.matmul(wa, wb, transpose_b=True)
result =  sess.run(product)
print(result)

va = tf.constant([0.0, 3., 0.01])
wva = tf.reshape(va, [3, 1])
print(sess.run(wva))

x = tf.random_normal(shape=[1,5],mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
y = x
x_shuffle = tf.random_shuffle(y)
#sess.run(tf.initialize_all_variables())  # what use??   python2
sess.run(tf.global_variables_initializer())  # what use??   python3
print(sess.run(y))
#print(y)
#print(x.eval())
print(sess.run(y))
print(sess.run(x_shuffle))	# why changed??

#y = [1.0, 2.0, 3.0, 4.0, 5.9]
#for i in xrange(5):
#	y[i] = x[0,i]
#print(sess.run(y))  Error!!!!????
#print(y)
#print(sess.run())

sess.close()
