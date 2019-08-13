import tensorflow as tf
hello = tf.constant('Hello, Yuanqu GPU!')
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))

