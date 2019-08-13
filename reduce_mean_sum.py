import tensorflow as tf

x = [[1,2,3],
      [1,2,3]]
 
xx = tf.cast(x,tf.float32)
 
mean_all = tf.reduce_mean(xx, keep_dims=False)
mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False)
mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False)

sum_all = tf.reduce_sum(xx, keep_dims=False)
sum_0 = tf.reduce_sum(xx, axis=0, keep_dims=False)
sum_1 = tf.reduce_sum(xx, axis=1, keep_dims=False)
 




# calculate cross_entropy 
y = tf.constant([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]])  
y_= tf.constant([[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0]])  
ysoft = tf.nn.softmax(y)
cross_entropyVec = -y_*tf.log(ysoft)
cross_entropy = tf.reduce_sum(cross_entropyVec)
cross_entropy_loss = tf.reduce_mean(cross_entropyVec)
 
#do cross_entropy just one step  
tnsce = tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_)
cross_entropy2 = tf.reduce_sum(tnsce)
cross_entropy_loss2 = tf.reduce_mean(tnsce)
 
with tf.Session() as sess:
    m_a,m_0,m_1, s_a,s_0,s_1 = sess.run([mean_all, mean_0, mean_1, sum_all, sum_0, sum_1])
    print('reduce_mean all,0,1...')
    print(m_a)    # output: 2.0
    print(m_0)    # output: [ 1.  2.  3.]
    print(m_1)    # output: [ 2.  2.]
    print('reduce_sum all,0,1...')
    print(s_a)    # output: 12.0
    print(s_0)    # output: [ 2.  4.  6.]
    print(s_1)    # output: [ 6.  6.]

    print("step1:softmax result=ysoft")  
    print(sess.run(ysoft))  
    print("step2:y_*log(ysoft):cross_entropy vector=")  
    print(sess.run(cross_entropyVec))  
    print("reduce_sum: cross_entropy result=")  
    print(sess.run(cross_entropy))  
    print("reduce_mean: cross_entropy_lost result=")  
    print(sess.run(cross_entropy_loss))  
    print("---此函数已对Y向量维度上求和，剩下一维是batchSize------")
    print("Function(softmax_cross_entropy_with_logits) result=")  
    print(sess.run(tnsce))
    print("reduce_sum: cross_entropy result=")  
    print(sess.run(cross_entropy2))
    print("reduce_mean: ross_entropy_loss result=")  
    print(sess.run(cross_entropy_loss2))
