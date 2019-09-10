#coding:utf-8

from distutils.version import LooseVersion
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

import numpy as np
import datapro as dp

checkpoint = "./model/temp.ckpt"
train_fn = './data/tencent_chat.txt'


# 读入训练数据
source_data, target_data = dp.read_data(train_fn)

# 构造映射表
source_int_to_letter, source_letter_to_int = dp.extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = dp.extract_character_vocab(target_data)

# Batch Size
batch_size = 128
			   
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    pad = source_letter_to_int["<PAD>"]

    print('模型已加载，开始聊天吧：...')
    while True:
        input_word = input('U: ') # 输入一个单词
        if input_word == '': break #空回车则退出
        text = dp.source_to_seq(input_word, source_letter_to_int)

        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size})[0]

        print('\nSource')
        print('  Word 编号:    {}'.format([i for i in text]))
        print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

        print('\nTarget')
        print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
        print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))
