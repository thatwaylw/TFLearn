#coding:utf-8

from distutils.version import LooseVersion
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

import numpy as np
import datapro as dp

checkpoint = "./model/tencent_chat_1.ckpt"
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

    fw = open('./data/test_some_out.txt', 'a', encoding='utf-8')
    with open('./data/test_some.txt', 'r', encoding='utf-8') as f:
        testin_data = f.read()
    for input_word in testin_data.split('\n'):
        text = dp.source_to_seq(input_word, source_letter_to_int)

        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          target_sequence_length: [len(input_word)] * batch_size,
                                          source_sequence_length: [len(input_word)] * batch_size})[0]
        
        buf = input_word
        buf += '\t\t\t{}'.format("".join([target_int_to_letter[i] for i in answer_logits if i != pad]))
        print(buf)
        fw.write(buf)
    fw.write('\n')
    fw.close()
