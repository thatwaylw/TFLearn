#coding:utf-8

from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))
import numpy as np
import time
import tensorflow as tf
import io

with io.open('./data/tencent_chat_q_c.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()
with io.open('./data/tencent_chat_a_c.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

def extract_character_vocab(data):
    # 构造映射表
    vocab_to_int = dict()
    # 这里要把四个特殊字符添加进词典
    vocab_to_int['<PAD>'] = 0
    vocab_to_int['<UNK>'] = 1
    vocab_to_int['<GO>']  = 2
    vocab_to_int['<EOS>'] = 3
    idx = 4
    for line in data.split('\n'):
        for character in line:
            if character not in vocab_to_int:
                vocab_to_int[character] = idx
                idx += 1
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}

    return int_to_vocab, vocab_to_int

# 构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

# Batch Size
batch_size = 128
               
def source_to_seq(text):
    # 对源数据进行转换
#    sequence_length = 7
    sequence_length = len(text)+1
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))

checkpoint = "./model/tencent_chat3.ckpt"

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

    fw = io.open('./data/test_out.txt', 'a', encoding='utf-8')
    with io.open('./data/test_in.txt', 'r', encoding='utf-8') as f:
        testin_data = f.read()
    for input_word in testin_data.split('\n'):
        text = source_to_seq(input_word)

        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          target_sequence_length: [len(input_word)] * batch_size,
                                          source_sequence_length: [len(input_word)] * batch_size})[0]
        
        buf = input_word
        buf += '\t\t\t{}\n'.format("".join([target_int_to_letter[i] for i in answer_logits if i != pad]))
        print(buf)
        fw.write(buf)
        
    fw.close()
