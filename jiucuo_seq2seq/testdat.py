#coding:utf-8
'''中文注释'''
'''
import tensorflow as tf
max_value = tf.reduce_max([1, 3, 3, 2])
with tf.Session() as sess:
    max_value = sess.run(max_value)
    print(max_value)
'''
import io
with io.open('./data/tencent_chat_q_c.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()
with io.open('./data/tencent_chat_a_c.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()
    
#testline = '我 是 这样 分词 的'
#print(testline.split())


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
    #int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}

    return int_to_vocab, vocab_to_int
    

# 构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

def save_vocab(voca_dict, fn):
    fw = io.open(fn, 'w', encoding='utf-8')
    for word, idx in voca_dict.items():
        fw.write('%s\t\t\t%d\n'%(word, idx))
    fw.close()

#save_vocab(source_letter_to_int, './data/src_dict0.txt')
#save_vocab(target_letter_to_int, './data/tgt_dict0.txt')

#print(source_int_to_letter)
print(source_letter_to_int['<UNK>'], source_letter_to_int['<PAD>'], source_letter_to_int['<GO>'], source_letter_to_int['<EOS>'])
print(target_letter_to_int['<UNK>'], target_letter_to_int['<PAD>'], target_letter_to_int['<GO>'], target_letter_to_int['<EOS>'])
#print(target_int_to_letter)    一样的
#print(target_letter_to_int)

# 对字母进行转换
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
print(source_int[:5])

target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]
print(target_int[:5])

print(len(source_int), len(target_int))

#print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in [1, 2433, 2718, 1649]])))
print(source_int_to_letter[1122]+target_int_to_letter[357]+ target_int_to_letter[1754]+source_int_to_letter[1])

def source_to_seq(text):
    '''
    对源数据进行转换
    '''
#    sequence_length = 7
    sequence_length = len(text)+1
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))

input_word = u'你好我爱你'
for ww in input_word:
    print(ww)
text = source_to_seq(input_word)
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

with io.open('./data/test_in.txt', 'r', encoding='utf-8') as f:
    testin_data = f.read()

for lin in testin_data.split('\n'):
    text = source_to_seq(lin)
    print('  Word 编号:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))


'''
print('根据两点坐标计算直线斜率k，截距b：')
while True:
    line = raw_input()
    if line == '': break
    print('Robot: '+line)
    x1, y1, x2, y2 = (float(x) for x in line.split())
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    print('斜率:{}，截距:{}'.format(k, b))
'''