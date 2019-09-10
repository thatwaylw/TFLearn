#coding:utf-8

def read_data(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        all_data = f.read()
    
    source_data = []
    target_data = []
    for line in all_data.split('\n'):
        lin = line.replace(' ', '')     # 去掉空格，忽略分词
        tok = lin.split('\t')
        if(len(tok)<2): continue
        source_data.append(tok[0])
        target_data.append(tok[1])
    return source_data, target_data

def extract_character_vocab(data):
    # 构造映射表
    vocab_to_int = dict()
    # 这里要把四个特殊字符添加进词典
    vocab_to_int['<PAD>'] = 0
    vocab_to_int['<UNK>'] = 1
    vocab_to_int['<GO>']  = 2
    vocab_to_int['<EOS>'] = 3
    idx = 4
    for line in data:
        for character in line:      # 逐字。。。
            if character not in vocab_to_int:
                vocab_to_int[character] = idx
                idx += 1
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}

    return int_to_vocab, vocab_to_int
    
def source_to_seq(text, source_letter_to_int):
    # 对源数据进行转换
#    sequence_length = 7
    sequence_length = len(text)+1
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))


# 保存字典，确保每次加载的时候字典不变，否则显示乱掉
def save_vocab(voca_dict, fn):
    fw = io.open(fn, 'w', encoding='utf-8')
    for word, idx in voca_dict.items():
        fw.write('%s\t\t\t%d\n'%(word, idx))
    fw.close()
