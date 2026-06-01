import jieba
import tqdm
import config

class JiebaTokenizer():
    unk_token = '<unk>'
    pad_token = '<pad>' 
    def __init__(self,vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word : index for index,word in enumerate(vocab_list)}
        self.index2word = {index : word for index,word in enumerate(vocab_list)}
        #self.unk_token = '<unk>'   #放在init里面是实例属性，放在外面是类属性
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
    #实例方法、类方法

    #实例方法
    #@staticmethod静态方法
    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)
    
    def encode(self,text,seq_len):
        tokens = self.tokenize(text)
        #截取、填充 到指定长度
        if len(tokens) > seq_len:
            tokens = tokens [:seq_len]
        elif len(tokens) < seq_len:
            tokens = tokens + [self.pad_token] * (seq_len - len(tokens))
        return [self.word2index.get(token,self.unk_token_index) for token in tokens]
    
    #构建词表,使其为一个类方法或者静态方法，最好是类方法
    @classmethod
    def build_vocab(cls,sentences,vocab_path):
        vocab = set()  # 集合好去重
        for sentence in tqdm.tqdm(sentences):
            vocab.update(jieba.lcut(sentence))
        # 先转成列表（为了特殊符号顺序）
        # 增加特殊符号<unk>
        vocab_list = [cls.pad_token,cls.unk_token] + [token for token in vocab if token.strip()!='']
        #去掉了空格
        # 5.词表保存本地（什么格式）
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(vocab_list))
    @classmethod
    def from_vocab(cls,vocab_path):
        with open(vocab_path,'r',encoding='utf-8') as f:
            vocab_list= [line.strip() for line in f.readlines()]
        return cls(vocab_list)
    
if __name__ == '__main__':
    tokenizer=JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print(f'词表大小{tokenizer.vocab_size}')
    print(f'特殊符号{tokenizer.unk_token}')
    encode=tokenizer.encode("天气不错")
    print(type(encode))
    print(f'天气不错encode结果为{encode}')