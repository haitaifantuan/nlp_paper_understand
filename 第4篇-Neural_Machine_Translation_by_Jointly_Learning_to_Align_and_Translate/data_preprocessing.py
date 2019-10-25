#coding=utf-8
'''
Author:Haitaifantuan
'''

import os
import nltk
import pickle
import train_args
import collections


class Data_preprocess(object):
    def __init__(self):
        pass

    def tokenize_corpus(self):
        '''
        该函数的作用是：将英文语料和中文语料进行tokenize，然后保存到本地。
        '''
        # 将英文语料tokenize，保存下来。
        if not os.path.exists(train_args.raw_train_english_after_tokenization_data_path.replace('train.raw.en.after_tokenization.txt', '')):
            os.mkdir(train_args.raw_train_english_after_tokenization_data_path.replace('train.raw.en.after_tokenization.txt', ''))
        fwrite = open(train_args.raw_train_english_after_tokenization_data_path, 'w', encoding='utf-8')
        with open(train_args.raw_train_english_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = nltk.word_tokenize(line)
                # 将tokenization后的句子写入文件
                fwrite.write(' '.join(line) + '\n')
        fwrite.close()

        # 将中文语料tokenize，保存下来。
        if not os.path.exists(train_args.raw_train_chinese_after_tokenization_data_path.replace('train.raw.zh.after_tokenization.txt', '')):
            os.mkdir(train_args.raw_train_chinese_after_tokenization_data_path.replace('train.raw.zh.after_tokenization.txt', ''))
        fwrite = open(train_args.raw_train_chinese_after_tokenization_data_path, 'w', encoding='utf-8')
        with open(train_args.raw_train_chinese_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = list(line)
                # 将tokenization后的句子写入文件
                fwrite.write(' '.join(line) + '\n')
        fwrite.close()

        print('语料tokenization完成')

    def build_token_dictionary(self):
        '''
        该函数的作用是：根据英文语料和中文语料，建立各自的，以字为单位的token dictionary。
        '''
        # 生成英文的token_dictionary
        english_token_id_dictionary = {}
        # 我们定义unk的id是0，unk的意思是，
        # 当句子中碰到token dictionary里面没有的token的时候，就转换为这个
        english_token_id_dictionary['<unk>'] = 0  
        english_token_id_dictionary['<sos>'] = 1  # 我们定义sos的id是1
        english_token_id_dictionary['<eos>'] = 2  # 我们定义eos的id是1
        en_counter = collections.Counter(
        )  # 创建一个英文token的计数器，专门拿来计算每个token出现了多少次
        with open(train_args.raw_train_english_after_tokenization_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split(' ')
                for token in line:
                    en_counter[token] += 1
        most_common_en_token_list = en_counter.most_common(train_args.Source_vocab_size - 3)  # 找出最常见的Source_vocab_size-3的token
        for token_tuple in most_common_en_token_list:
            english_token_id_dictionary[token_tuple[0]] = len(english_token_id_dictionary)
        # 保存english_token_id_dictionary
        if not os.path.exists(train_args.english_token_id_dictionary_pickle_path.replace('english_token_id_dictionary.pickle', '')):
            os.mkdir(train_args.english_token_id_dictionary_pickle_path.replace('english_token_id_dictionary.pickle', ''))
        with open(train_args.english_token_id_dictionary_pickle_path, 'wb') as file:
            pickle.dump(english_token_id_dictionary, file)

        # 生成中文的token_dictionary 以及把 tokenization后的结果保存下来
        chinese_token_id_dictionary = {}
        # 我们定义unk的id是0，unk的意思是，
        # 当句子中碰到token dictionary里面没有的token的时候，就转换为这个
        chinese_token_id_dictionary['<unk>'] = 0  
        chinese_token_id_dictionary['<sos>'] = 1  # 我们定义sos的id是1
        chinese_token_id_dictionary['<eos>'] = 2  # 我们定义eos的id是1
        # 创建一个中文token的计数器，专门拿来计算每个token出现了多少次
        zh_counter = collections.Counter()
        with open(train_args.raw_train_chinese_after_tokenization_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split(' ')
                for token in line:
                    zh_counter[token] += 1
        most_common_zh_token_list = zh_counter.most_common(train_args.Target_vocab_size - 3)  # 找出最常见的Target_vocab_size-3的token
        for token_tuple in most_common_zh_token_list:
            chinese_token_id_dictionary[token_tuple[0]] = len(chinese_token_id_dictionary)
        # 保存token_dictionary
        if not os.path.exists(train_args.chinese_token_id_dictionary_pickle_path.replace('chinese_token_id_dictionary.pickle', '')):
            os.mkdir(train_args.chinese_token_id_dictionary_pickle_path.replace('chinese_token_id_dictionary.pickle', ''))
        with open(train_args.chinese_token_id_dictionary_pickle_path, 'wb') as file:
            pickle.dump(chinese_token_id_dictionary, file)
        print('英文token_dictionary和中文token_dictionary创建完毕')

    def convert_data_to_id_pad_eos(self):
        '''
        该函数的作用是：
        将英文语料转换成id形式，并在末尾添加[EOS]
        将中文语料转换成id形式，并在句子开头添加[SOS]
        '''
        # 读取英文的token_dictionary
        with open(train_args.english_token_id_dictionary_pickle_path, 'rb') as file:
            english_token_id_dictionary = pickle.load(file)

        if not os.path.exists(train_args.train_en_converted_to_id_path.replace('train.en.converted_to_id.txt', '')):
            os.mkdir(train_args.train_en_converted_to_id_path.replace('train.en.converted_to_id.txt', ''))
        fwrite = open(train_args.train_en_converted_to_id_path, 'w', encoding='utf-8')
        # 读取tokenization后的英文语料，并将其转换为id形式。
        with open(train_args.raw_train_english_after_tokenization_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_converted_to_id = []
                line = line.strip().split(' ')
                for token in line:
                    # 将token转换成id
                    token_id = english_token_id_dictionary.get(
                        token, english_token_id_dictionary['<unk>'])
                    line_converted_to_id.append(str(token_id))
                # 在英文语料最后加上EOS
                line_converted_to_id.append(
                    str(english_token_id_dictionary['<eos>']))
                # 写入本地文件
                fwrite.write(' '.join(line_converted_to_id) + '\n')
        fwrite.close()

        # 读取中文的token_dictionary
        with open(train_args.chinese_token_id_dictionary_pickle_path, 'rb') as file:
            chinese_token_id_dictionary = pickle.load(file)

        if not os.path.exists(train_args.train_zh_converted_to_id_path.replace('train.zh.converted_to_id.txt', '')):
            os.mkdir(train_args.train_zh_converted_to_id_path.replace('train.zh.converted_to_id.txt', ''))
        fwrite = open(train_args.train_zh_converted_to_id_path, 'w', encoding='utf-8')
        # 读取tokenization后的中语料，并将其转换为id形式。
        with open(train_args.raw_train_chinese_after_tokenization_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_converted_to_id = []
                line = line.strip().split(' ')
                for token in line:
                    # 将token转换成id
                    token_id = chinese_token_id_dictionary.get(
                        token, english_token_id_dictionary['<unk>'])
                    line_converted_to_id.append(str(token_id))
                # 因为这个中文语料是当做目标词的，因此也需要在中文语料最后面加上EOS
                # decoder的输入的最开始的BOS，会在train.py里面添加。
                line_converted_to_id.append(
                    str(chinese_token_id_dictionary['<eos>']))
                # 写入本地文件
                fwrite.write(' '.join(line_converted_to_id) + '\n')
        fwrite.close()

        print('英文语料转换为id并且添加[EOS]标致完毕')
        print('中文语料转换为id并且添加[EOS]标致完毕')


# 创建预处理data对象
data_obj = Data_preprocess()
# 将英文语料和中文语料进行tokenization
data_obj.tokenize_corpus()
# 创建英文语料和中文语料的token_dictionary
data_obj.build_token_dictionary()
# 根据token_dictionary将英文语料和中文语料转换为id形式
# 并且在英文语料的最后添加[EOS]标致，在中文语料的最开始添加[SOS]标致
# 并将转化后的语料保存下来
data_obj.convert_data_to_id_pad_eos()
