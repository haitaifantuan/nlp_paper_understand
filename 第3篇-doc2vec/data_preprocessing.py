#coding=utf-8
'''
Author:Haitaifantuan
'''

import os
import nltk
import numpy as np
import collections
import pickle
import random


class Data_preprocessing(object):
    '''
    这个类是用来对数据进行预处理，根据aclImdb数据里面，有5万是带情感标签的数据，有5万是不带情感标签的数据。
    带标签的数据中，有2.5万是训练集的标签，另外2.5万是测试集的标签。
    2.5万训练集的数据中，有1.25万是负面情绪的标签，1.25万正面情绪的标签。
    2.5万测试集的数据中，有1.25万是负面情绪的标签，1.25万正面情绪的标签。
    '''
    def __init__(self):
        # 以下参数需要配置=======================================
        self.window_size = 10
        # 以上参数需要配置=======================================
        
        if not os.path.exists('./saved_things/explanation.txt'):
            '''
            # 从本地读取原始语料，并且将其分词后，返回回来。
            self.train_neg_raw_data, self.train_pos_raw_data, self.test_neg_raw_data, self.test_pos_raw_data, self.upsup_raw_data = self.read_all_raw_data_from_local_file()
            # 统计词频，构造常用词词表。
            self.word2id_dictionary = self.construct_word_dictionary()
            '''
            #以下可删除，调试使用
            #with open('./saved_things/all_data_not_combined-可删除.pickle', 'wb') as file:
            #    pickle.dump([self.train_neg_raw_data, self.train_pos_raw_data, self.test_neg_raw_data, self.test_pos_raw_data, self.upsup_raw_data, self.word2id_dictionary], file)
            with open('./saved_things/all_data_not_combined-可删除.pickle', 'rb') as file:
                self.train_neg_raw_data, self.train_pos_raw_data, self.test_neg_raw_data, self.test_pos_raw_data, self.upsup_raw_data, self.word2id_dictionary = pickle.load(file)
            #以上可删除，调试使用
            
            # 根据常用词词表，将raw_data转换为词所对应的id。
            #self.train_zipped, self.test_zipped, self.upsup_raw_data_converted_to_id = self.convert_data_to_word_id_and_shuffle()
            
            #以下可删除，调试使用
            #with open('./saved_things/shuffled_data_combined-可删除.pickle', 'wb') as file:
            #    pickle.dump([self.train_zipped, self.test_zipped, self.upsup_raw_data_converted_to_id], file)
            with open('./saved_things/shuffled_data_combined-可删除.pickle', 'rb') as file:
                self.train_zipped, self.test_zipped, self.upsup_raw_data_converted_to_id = pickle.load(file)
            #以上可删除，调试使用
            
            
            # 基于转换为id后的训练数据，构建input样本。
            self.train_embedding_word_input_data, self.train_embedding_sentence_input, self.train_embedding_labels, self.train_sentiment_input, self.train_sentiment_labels = self.construct_input_data_and_label(self.train_zipped)
            # 基于转换为id后的测试数据，构建input样本。
            self.test_embedding_word_input_data, self.test_embedding_sentence_input, self.test_embedding_labels, self.test_sentiment_input, self.test_sentiment_labels = self.construct_input_data_and_label(self.test_zipped)
            
            #以下可删除，调试使用
            with open('./saved_things/train-input-data-可删除.pickle', 'wb') as file:
                pickle.dump([self.train_embedding_word_input_data, self.train_embedding_sentence_input, self.train_embedding_labels, self.train_sentiment_input, self.train_sentiment_labels], file)
            with open('./saved_things/test-input-data-可删除.pickle', 'wb') as file:
                pickle.dump([self.test_embedding_word_input_data, self.test_embedding_sentence_input, self.test_embedding_labels, self.test_sentiment_input, self.test_sentiment_labels], file)
            #with open('./saved_things/train-input-data-可删除.pickle', 'rb') as file:
            #    self.train_embedding_word_input_data, self.train_embedding_sentence_input, self.train_embedding_labels, self.train_sentiment_input, self.train_sentiment_labels = pickle.load(file)
            #with open('./saved_things/test-input-data-可删除.pickle', 'rb') as file:
            #    self.test_embedding_word_input_data, self.test_embedding_sentence_input, self.test_embedding_labels, self.test_sentiment_input, self.test_sentiment_labels = pickle.load(file)
            #以上可删除，调试使用
            
    def read_raw_data(self, file_path):
        '''
        这个函数的作用是：
        根据传进来的file_path，读取file_path下面所有的txt文件里面的数据（每个文件是一条语料）
        然后将其分词，将分词后的结果，append到一个列表里
        最后返回这个列表。
        '''
        data_list = []
        file_list = os.listdir(file_path)
        for each_file in file_list:
            with open(os.path.join(file_path, each_file), 'r', encoding='utf-8') as file:
                cnt = file.read().strip()
                cnt = cnt.lower()  # 将语料全部转为小写。因为下游任务不是POS之类的，所以变成小写没关系。
                cnt_tokenized = nltk.word_tokenize(cnt)  # 将语料分词
                data_list.append(cnt_tokenized)
        return data_list
    
    def convert_to_id(self, raw_data):
        data_converted_to_id = []
        for sentence in raw_data:
            this_sentence_converted_to_id = []
            for word in sentence:
                # 将词转换为id，如果碰到词表中没有的词，就转换为'<UNK>'的id。
                this_sentence_converted_to_id.append(self.word2id_dictionary.get(word, self.word2id_dictionary['<UNK>']))
            data_converted_to_id.append(this_sentence_converted_to_id)
        return data_converted_to_id
        
    
    def read_all_raw_data_from_local_file(self):
        '''
        这个函数目的是为了读取数据，并提前将它们分好词。
        '''
        # 读取训练集中，1.25万的负面情绪的原始语料，并且将其分词。
        train_neg_raw_data = self.read_raw_data('./aclImdb/train/neg')
        print('训练集中负面情绪的语料读取完毕-----共{}条'.format(len(train_neg_raw_data)))
        # 读取训练集中，1.25万的正面情绪的原始语料，并且将其分词。
        train_pos_raw_data = self.read_raw_data('./aclImdb/train/pos')
        print('训练集中正面情绪的语料读取完毕-----共{}条'.format(len(train_pos_raw_data)))
        
        # 读取测试集中，1.25万的负面情绪的原始语料，并且将其分词。
        test_neg_raw_data = self.read_raw_data('./aclImdb/test/neg')
        print('测试集中负面情绪的语料读取完毕-----共{}条'.format(len(test_neg_raw_data)))
        # 读取测试集中，1.25万的正面情绪的原始语料，并且将其分词。
        test_pos_raw_data = self.read_raw_data('./aclImdb/test/pos')
        print('测试集中正面情绪的语料读取完毕-----共{}条'.format(len(test_pos_raw_data)))
        
        # 读取unsup数据集中，5万的原始语料，并且将其分词。
        upsup_raw_data = self.read_raw_data('./aclImdb/train/unsup')
        print('无人工情绪标签的数据语料读取完毕-----共{}条'.format(len(upsup_raw_data)))
        
        return train_neg_raw_data, train_pos_raw_data, test_neg_raw_data, test_pos_raw_data, upsup_raw_data
    
    
    def construct_word_dictionary(self):
        '''
        这个函数的作用是，
        我们拿训练集以及无人工情感标签的语料，来构建常用词的单词词表。
        在这个代码里，我们不会拿无人工情感标签的语料是拿来训练词向量和句向量的，因此仅仅拿它们来构建词表。
        读者可以自己把无人工情感标签的语料放进去训练试试，看看效果有没有提高。
        '''
        # 先将所有的词，放入到一个列表里，统计出每个词的词频数。然后找出最常用的30000个词构建词表。
        all_words = []
        for eachCorpus in self.train_neg_raw_data:
            all_words.extend(eachCorpus)
        for eachCorpus in self.train_pos_raw_data:
            all_words.extend(eachCorpus)
        for eachCorpus in self.upsup_raw_data:
            all_words.extend(eachCorpus)
        
        # 统计词频数。
        counter = collections.Counter(all_words)
        common_words = dict(counter.most_common(29998))
        word2id_dictionary = {'<UNK>':0, '<PAD>':1}  # 除了29998个常用词外，其他所有的词都转为'<UNK>'
        for eachWord in common_words:
            word2id_dictionary[eachWord] = len(word2id_dictionary)
            
        return word2id_dictionary
    
    
    def convert_data_to_word_id_and_shuffle(self):
        '''
        该函数的作用：
        根据常用词词表，将raw_data转换为词所对应的id。
        并且将pos数据和neg数据合并掉，
        '''
        train_neg_data_converted_to_id = self.convert_to_id(self.train_neg_raw_data)  # 将训练集中，带有标签的负样本的词都转换为id。
        train_pos_data_converted_to_id = self.convert_to_id(self.train_pos_raw_data)  # 将训练集中，带有标签的正样本的词都转换为id。
        test_neg_data_converted_to_id = self.convert_to_id(self.test_neg_raw_data)  # 将测试集中，带有标签的负样本的词都转换为id。
        test_pos_data_converted_to_id = self.convert_to_id(self.test_pos_raw_data)  # 将训练集中，带有标签的正样本的词都转换为id。
        upsup_raw_data_converted_to_id = self.convert_to_id(self.upsup_raw_data)  # 将训练集中，无情感标签的样本的词都转换为id。

        # 将训练集的正样本和负样本进行合并，同时将训练数据和情感标签的labelzip在一起，并且shuffle。
        train_data = train_neg_data_converted_to_id + train_pos_data_converted_to_id
        train_sentiment_label = [0] * len(train_neg_data_converted_to_id) + [1] * len(train_pos_data_converted_to_id)
        train_zipped = list(zip(train_data, train_sentiment_label))
        random.shuffle(train_zipped)
        
        # 将测试集的正样本和负样本进行合并，并且shuffle。情感类别的label也和样本的顺序保持一致。
        test_data = test_neg_data_converted_to_id + test_pos_data_converted_to_id
        test_sentiment_label = [0] * len(test_neg_data_converted_to_id) + [1] * len(test_pos_data_converted_to_id)
        test_zipped = list(zip(test_data, test_sentiment_label))
        random.shuffle(test_zipped)
        
        return train_zipped, test_zipped, upsup_raw_data_converted_to_id
        
        
    def construct_input_data_and_label(self, data_zipped):
        '''
        该函数的作用：
        构建句向量训练阶段的input数据。
        构建情感分类训练阶段的input数据。
        '''
        #以下是构造train的输入数据==================================================================================================================
        embedding_word_input_data = []  # 窗口中，最后一个词为embedding_train_labels，其他的词为input
        embedding_sentence_input = []  # 放的是对应句子的id
        embedding_labels = []
        sentiment_input = []
        sentiment_labels = []
        
        sentence_id = 0
        for eachSentence_tuple in data_zipped:  # eachSentence_tuple[0]是句子转换成id后的样子，eachSentence_tuple[1]是句子的情感分类的类别。
            # 首先判断句长是否大于self.window_size，如果小于它，那就将句子的进行pad，补在句子最前面。
            if len(eachSentence_tuple[0]) < 10:
                if set(eachSentence_tuple[0][0:-1]) == self.word2id_dictionary['<UNK>']:
                    # 如果所有的词都是'<UNK>'，那这个样本就抛弃了
                    pass
                else:
                    # append'<PAD>'的id。句子的最后一个词为训练句向量时的目标词。
                    temp_embedding_word_input_data = []
                    temp_embedding_word_input_data.extend([self.word2id_dictionary['<PAD>']]*(self.window_size-len(eachSentence_tuple[0])))
                    temp_embedding_word_input_data.extend(eachSentence_tuple[0][0:-1])
                    embedding_word_input_data.append(temp_embedding_word_input_data)
                    embedding_sentence_input.append([sentence_id])
                    embedding_labels.append([eachSentence_tuple[0][-1]])  # 把最后一个词作为label
                    # 构造sentiment_train_input和sentiment_train_labels
                    sentiment_input.append([sentence_id])
                    sentiment_labels.append([eachSentence_tuple[1]])
                    sentence_id += 1
            else:
                for input_begin_index in range(len(eachSentence_tuple[0])):
                    # 如果窗口已经到达最后一个了，那就break
                    if input_begin_index == (len(eachSentence_tuple[0])-self.window_size+1):
                        sentiment_input.append([sentence_id])
                        sentiment_labels.append([eachSentence_tuple[1]])
                        sentence_id += 1
                        break
                    else:
                        embedding_word_input_data.append(eachSentence_tuple[0][input_begin_index:input_begin_index+9])
                        embedding_sentence_input.append([sentence_id])
                        embedding_labels.append([eachSentence_tuple[0][input_begin_index+9]])
                        
        return embedding_word_input_data, embedding_sentence_input, embedding_labels, sentiment_input, sentiment_labels
                
                
        
        








#t = Data_preprocessing()



