# coding: utf-8

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""word2vec skip-gram code"""
# 导入一些需要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import os
import random
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


#参数配置=================================
corpus_url = 'http://mattmahoney.net/dc/'  # 语料库的下载地址
#参数配置=================================

def downloadFromInternet(corpus_url, corpus_name):
    """
    函数作用：
    如果本地数据集不存在，就下载，否则跳过
    """
    if not os.path.exists(corpus_name):
        corpus_name, _ = urllib.request.urlretrieve(corpus_url + corpus_name, corpus_name)
    
    return corpus_name


def read_data(filename):
    """
    函数作用：
    将件解压，然后读取为word的list。
    """
    with zipfile.ZipFile(filename) as f:
        data_list = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data_list


def build_dataset(data_list, max_vocabulary_size):
    """
    函数作用：
    首先，创建词典，然后将原始的单词表示成词对应的index
    """
    word_count = [['UNK', 0]]
    word_count.extend(collections.Counter(data_list).most_common(max_vocabulary_size - 1))
    dictionary = {}
    
    # 构建词典
    for word, word_frequency in word_count:
        dictionary[word] = len(dictionary)
    
    # 将data_list里面的词，转换成index，同时统计下词频数。
    data_index_list = []
    unk_count = 0
    for word in data_list:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # UNK的index为0
            unk_count += 1
        data_index_list.append(index)
      
    word_count[0][1] = unk_count  # 将'UNK'的频数改变一下
    
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return data_index_list, word_count, dictionary, reversed_dictionary


def batch_generation(batch_size=8, num_skips=2, skip_window=1):
    '''
    函数作用：
    产生batch_size个样本
    num_skips就是一个窗口中，采几个样本。因此，batch_size有一定需要是num_skips的倍数，不然不能整除。
    默认情况下skip_window=1, num_skips=2。
    此时就是从连续的3(3 = skip_window*2 + 1)个词中生成2(num_skips)个样本。
    例子：假如连续的三个词['used', 'against', 'early']，并且batch_size=8, num_skips=2, skip_window=1
    那么生成两个样本就是：against（输入） -> used（目标预测词）, against（输入） -> early（目标预测词）
    '''
    # data_index相当于一个指针，初始为0
    # 每次生成一个batch，data_index就会相应地往后推
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ] skip_window可以看成是窗口中心的词的两边词的数量。
    buffer = collections.deque(maxlen=span)  # 是一个队列，FIFO的队列，其实就是滑动窗口。窗口大小为span
    
    for _ in range(span):
        buffer.append(data[data_index])  # data_index是当前数据开始的位置，产生batch_size个样本后就往后推1位（产生batch）。
        data_index = (data_index + 1) % len(data)  # 一个周期结束后，重新回到开头。
        
    for i in range(batch_size // num_skips):
        # i表示第几次滑动窗口
        # buffer是一个长度为 2 * skip_window + 1长度的word list
        # 一个buffer生成num_skips个数的样本
        # 目标词在窗口的中心位置，由于默认skip_window为1，因此窗口大小为3，因此位于窗口的skip_window索引位置的就是中心词。
        # 中心词在skip-gram中是输入，在CBOW中是预测的目标词。
        target = skip_window  
        
        targets_to_avoid = [skip_window]
        
        for j in range(num_skips):
            while target in targets_to_avoid:  # targets_to_avoid保证样本不重复
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        # 每次num_skips个样本，这个窗口内采样就结束了，data_index就向后推进一位
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def plot_with_labels(low_dim_embs, labels, file_saved_path='wordvec_visualization.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    
    plt.savefig(file_saved_path)
  
  

# 步骤1: 下载语料库text8.zip，如果当前文件路径存在，就不下载，否则下载。
filename = downloadFromInternet(corpus_url, 'text8.zip')
data_list = read_data(filename)  # 将语料库解压，并转换成一个word的list

# 步骤2: 制作词典（也称为词表），同时，将不常见的词变成UNK标识符。
vocabulary_size = 50000  # 假定我们只取前50000个高频词形成词典。
data, count, dictionary, reverse_dictionary = build_dataset(data_list, vocabulary_size)
del data_list  # 删除已节省内存

# 步骤3: 建模
#模型参数======================================================================
batch_size = 128
embedding_size = 128  # 词嵌入空间是128维的。即word2vec中的vec是一个128维的向量
skip_window = 1       # skip_window参数和之前保持一致。就是skip-gram window
num_skips = 2         # num_skips参数和之前保持一致
num_sampled = 64      # 构造损失时选取的噪声词的数量
#模型参数======================================================================

valid_size = 16     # 在训练过程中，会对模型进行验证 ，找出和某个词最近的词，每次验证16个词
valid_window = 100  # 这16个词是在前100个最常见的词中选出来的
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 只对前valid_window的词进行验证

graph = tf.Graph()
with graph.as_default():
    # 定义输入和label的占位符
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    # 虽然skwip-gram模型要预测的是中心词两边的词，但是实际上每次只放入一个词作为label，而不是中心词两边的所有的词。
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # 用于验证的词
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
    # 我们在cpu上定义模型，可以改成'/gpu:0'
    with tf.device('/gpu:0'):
        # 定义embeddings变量，其实就是词向量矩阵。
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 利用embedding_lookup可以轻松得到一个batch内的所有的词向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
        # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
    # tf.nn.nce_loss会自动选取噪声词，并且形成损失。tf.nn.nce_loss将矩阵相乘封装好了，直接传入接口需要的参数就可以了。
    # 随机选取num_sampled个噪声词
    # 由于并不需要即将预测的值的词向量，因此我们这里没有传入。只需要传入目标词的下标（train_labels），以及词典大小就可以了（vocabulary_size）。
    # loss计算的是一个batch的一个词的平均损失
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))
  
    # 得到loss后，我们就可以构造优化器了
    train_operation = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  
    # 计算词和词的相似度，在这里，我们提前计算每个词向量的二范式，因为余弦相似度的公式的分母中，是两个向量的二范式。
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    
    # 找出和验证词的embedding并计算它们和所有单词的相似度
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
  
    # 变量初始化operation。
    init = tf.global_variables_initializer()


# 步骤4：开始训练
data_index = 0  # 相当于是data的指针，代表着当前的窗口的中心滑动到的位置。 
num_steps = 100000+1
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(graph=graph, config=config) as session:  # 打开session
    init.run()  # 初始化变量
    
    total_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = batch_generation(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        # 一个step获取一个batch_size的样本，优化一次。
        _, loss_val = session.run([train_operation, loss], feed_dict=feed_dict)
        total_loss += loss_val
        
        if (step+1) % 2000 == 0:
            average_loss = total_loss/2000  # 2000个batch的平均损失
            print('Average loss at step {}-----loss is {}'.format((step+1), average_loss))
            total_loss = 0
        
        # 每1万步，我们进行一次验证
        if (step+1) % 10000 == 0:
            # sim是验证词与所有词之间的相似度
            sim = similarity.eval()
            # 一共有valid_size个验证词
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # 输出最相邻的8个词语
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    # word_embedding是我们最后得到的embedding向量
    # 它的形状是[vocabulary_size, embedding_size]
    # 每一行就代表着对应index词（也就是某个词在词典中的编号）的词向量表示
    #word_embedding_matrix = normalized_embeddings.eval()  # 有些人把正则化的向量作为词向量。
    word_embedding_matrix = embeddings.eval()  # 有些人把未正则化的向量作为词向量。


# 步骤5: 可视化
# embedding的维度为128维，没法可视化，因此对其进行降维
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# 画出words_plot_num个词的位置
words_plot_num = 100
low_dim_embs = tsne.fit_transform(word_embedding_matrix[:words_plot_num, :])
labels = [reverse_dictionary[i] for i in xrange(words_plot_num)]
plot_with_labels(low_dim_embs, labels)



