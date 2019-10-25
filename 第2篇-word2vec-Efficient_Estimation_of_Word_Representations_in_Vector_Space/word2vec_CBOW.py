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
"""word2vec CBOW example."""
# 导入一些需要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
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
        corpus_name, _ = urllib.request.urlretrieve(corpus_url + corpus_name,corpus_name)

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


def batch_generation(batch_size, cbow_window):
    '''
    函数作用：
    产生batch_size个样本，cbow_window就是CBOW模型的滑动窗口中
    中心词左边或者右边词数量。
    '''
    global data_index
    assert cbow_window % 2 == 1  # 确保滑动窗口大小为奇数
    span = 2 * cbow_window + 1  # 指的就是滑动窗口的大小
    # 去除中心word: span - 1
    train_batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    buffer = collections.deque(maxlen=span)  # FIFO的队列
    for _ in range(span):
        buffer.append(data[data_index])
        # 循环选取 data中数据，到尾部则从头开始
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size):
        # target at the center of span。在CBOW模型中，也就是目标词的索引。
        target = cbow_window

        col_idx = 0
        for j in range(span):
            if j == span // 2:  # 确保模型输入不是滑动窗口中的中心词。
                continue
            train_batch[i, col_idx] = buffer[j]
            col_idx += 1
        labels[i, 0] = buffer[target]
        # 更新 buffer
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return train_batch, labels


def plot_with_labels(low_dim_embs,
                     labels,
                     file_saved_path='./wordvec_visualization.png'):
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
data, count, dictionary, reverse_dictionary = build_dataset(
    data_list, vocabulary_size)
del data_list  # 删除已节省内存

# 步骤3: 建模
#模型参数======================================================================
batch_size = 256
embedding_size = 128  # 词嵌入空间是128维的。即word2vec中的vec是一个128维的向量
cbow_window = 1  # cbow_window参数和之前保持一致。就是CBOW模型的滑动窗口中，中心词左边或者右边词数量。
num_skips = 2  # num_skips参数和之前保持一致
num_sampled = 64  # 构造损失时选取的噪声词的数量
#模型参数======================================================================

data_index = 0
num_steps = 100000 + 1
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
# pick 16 samples from 100
valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
valid_examples = np.append(
    valid_examples,
    random.sample(range(1000, 1000 + valid_window), valid_size // 2))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()
with graph.as_default():
    # 如果要使用gpu，请将import os后面那句话释放到代码里。
    # Input data.定义输入和label的占位符。因为输入的是窗口中心词两边的词，所以是大小为2 * cbow_window
    train_dataset = tf.placeholder(tf.int32,
                                   shape=[batch_size, 2 * cbow_window
                                          ])  # 传进来的train_dataset是词的索引
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 我们在cpu上定义模型，可以改成'/gpu:0'。只需要将变量放在cpu或者gpu上就可以了。其他操作不需要定义，
    # 因为操作一般都是对变量进行操作的。只要变量存放在CPU或者GPU上了，它的操作自然就是在CPU或者GPU上了。
    with tf.device('/gpu:0'):
        # Variables.
        # embedding, vector for each word in the vocabulary。embeddings变量就是词向量矩阵。
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    # this might efficiently find the embeddings for given ids (traind dataset)
    # manually doing this might not be efficient given there are 50000 entries in embeddings
    embeds = None
    for i in range(2 * cbow_window):
        # 从词向量矩阵中，根据index找到对应的词的词向量。
        embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
        print('embedding %d shape: %s' % (i, embedding_i.get_shape().as_list()))
        emb_x, emb_y = embedding_i.get_shape().as_list()
        if embeds is None:
            embeds = tf.reshape(embedding_i, [emb_x, emb_y, 1])
        else:
            embeds = tf.concat([embeds, tf.reshape(embedding_i, [emb_x, emb_y, 1])], 2)

    assert embeds.get_shape().as_list()[2] == 2 * cbow_window
    print("Concat embedding size: %s" % embeds.get_shape().as_list())
    # 将每个batch中，每个词的词向量的每个维度相加取平均。而原始论文是用的是sum，并不是求平均。
    # avg_embed其实就是隐层的结果了。然后这个avg_embed会被作为隐层的输出，传递给输出层。
    avg_embed = tf.reduce_mean(embeds, 2, keep_dims=False)
    print("Avg embedding size: %s" % avg_embed.get_shape().as_list())

    # loss计算的是一个batch的一个词的平均损失
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights,
                       nce_biases,
                       labels=train_labels,
                       inputs=avg_embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    # Adagrad is required because there are too many things to optimize
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    # 计算词和词的相似度，在这里，我们提前计算每个词向量的二范式，因为余弦相似度的公式的分母中，是两个向量的二范式。
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # 找出和验证词的embedding并计算它们和所有单词的相似度
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings,
                           tf.transpose(normalized_embeddings))

    # 变量初始化operation。
    init = tf.global_variables_initializer()

# 步骤4：打开session，开始训练
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(graph=graph, config=config) as session:
    session.run(init)

    total_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = batch_generation(batch_size, cbow_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        total_loss += l
        if (step + 1) % 2000 == 0:
            average_loss = total_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % ((step + 1), average_loss))
            total_loss = 0

        # 计算验证集中的词的近义词是什么。
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

    # 有些人把正则化的向量作为词向量。
    # word_embedding_matrix = normalized_embeddings.eval()  
    # 有些人把未正则化的向量作为词向量。
    word_embedding_matrix = embeddings.eval()  

# 步骤5：可视化
# embedding的维度为128维，没法可视化，因此对其进行降维
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# 画出words_plot_num个词的位置
words_plot_num = 100
low_dim_embs = tsne.fit_transform(word_embedding_matrix[:words_plot_num, :])
labels = [reverse_dictionary[i] for i in xrange(words_plot_num)]
plot_with_labels(low_dim_embs, labels)
