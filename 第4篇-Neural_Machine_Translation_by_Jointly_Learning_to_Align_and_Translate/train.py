#coding=utf-8
'''
Author:Haitaifantuan
'''

import tensorflow as tf
import train_args
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 如果有GPU的同学，可以把这个打开，或者自己研究下怎么打开。

# 首先判断模型保存的路径存不存在，不存在就创建
if not os.path.exists('./saved_things/'):
    os.mkdir('./saved_things/')
if not os.path.exists('./saved_things/doesnt_finish_training_model/'):
    os.mkdir('./saved_things/doesnt_finish_training_model/')
if not os.path.exists('./saved_things/finish_training_model/'):
    os.mkdir('./saved_things/finish_training_model/')


mt_graph = tf.Graph()  # 创建machine translation 专用的graph
data_graph = tf.Graph()  # 创建数据专用的graph

class data_batch_generation(object):
    def __init__(self):
        with data_graph.as_default():  # 定义在这个图下面创建模型
            # 通过tf.data.TextLineDataset()来读取训练集数据
            self.src_data = tf.data.TextLineDataset(train_args.train_en_converted_to_id_path)
            self.trg_data = tf.data.TextLineDataset(train_args.train_zh_converted_to_id_path)

            # 因为刚读进来是string格式，这里将string改为int，并形成tensor形式。
            self.src_data = self.src_data.map(lambda line: tf.string_split([line], delimiter=' ').values)
            self.src_data = self.src_data.map(lambda line: tf.string_to_number(line, tf.int32))
            self.trg_data = self.trg_data.map(lambda line: tf.string_split([line], delimiter=' ').values)
            self.trg_data = self.trg_data.map(lambda line: tf.string_to_number(line, tf.int32))

            # 为self.src_data添加一下每个句子的长度
            self.src_data = self.src_data.map(lambda x: (x, tf.size(x)))
            # 为self.trg_data添加一下decoder的输入。形式为(dec_input, trg_label, trg_length)
            # tf.size(x)后面计算loss的时候拿来mask用的以及使用tf.nn.bidirectional_dynamic_rnn()这个函数的时候使用的。
            self.trg_data = self.trg_data.map(lambda x: (tf.concat([[1], x[:-1]], axis=0), x, tf.size(x)))

            # 将self.src_data和self.trg_data zip起来，方便后面过滤数据。
            self.data = tf.data.Dataset.zip((self.src_data, self.trg_data))

            # 将句子长度小于1和大于train_args.train_max_sent_len的都去掉。
            def filter_according_to_length(src_data, trg_data):
                ((enc_input, enc_input_size), (dec_input, dec_target_label, dec_target_label_size)) = (src_data, trg_data)
                enc_input_flag = tf.logical_and(tf.greater(enc_input_size, 1), tf.less_equal(enc_input_size, train_args.train_max_sent_len))
                # decoder的input的长度和decoder的label是一样的，所以这里可以这样用。
                dec_input_flag = tf.logical_and(tf.greater(dec_target_label_size, 1), tf.less_equal(dec_target_label_size, train_args.train_max_sent_len))
                flag = tf.logical_and(enc_input_flag, dec_input_flag)
                return flag

            self.data = self.data.filter(filter_according_to_length)

            # 由于句子长短不同，我们这里将句子的长度pad成固定的，pad成当前batch里面最长的那个。
            # 我们使用0来pad，也就是['<unk>']标志
            # 后续计算loss的时候，会根据trg_label的长度来mask掉pad的部分。
            # 设置为None的时候，就代表把这个句子pad到当前batch的样本下最长的句子的长度。
            # enc_input_size本来就是单个数字，因此不用pad。
            self.padded_data = self.data.padded_batch(batch_size=train_args.train_batch_size,
                                    padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])),
                                    (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))))
            self.padded_data = self.padded_data.shuffle(10000)

            # 创建一个iterator
            self.padded_data_iterator = self.padded_data.make_initializable_iterator()
            self.line = self.padded_data_iterator.get_next()

    def iterator_initialization(self, sess):
        # 初始化iterator
        sess.run(self.padded_data_iterator.initializer)

    def next_batch(self, sess):
        # 获取一个batch_size的数据
        ((enc_inp, enc_size), (dec_inp, dec_trg, dec_trg_size))= sess.run(self.line)
        return ((enc_inp, enc_size), (dec_inp, dec_trg, dec_trg_size))


class Model(object):
    def __init__(self):
        with mt_graph.as_default():
            # 创建placeholder
            with tf.variable_scope("ipt_placeholder"):
                self.enc_inp = tf.placeholder(tf.int32, shape=[train_args.train_batch_size, None])  # 第一个None因为batch_size在变化，第2个None是因为句长不确定
                self.enc_inp_size = tf.placeholder(tf.int32, shape=[train_args.train_batch_size])  # None是代表batch_size
                self.dec_inp = tf.placeholder(tf.int32, shape=[train_args.train_batch_size, None])  # None是因为句长不确定
                self.dec_label = tf.placeholder(tf.int32, shape=[train_args.train_batch_size, None])  #None是因为句长不确定
                self.dec_label_size = tf.placeholder(tf.int32, shape=[train_args.train_batch_size])  # None是代表batch_size
            
            # 创建源语言的token的embedding和目标语言的token的embedding
            with tf.variable_scope("token_embedding"):
                # 源语言的token的embedding
                self.src_embedding = tf.Variable(initial_value=tf.truncated_normal(shape=[train_args.Source_vocab_size, train_args.RNN_hidden_size], dtype=tf.float32), trainable=True)
                # 目标语言的token的embedding
                self.trg_embedding = tf.Variable(initial_value=tf.truncated_normal(shape=[train_args.Target_vocab_size, train_args.RNN_hidden_size], dtype=tf.float32), trainable=True)
                # 全连接层的参数
                if train_args.Share_softmax_embedding:
                    self.full_connect_weights = tf.transpose(self.trg_embedding)
                else:
                    self.full_connect_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[train_args.RNN_hidden_size, train_args.Target_vocab_size], dtype=tf.float32), trainable=True)
                self.full_connect_biases = tf.Variable(initial_value=tf.truncated_normal(shape=[train_args.Target_vocab_size], dtype=tf.float32))

            with tf.variable_scope("encoder"):
                # 根据输入，得到输入的token的向量
                self.src_emb_inp = tf.nn.embedding_lookup(self.src_embedding, self.enc_inp)
                # 构建编码器中的双向LSTM
                self.enc_forward_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=train_args.RNN_hidden_size)
                self.enc_backward_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=train_args.RNN_hidden_size)
                # 使用bidirectional_dynamic_rnn构造双向RNN网络。把输入的token的向量放入到encoder里面去，得到输出。
                # enc_top_outputs包含了前向LSTM和反向LSTM的输出。enc_top_states也一样。
                # 我们把前向的LSTM顶层的outputs和反向的LSTM顶层的outputs concat一下，作为attention的输入。
                # enc_top_outputs这个tuple，每一个元素的shape都是[batch_size, time_step, hidden_size]
                self.enc_top_outputs, self.enc_top_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.enc_forward_lstm_cell, cell_bw=self.enc_backward_lstm_cell,
                                                                    inputs=self.src_emb_inp, sequence_length=self.enc_inp_size, dtype=tf.float32)
                self.enc_outpus = tf.concat([self.enc_top_outputs[0], self.enc_top_outputs[1]], -1)

            with tf.variable_scope("decoder"):
                # 创建多层decoder。
                self.dec_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=train_args.RNN_hidden_size)
                                                                for _ in range(train_args.num_decoder_layers)])
                # 选择BahdanauAttention作为注意力机制。它是使用一层隐藏层的前馈神经网络。
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=train_args.RNN_hidden_size, memory=self.enc_outpus, 
                                                                            memory_sequence_length=self.enc_inp_size)
                # 将elf.dec_lstm_cell和attention_mechanism封装成更高级的API
                after_attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_lstm_cell,
                                                                            attention_mechanism, attention_layer_size=train_args.RNN_hidden_size)
                # 目标token的embedding
                self.trg_emb_inp = tf.nn.embedding_lookup(self.trg_embedding, self.dec_inp)
                self.dec_top_outpus, self.dec_states = tf.nn.dynamic_rnn(after_attention_cell, self.trg_emb_inp, self.dec_label_size, dtype=tf.float32)

            # 将输出经过一个全连接层
            self.outpus = tf.reshape(self.dec_top_outpus, [-1, train_args.RNN_hidden_size])  # shape=[None, 1024]
            self.logits = tf.matmul(self.outpus, self.full_connect_weights) + self.full_connect_biases  # shape=[None, 4003]

            # tf.nn.sparse_softmax_cross_entropy_with_logits可以不需要将label变成one-hot形式，减少了步骤，大家后续可以自己尝试下。
            self.dec_label_reshaped = tf.reshape(self.dec_label, [-1])

            # 将self.dec_label_reshaped转换成one-hot的形式
            self.dec_label_after_one_hot = tf.one_hot(self.dec_label_reshaped, train_args.Target_vocab_size)

            # 计算交叉熵损失函数
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.dec_label_after_one_hot, logits=self.logits)

            # 由于我们在构造数据的时候，将没到长度的地方用[UNK]补全了，因此这些地方的loss不能参与计算，我们要将它们mask掉。
            # 这里我们设置dtype=tf.float32，意思是让没有mask掉的地方输出为1，被mask掉的地方输出为0，方便我们后面做乘积。
            # 如果不设置dtype=tf.float32的话，默认输出是True或者False
            self.mask_result = tf.sequence_mask(lengths=self.dec_label_size, maxlen=tf.shape(self.dec_inp)[1], dtype=tf.float32)
            self.mask_result = tf.reshape(self.mask_result, [-1])

            self.loss = self.loss * self.mask_result
            self.loss = tf.reduce_sum(self.loss)
            # 计算平均损失
            self.per_token_loss = self.loss / tf.reduce_sum(self.mask_result)

            # 定义train操作
            self.trainable_variables = tf.trainable_variables()
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=train_args.learning_rate)  # 定义optimizer
            # 计算梯度
            self.grads = tf.gradients(self.loss / tf.to_float(train_args.train_batch_size), self.trainable_variables)
            # 设定一个最大的梯度值，防止梯度爆炸。
            self.grads, _ = tf.clip_by_global_norm(self.grads, 7)
            # apply 梯度到每个Variable上去。
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.trainable_variables))

            # 构建global_step，后面保存模型的时候使用
            self.global_step = tf.Variable(initial_value=0)
            self.global_step_op = tf.add(self.global_step, 1)
            self.global_step_assign = tf.assign(self.global_step, self.global_step_op)
            self.global_step_per_epoch = tf.Variable(initial_value=1000000)

            

    def train(self, sess, data):
        # 训练的操作
        ((enc_inp, enc_size), (dec_inp, dec_trg, dec_trg_size)) = data
        feed = {self.enc_inp:enc_inp, self.enc_inp_size:enc_size,
                    self.dec_inp:dec_inp, self.dec_label:dec_trg, self.dec_label_size:dec_trg_size}

        _, per_token_loss, current_global_step = sess.run([self.train_op, self.per_token_loss, self.global_step_assign], feed_dict=feed)

        return per_token_loss, current_global_step


data_batch_generation_obj = data_batch_generation()
sess_data = tf.Session(graph=data_graph)  # 创建一个图专用的session

nm_model = Model()

session_config = tf.ConfigProto(allow_soft_placement=True)  # sesstion的config
session_config.gpu_options.allow_growth  = True
# 打开Sesstion，开始训练模型
with tf.Session(graph=mt_graph) as sess:  # 创建一个模型的图的sesstion
    saver = tf.train.Saver(max_to_keep=5)  # 构建saver
    data_batch_generation_obj.iterator_initialization(sess_data)
    sess.run(tf.global_variables_initializer())
    current_epoch = 0

    # 从未训练完的模型加载，继续断点训练。
    if os.path.exists(train_args.doesnt_finish_model_saved_path_cheackpoint):
        restore_path = tf.train.latest_checkpoint(train_args.doesnt_finish_model_saved_path.replace('/model', ''))
        saver.restore(sess, restore_path)
        current_epoch = sess.run(nm_model.global_step) // sess.run(nm_model.global_step_per_epoch)
        print('从未训练完的模型加载-----未训练完的模型已训练完第{}个epoch-----共需要训练{}个epoch'.format(current_epoch, train_args.max_global_epochs))

    global_step_per_epoch_count = 0
    while sess.run(nm_model.global_step) < train_args.max_global_epochs * sess.run(nm_model.global_step_per_epoch):
        try:
            data = data_batch_generation_obj.next_batch(sess_data)  # 这里要传入data的sesstion
            if data[0][0].shape[0] == train_args.train_batch_size:
                per_token_loss, current_global_step = nm_model.train(sess, data)
                print("当前为第{}个epoch-----第{}个global_step-----每个token的loss是-----{}".format(current_epoch, current_global_step, per_token_loss))
                global_step_per_epoch_count += 1
        except tf.errors.OutOfRangeError as e:
            current_epoch += 1
            with mt_graph.as_default():
                _ = sess.run(tf.assign(nm_model.global_step_per_epoch, global_step_per_epoch_count))
            global_step_per_epoch_count = 0

            # 如果报tf.errors.OutOfRangeError这个错，说明数据已经被遍历完了，也就是一个epoch结束了。我们重新initialize数据集一下，进行下一个epoch。
            data_batch_generation_obj.iterator_initialization(sess_data)  # 这里要传入data的sesstion
            # 暂时保存下未训练完的模型
            if current_epoch % train_args.num_epoch_per_save == 0:
                saver.save(sess=sess, save_path=train_args.doesnt_finish_model_saved_path, global_step=sess.run(nm_model.global_step))
    
    # 跳出while循环说明整个global_epoch训练完毕，那就保存最终训练好的模型。
    saver.save(sess=sess, save_path=train_args.finish_model_saved_path, global_step=sess.run(nm_model.global_step))