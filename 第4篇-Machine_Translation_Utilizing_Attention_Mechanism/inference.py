#coding=utf-8
'''
Author:Haitaifantuan
'''

import tensorflow as tf
import train_args
import pickle
import nltk


mt_graph = tf.Graph()

class Model(object):
    def __init__(self):
        with mt_graph.as_default():
            # 创建placeholder
            with tf.variable_scope("ipt_placeholder"):
                self.enc_inp = tf.placeholder(tf.int32, shape=[1, None])  # None是因为句长不确定
                self.enc_inp_size = tf.placeholder(tf.int32, shape=[1])  # batch_size是1
            
            # 创建源语言的token的embedding和目标语言的token的embedding
            with tf.variable_scope("token_embedding"):
                # 源语言的token的embedding。这一层里面的都是变量，resotre的时候会被恢复。
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
                self.src_emb_inp = tf.nn.embedding_lookup(self.src_embedding, self.enc_inp)  # 这是变量，resotre的时候会被恢复。
                # 构建编码器中的双向LSTM
                self.enc_forward_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=train_args.RNN_hidden_size)  # 这是变量，resotre的时候会被恢复。
                self.enc_backward_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=train_args.RNN_hidden_size)  # 这是变量，resotre的时候会被恢复。
                # 使用bidirectional_dynamic_rnn构造双向RNN网络。把输入的token的向量放入到encoder里面去，得到输出。
                # enc_top_outputs包含了前向LSTM和反向LSTM的输出。enc_top_states也一样。
                # 我们把前向的LSTM顶层的outputs和反向的LSTM顶层的outputs concat一下，作为attention的输入。
                # enc_top_outputs这个tuple，每一个元素的shape都是[batch_size, time_step, hidden_size]
                # 这一层以下两个操作，不是变量。resotre的时候对它们没有影响。
                self.enc_top_outputs, self.enc_top_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.enc_forward_lstm_cell, cell_bw=self.enc_backward_lstm_cell,
                                                                    inputs=self.src_emb_inp, sequence_length=self.enc_inp_size, dtype=tf.float32)
                self.enc_outpus = tf.concat([self.enc_top_outputs[0], self.enc_top_outputs[1]], -1)

            with tf.variable_scope("decoder"):
                # 创建多层decoder。这是变量，resotre的时候会被恢复。
                self.dec_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=train_args.RNN_hidden_size)
                                                                for _ in range(train_args.num_decoder_layers)])
                # 选择BahdanauAttention作为注意力机制。它是使用一层隐藏层的前馈神经网络。这个操作不是变量。resotre的时候对它们没有影响。
                self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=train_args.RNN_hidden_size, memory=self.enc_outpus, 
                                                                            memory_sequence_length=self.enc_inp_size)
                # 将self.dec_lstm_cell和self.attention_mechanism封装成更高级的API。这个操作不是变量。resotre的时候对它们没有影响。
                self.after_attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_lstm_cell,
                                                                            self.attention_mechanism, attention_layer_size=train_args.RNN_hidden_size)

            # 这里的tf.variable_scope()一定是"decoder/rnn/attention_wrapper"。否则decoder的参数加载不进来。
            # 大家可以在train.py文件和这个文件里面写tf.trainable_variables()，然后打断点查看下变量以及变量域
            with tf.variable_scope("decoder/rnn/attention_wrapper"):
                # 这里我们使用变长的tf.TensorArray()来放置decoder的输入和输出内容。
                self.dec_inp = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True, clear_after_read=False)
                # 我们先在self.dec_inp里放入[SOS]的id，代表开始标致。
                self.dec_inp = self.dec_inp.write(0, 1)  # 1代表[SOS]的id
                # 我们接下去会使用tf.while_loop()来不断的让decoder输出，因此我们需要提前定义好两个函数。
                # 一个是循环条件，另一个是循环体，还有一个是初始变量。
                # 我们先来定义初始变量，decoder有状态，输入两个变量，我们还要加一个step_count变量。
                # 当step_count超出我们设定的范围的时候，就跳出循环。防止decoder无休止的产生outputs。
                init_dec_state = self.after_attention_cell.zero_state(batch_size=1, dtype=tf.float32)
                input_index_ = 0
                init_variables = (init_dec_state, self.dec_inp, input_index_)

                def continue_loop_condition(state, dec_inp, input_index):
                    end_flag = tf.not_equal(dec_inp.read(input_index), 2)  # 2代表[EOS]的标致
                    length_flag = tf.less_equal(input_index, train_args.test_max_output_sentence_length)
                    continue_flag = tf.logical_and(end_flag, length_flag)
                    continue_flag = tf.reduce_all(continue_flag)
                    return continue_flag

                def loop_body_func(state, dec_inp, input_index):
                    # 读取decoder的输入
                    inp = [dec_inp.read(input_index)]
                    inp_embedding = tf.nn.embedding_lookup(self.trg_embedding, inp)

                    # 调用call函数，向前走一步
                    new_output, new_state = self.after_attention_cell.call(state=state, inputs=inp_embedding)
                    
                    # 将new_output再做一次映射，映射到字典的维度
                    # 先将它reshape一下。
                    new_output = tf.reshape(new_output, [-1, train_args.RNN_hidden_size])
                    logits = (tf.matmul(new_output, self.full_connect_weights) + self.full_connect_biases)
                    # 做一次softmax操作
                    #predict_idx = tf.arg_max(logits, dimension=1, output_type=tf.int32)
                    predict_idx = tf.argmax(logits, axis=1, output_type=tf.int32)

                    # 把infer出的下一个idx加入到dec_inp里面去。
                    dec_inp = dec_inp.write(input_index+1, predict_idx[0])

                    return new_state, dec_inp, input_index+1

                # 执行tf.while_loop()，它就会返回最终的结果
                self.final_state_op, self.final_dec_inp_op, self.final_input_index_op = tf.while_loop(continue_loop_condition, loop_body_func, init_variables)
                self.final_dec_inp_op = self.final_dec_inp_op.stack()

    def inference(self):
        with mt_graph.as_default():
            with tf.variable_scope("decoder/rnn/attention_wrapper"):
                # 这里我们使用变长的tf.TensorArray()来放置decoder的输入和输出内容。
                self.dec_inp = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True, clear_after_read=False)
                # 我们先在self.dec_inp里放入[SOS]的id，代表开始标致。
                self.dec_inp = self.dec_inp.write(0, 1)  # 1代表[SOS]的id
                # 我们接下去会使用tf.while_loop()来不断的让decoder输出，因此我们需要提前定义好两个函数。
                # 一个是循环条件，另一个是循环体，还有一个是初始变量。
                # 我们先来定义初始变量，decoder有状态，输入两个变量，我们还要加一个step_count变量。
                # 当step_count超出我们设定的范围的时候，就跳出循环。防止decoder无休止的产生outputs。
                init_dec_state = self.after_attention_cell.zero_state(batch_size=1, dtype=tf.float32)
                input_index_ = 0
                init_variables = (init_dec_state, self.dec_inp, input_index_)

                def continue_loop_condition(state, dec_inp, input_index):
                    end_flag = tf.not_equal(dec_inp.read(input_index), 2)  # 2代表[EOS]的标致
                    length_flag = tf.less_equal(input_index, train_args.test_max_output_sentence_length)
                    continue_flag = tf.logical_and(end_flag, length_flag)
                    continue_flag = tf.reduce_all(continue_flag)
                    return continue_flag

                def loop_body_func(state, dec_inp, input_index):
                    # 读取decoder的输入
                    inp = [dec_inp.read(input_index)]
                    inp_embedding = tf.nn.embedding_lookup(self.trg_embedding, inp)

                    # 调用call函数，向前走一步
                    new_output, new_state = self.after_attention_cell.call(state=state, inputs=inp_embedding)
                    
                    # 将new_output再做一次映射，映射到字典的维度
                    # 先将它reshape一下。
                    new_output = tf.reshape(new_output, [-1, train_args.RNN_hidden_size])
                    logits = (tf.matmul(new_output, self.full_connect_weights) + self.full_connect_biases)
                    # 做一次softmax操作
                    #predict_idx = tf.arg_max(logits, dimension=1, output_type=tf.int32)
                    predict_idx = tf.argmax(logits, axis=1, output_type=tf.int32)

                    # 把infer出的下一个idx加入到dec_inp里面去。
                    dec_inp = dec_inp.write(input_index+1, predict_idx[0])

                    return new_state, dec_inp, input_index+1

                # 执行tf.while_loop()，它就会返回最终的结果
                final_state_op, final_inp_op, final_input_index_op = tf.while_loop(continue_loop_condition, loop_body_func, init_variables)
                
        return final_inp_op.stack()


# 读取英文的token_dictionary
with open(train_args.english_token_id_dictionary_pickle_path, 'rb') as file:
    english_token_id_dictionary = pickle.load(file)
# 读取中文的token_dictionary
with open(train_args.chinese_token_id_dictionary_pickle_path, 'rb') as file:
    chinese_token_id_dictionary = pickle.load(file)
chinese_id_token_dictionary = {idx:token for token, idx in chinese_token_id_dictionary.items()}

nmt_model = Model()  # 创建模型
#inference_op = nmt_model.inference()

# sesstion 的config
session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.allow_growth  = True
with tf.Session(graph=mt_graph, config=session_config) as sess:
    # 构建saver
    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    restore_path = tf.train.latest_checkpoint(train_args.finish_model_saved_path.replace('/model', ''))
    saver.restore(sess, restore_path)
    while True:
        sentence = input("请输入英文句子：")
        # 将英文句子根据token_dictionary转换成idx的形式
        sentence = nltk.word_tokenize(sentence)
        # 将输入的英文句子tokenize成token后转换成id形式。
        for idx, word in enumerate(sentence):
            sentence[idx] = english_token_id_dictionary.get(word, english_token_id_dictionary['<unk>'])
        # 在英文句子的最后添加一个'<eos>'
        sentence.append(english_token_id_dictionary['<eos>'])
        # 句子的长度
        sentence_length = len(sentence)

        translation_result = sess.run(nmt_model.final_dec_inp_op, feed_dict={nmt_model.enc_inp:[sentence], nmt_model.enc_inp_size:[sentence_length]})
        translation_result = list(translation_result)
        for index, idx in enumerate(translation_result):
            translation_result[index] = chinese_id_token_dictionary[idx]
        print(''.join(translation_result[1:]))  # 因为返回的屎decoder的输入部分，因此第1位为<sos>，不需要展现出来。