#coding=utf-8
'''
Author:Haitaifantuan
'''

import tensorflow as tf
import data_preprocessing
from sklearn.metrics import accuracy_score


# 以下参数需要配置=======================================
window_size = 10
word_dict_size = 30000
sentence_and_word_embedding_size = 400
batch_size = 5000
classfication_batch_size = 100
# 以上参数需要配置=======================================


class Model(object):
    def __init__(self, embedding_train_flag):
        self.embedding_train_flag = embedding_train_flag
        self.graph_name = self.create_model()
        self.embedding_train_total_epoch = 500
        self.classifier_train_total_epoch = 50
        self.embedding_test_total_epoch = 50
    
    def create_model(self):
        doc2vec_graph = tf.Graph()
        with doc2vec_graph.as_default(), tf.device('/gpu:0'):
            with tf.variable_scope("placeholder"):
                # 这一个scope存放的是placehoder
                self.words_input = tf.placeholder(dtype=tf.int32, shape=[None, window_size-1])
                self.sentence_input = tf.placeholder(dtype=tf.int32, shape=[None, 1])
                self.embedding_label = tf.placeholder(dtype=tf.int32, shape=[None, 1])
                
                self.sentiment_input = tf.placeholder(dtype=tf.int32, shape=[None, 1])
                self.sentiment_label = tf.placeholder(dtype=tf.int32, shape=[None, 1])
                
            with tf.variable_scope("sentence_embedding_train_parameters_scope"):
                # 这一个scope放的是训练训练集的句向量时，需要被训练的参数
                self.word_embedding_matrix = tf.Variable(initial_value=tf.truncated_normal(shape=[word_dict_size, sentence_and_word_embedding_size], dtype=tf.float32), trainable=True, name="word_embedding_matrix")
                self.train_sentence_embedding_matrix = tf.Variable(initial_value=tf.truncated_normal(shape=[25000, sentence_and_word_embedding_size], dtype=tf.float32), trainable=True)
                
            with tf.variable_scope("sentence_embedding_test_parameters_scope"):
                # 这一个scope放的是，训练测试集的句向量时，需要被训练的参数，和sentence_embedding_train_parameters_scope不同的是
                # 训练测试集的句向量时，不需要训练self.word_embedding_matrix，以及模型的参数，只需要训练self.test_sentence_embedding_matrix
                self.test_sentence_embedding_matrix = tf.Variable(initial_value=tf.truncated_normal(shape=[25000, sentence_and_word_embedding_size], dtype=tf.float32), trainable=True)
                
            with tf.variable_scope("sentence_embedding_unsup_parameters_scope"):
                self.unsup_sentence_embedding_matrix = tf.Variable(initial_value=tf.truncated_normal(shape=[50000, sentence_and_word_embedding_size], dtype=tf.float32), trainable=True)
                
            with tf.variable_scope("concat_layer"):
                if self.embedding_train_flag == True:
                    self.sentence_embedding_input = tf.nn.embedding_lookup(self.train_sentence_embedding_matrix, self.sentence_input)
                else:
                    self.sentence_embedding_input = tf.nn.embedding_lookup(self.test_sentence_embedding_matrix, self.sentence_input)
                
                self.word_embedding_input = tf.nn.embedding_lookup(self.word_embedding_matrix, self.words_input)
                self.concat_result = tf.concat([self.sentence_embedding_input, self.word_embedding_input], axis=1)
                self.concat_result = tf.layers.flatten(self.concat_result)  # 将拼接后的结果，弄平。这个时候shape应该是batch_size*400*10
            
            with tf.variable_scope("sentence_embedding_train_parameters_scope"):
                # 注意！这里我们又回到了sentence_embedding_train_parameters_scope，因为这个模型参数也需要在训练训练集的句向量时候，被更新
                print(tf.get_default_graph())
                # 这里不直接使用softmax是因为计算loss的时候，会自动进行softmax操作
                self.embedding_logits = tf.layers.dense(inputs=self.concat_result, units=word_dict_size, trainable=True, name="hidden_to_output")  # 因为词表大小是30000
            
            with tf.variable_scope("sentiment_classification_train_scope"):
                # 对于情感分类，我们直接使用全连接方式，这里要注意，我们是直接从训练好的句向量矩阵拿句子的向量的
                if self.embedding_train_flag == True:
                    self.classification_input = tf.nn.embedding_lookup(self.train_sentence_embedding_matrix, self.sentiment_input)
                else:
                    self.classification_input = tf.nn.embedding_lookup(self.test_sentence_embedding_matrix, self.sentiment_input)
                # 进行全连接线性映射。这里不直接使用softmax是因为计算loss的时候，会自动进行softmax操作
                self.classification_input = tf.reshape(self.classification_input, [-1, sentence_and_word_embedding_size])
                self.classification_logits = tf.layers.dense(inputs=self.classification_input, units=50, trainable=True, name="cls_hidden_to_output_0")
                self.classification_logits = tf.layers.dense(inputs=self.classification_logits, units=2, trainable=True, name="cls_hidden_to_output_1")
                
            # 下面开始计算loss。tf.nn.softmax_cross_entropy_with_logits_v2内置直接计算好softmax。
            self.sentence_embedding_label_one_hot_format = tf.one_hot(indices=self.embedding_label, depth=word_dict_size)  # 因为词表大小是30000
            self.sentence_embedding_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.sentence_embedding_label_one_hot_format, self.embedding_logits)
            
            
            self.sentiment_label_one_hot_format = tf.one_hot(self.sentiment_label, 2)
            self.sentiment_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.sentiment_label_one_hot_format, self.classification_logits)
            
            # 下面定义train操作
            self.optimizer = tf.train.AdamOptimizer()
            # 找出不同阶段需要训练不同的变量。找出训练训练集的句向量时，需要被训练的参数
            self.sentence_embedding_train_stage_need_to_update_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="sentence_embedding_train_parameters_scope")
            # 找出训练测试集的句向量时，需要被训练的参数
            self.sentence_embedding_test_stage_need_to_update_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="sentence_embedding_test_parameters_scope")
            # 找出训练情感分类时，需要被训练的参数
            self.sentiment_stage_need_to_update_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="sentiment_classification_train_scope")
            
            # 定义反向传播操作
            if self.embedding_train_flag == True:
                self.embedding_train_op = self.optimizer.minimize(self.sentence_embedding_loss, var_list=self.sentence_embedding_train_stage_need_to_update_variables)
            else:
                self.embedding_test_op = self.optimizer.minimize(self.sentence_embedding_loss, var_list=self.sentence_embedding_test_stage_need_to_update_variables)
            self.sentiment_train_op = self.optimizer.minimize(self.sentiment_loss, var_list=self.sentiment_stage_need_to_update_variables)
            
            # 定义情感分类的预测结果
            self.sentiment_predict = tf.argmax(self.classification_logits, axis=1)
            
            # 其他无关紧要的操作
            # 计算平均loss
            self.mean_loss = tf.reduce_mean(self.sentence_embedding_loss, axis=0)
        
        return doc2vec_graph
        
    
    def embedding_train_classifier_train(self, data_obj, sess):
        sess.run(tf.global_variables_initializer())
        total_steps = len(data_obj.train_embedding_word_input_data) // batch_size
        for epoch in range(self.embedding_train_total_epoch):
            for each_step in range(total_steps):
                train_embedding_word_input_data = data_obj.train_embedding_word_input_data[each_step*batch_size:(each_step+1)*batch_size][:]
                train_embedding_sentence_input = data_obj.train_embedding_sentence_input[each_step*batch_size:(each_step+1)*batch_size][:]
                train_embedding_labels = data_obj.train_embedding_labels[each_step*batch_size:(each_step+1)*batch_size][:]
                
                feed_dict = {self.words_input:train_embedding_word_input_data,
                             self.sentence_input:train_embedding_sentence_input,
                             self.embedding_label:train_embedding_labels}
                
                _, mean_loss = sess.run([self.embedding_train_op, self.mean_loss], feed_dict=feed_dict)
                print('当前为第{}个epoch-----当前为第{}步-----当前Loss为：{}-----已训练{}个样本'.format(epoch+1, each_step, mean_loss, each_step*batch_size))
                if (each_step%50)==0:
                    with self.graph_name.as_default():
                        need_to_save_var_list = [tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/word_embedding_matrix:0"),
                                                 tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/hidden_to_output/kernel:0"),
                                                 tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/hidden_to_output/bias:0"),
                                                 tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_0/kernel:0"),
                                                 tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_0/bias:0"),
                                                 tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_1/kernel:0"),
                                                 tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_1/bias:0")]
                        saver = tf.train.Saver(var_list=need_to_save_var_list)
                        saver.save(sess, './saved_things/model/model-{}-epoch-未训练情感分类模型.ckpt'.format(each_step))                    
        # epoch训练完毕后，开始训练情感分类相关的参数
        total_steps = len(data_obj.train_sentiment_input) // classfication_batch_size
        for epoch in range(self.classifier_train_total_epoch):
            for each_step in range(total_steps):
                train_sentiment_input = data_obj.train_sentiment_input[each_step*classfication_batch_size:(each_step+1)*classfication_batch_size][:]
                train_sentiment_labels = data_obj.train_sentiment_labels[each_step*classfication_batch_size:(each_step+1)*classfication_batch_size][:]
                feed_dict = {self.sentiment_input:train_sentiment_input,
                             self.sentiment_label:train_sentiment_labels}
                
                _, sentiment_predict = sess.run([self.sentiment_train_op, self.sentiment_predict], feed_dict=feed_dict)
                acc = accuracy_score(y_true=train_sentiment_labels,y_pred=sentiment_predict)
                print('当前为第{}个epoch-----当前样本数为：{}-----准确率为：{}'.format(epoch+1, batch_size, acc))
            
        #所需要保存的参数
        with self.graph_name.as_default():
            need_to_save_var_list = [tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/word_embedding_matrix:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/hidden_to_output/kernel:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/hidden_to_output/bias:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_0/kernel:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_0/bias:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_1/kernel:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_1/bias:0")]
            saver = tf.train.Saver(var_list=need_to_save_var_list)
            saver.save(sess, './saved_things/model/model.ckpt')
            
    
    def embedding_test_classifier_predict(self, data_obj, sess):
        sess.run(tf.global_variables_initializer())
        with self.graph_name.as_default():
            need_to_restore_var_list = [tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/word_embedding_matrix:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/hidden_to_output/kernel:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentence_embedding_train_parameters_scope/hidden_to_output/bias:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_0/kernel:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_0/bias:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_1/kernel:0"),
                                     tf.get_default_graph().get_tensor_by_name("sentiment_classification_train_scope/cls_hidden_to_output_1/bias:0")]            
            saver = tf.train.Saver(var_list=need_to_restore_var_list)
            saver.restore(sess, './saved_things/model/model.ckpt')
        # 模型的部分参数恢复成功后，我们开始训练测试集的句向量。训练一定数量的epoch后，然后调用情感分类器进行预测
        total_steps = len(data_obj.test_embedding_sentence_input) // batch_size
        for epoch in range(self.embedding_test_total_epoch):
            for each_step in range(total_steps):
                test_embedding_word_input_data = data_obj.test_embedding_word_input_data[each_step*batch_size:(each_step+1)*batch_size][:]
                test_embedding_sentence_input = data_obj.test_embedding_sentence_input[each_step*batch_size:(each_step+1)*batch_size][:]
                test_embedding_labels = data_obj.test_embedding_labels[each_step*batch_size:(each_step+1)*batch_size][:]
    
                feed_dict = {self.words_input:test_embedding_word_input_data,
                                 self.sentence_input:test_embedding_sentence_input,
                                 self.embedding_label:test_embedding_labels}
    
                _, mean_loss = sess.run([self.embedding_test_op, self.mean_loss], feed_dict=feed_dict)
                print('当前为第{}个epoch-----当前为第{}步-----当前Loss为：{}-----已训练{}个样本'.format(epoch+1, each_step, mean_loss, each_step*batch_size))
                #if each_step == 1:
                #    break                
                
        # 测试集的句向量训练完毕后，调用情感分类模型进行预测
        total_steps = len(data_obj.test_sentiment_input) // classfication_batch_size
        total_sample = 0
        correct_sample = 0
        for each_step in range(total_steps):
            test_sentiment_input = data_obj.test_sentiment_input[each_step*classfication_batch_size:(each_step+1)*classfication_batch_size][:]
            test_sentiment_labels = data_obj.test_sentiment_labels[each_step*classfication_batch_size:(each_step+1)*classfication_batch_size][:]
            feed_dict = {self.sentiment_input:test_sentiment_input,
                             self.sentiment_label:test_sentiment_labels}

            sentiment_predict = sess.run([self.sentiment_predict], feed_dict=feed_dict)
            acc = accuracy_score(y_true=test_sentiment_labels,y_pred=sentiment_predict[0])
            total_sample += len(data_obj.test_sentiment_labels[each_step*classfication_batch_size:(each_step+1)*classfication_batch_size][:])
            correct_sample += acc*total_sample
            print('当前为第{}个epoch-----当前样本数为：{}-----准确率为：{}'.format(total_epoch+1, batch_size, acc))
        
        print('预测完毕-----总样本数为：{}-----正确率为：{}'.format(len(data_obj.test_sentiment_input), correct_sample/total_sample))
                    

    

data = data_preprocessing.Data_preprocessing()
session_config = tf.ConfigProto(
                    log_device_placement=False,
                    inter_op_parallelism_threads=0,
                    intra_op_parallelism_threads=0,
                    allow_soft_placement=True)
session_config.gpu_options.allow_growth = True # 动态申请显存

# 训练训练集的句向量阶段
model = Model(embedding_train_flag=True)
with tf.Session(graph=model.graph_name, config=session_config) as sess:
    model.embedding_train_classifier_train(data, sess)

# 训练测试集的句向量阶段，训练好测试集的句向量后，直接放入到情感分类器里面进行情感分类。
tf.reset_default_graph() #清空计算图并重新创建。
model = Model(embedding_train_flag=False)
with tf.Session(graph=model.graph_name, config=session_config) as sess:
    model.embedding_test_classifier_predict(data, sess)

    
