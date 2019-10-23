#coding=utf-8
'''
Author:Haitaifantuan
'''

raw_train_english_data_path = 'data/raw_data/train.raw.en'
raw_train_chinese_data_path = 'data/raw_data/train.raw.zh'
raw_train_english_after_tokenization_data_path = 'data/raw_data_after_tokenization/train.raw.en.after_tokenization.txt'
raw_train_chinese_after_tokenization_data_path = 'data/raw_data_after_tokenization/train.raw.zh.after_tokenization.txt'
english_token_id_dictionary_pickle_path = 'data/token_dictionary/english_token_id_dictionary.pickle'
chinese_token_id_dictionary_pickle_path = 'data/token_dictionary/chinese_token_id_dictionary.pickle'
train_en_converted_to_id_path = 'data/train_data_converted_to_id/train.en.converted_to_id.txt'
train_zh_converted_to_id_path = 'data/train_data_converted_to_id/train.zh.converted_to_id.txt'


doesnt_finish_model_saved_path = 'saved_things/doesnt_finish_training_model/model'
doesnt_finish_model_saved_path_cheackpoint = doesnt_finish_model_saved_path.replace('/model', '/checkpoint')
finish_model_saved_path = 'saved_things/finish_training_model/model'

train_max_sent_len = 50
RNN_hidden_size = 1024
num_decoder_layers = 2
Source_vocab_size = 10000
Target_vocab_size = 4000
Share_softmax_embedding = True
train_batch_size = 200
learning_rate = 0.1
max_global_epochs = 50
num_epoch_per_save = 1

test_max_output_sentence_length = 50