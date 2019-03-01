# -*- coding: utf-8 -*-
from __future__ import print_function, division

configure = {
    # if doing training, set 'do_train' to True and 'do_predict' to False
    'do_train': False,
    'training_data_path': '../dataset/trainIn_new.txt',
    'validate_data_path': '../dataset/devIn_new.txt',
    'batch_size': 128,
    'num_spoch': 20,
    'optimizer': 'sgd',
    'lrate': 0.001,
    'save_path': '../models/' + 'newmodel',
    'log_dir': '../log/' + 'newmodel/',
    "label2id_file":"../dataset/labelidx",

    # if doing predicting, set 'do_train' to False and 'do_predict' to True
    'do_predict': True,
    'testing_data_path': '../dataset/devIn_new.txt',
    'testing_sourcefile_path': '../dataset/cpbdev.txt',

    # shared settings in training and predicting
    'vocab_size': 17073, # 0 for <eos>, 1 for <bos> and 2 for <unk>
    'use_crf': True,
    'n_label': 67,
    'embedding_dim': 50,
    'postag_dim': 20,
    'distance_dim': 20,
    'RNN_dim': 100,
    'num_of_rnn_layers': 1,
    'max_len': 175,
    "train_steps":5000,
    "log_dir":"logdir",
}
