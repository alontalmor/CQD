
import time
import os
import pandas as pd
import numpy as np
import json
import unicodedata
import traceback
from common.elastic_logger import ElasticLogger
import dropbox
import torch

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Config:
    def __init__(self):
        # Neural param
        self.LR = 0.007
        self.ADA_GRAD_LR_DECAY = 1e-6
        self.ADA_GRAD_L2 = 3e-4
        self.dropout_p = 0.25
        self.hidden_size = 512
        self.MAX_LENGTH = 28
        self.EMBEDDING_VEC_SIZE = 50

        self.NUM_OF_ITER = 1000000
        self.NUM_OF_SAMPLES = None
        self.print_every = 500
        self.evaluate_every = 3000
        self.MINI_BATCH_SIZE = 10
        self.output_size = 29

        self.use_teacher_forcing = True
        self.teacher_forcing_full_until = 10000
        self.teacher_forcing_partial_until = 30000

        # used to limit size of dev set when training
        self.use_output_masking = True

        # manual seeding for run comparison
        torch.manual_seed(0)

        # used for generated the actual output in run_ptrnet
        self.gen_model_output = True

        # used to limit size of dev set when training
        self.max_evalset_size = 4000

        # Number of training iteration with no substantial dev accuracy improvement to stop training ("early stopping")
        self.NO_IMPROVEMENT_ITERS_TO_STOP = 50000



        self.SOS_token = 0
        self.EOS_token = 1

        # Neural Options
        self.use_cuda = False
        self.USE_GLOVE = True
        self.WRITE_TO_TENSORBOARD = False
        self.LOAD_SAVED_MODEL = True
        self.PERFORM_TRAINING = False
        self.GEN_RL_SAMPLES = False
        self.PRINT_PROB = False

        # choose dev or test
        self.EVALUATION_SET = 'dev'

        # Stanford NLP
        os.environ["CLASSPATH"] =  "Lib/stanford-ner-2016-10-31"
        os.environ["STANFORD_MODELS"] = "Lib/stanford-ner-2016-10-31/classifiers"
        self.StanfordCoreNLP_Path = 'http://127.0.0.1:9000'

        # Paths
        self.data_dir = 'Data/'
        self.complexwebquestions_dir = self.data_dir + "complex_web_questions/"
        self.noisy_supervision_dir = self.data_dir + 'noisy_supervision/'
        self.neural_model_dir = self.data_dir + "ptrnet_model/"
        self.split_points_dir = self.data_dir + "split_points/"
        self.rc_answer_cache_dir = self.data_dir + "rc_answer_cache/"

        self.logger = None
        self.dbx = None

        self.run_start_time = time.strftime("%m-%d_%H-%M-%S")

        # Data
        self.glove_50d = self.data_dir + "embeddings/glove/glove.6B.50d.txt.zip"
        self.glove_300d_sample = self.data_dir + "embeddings/glove/glove.sample.300d.txt.zip"

    def create_run_dir(self,run_tag, operation):
        self.run_tag = run_tag

        # training requires a new output dir for each training session
        if operation == 'train_model':
            self.neural_model_dir += run_tag + "_" + str(self.LR) + '_' + str(self.ADA_GRAD_L2) + \
                                     '_' + str(self.hidden_size) + \
                                     '_' + str(self.dropout_p) + \
                                     '_' + self.run_start_time + '/'

            if not os.path.isdir(self.neural_model_dir):
                os.mkdir(self.neural_model_dir)

    def write_log(self, level, message, context_dict={}):
        if self.logger is None:
            self.logger = ElasticLogger()

            repeated_context_dict = {'LR': self.LR,
                'ADA_GRAD_LR_DECAY': self.ADA_GRAD_LR_DECAY,
                'ADA_GRAD_L2' : self.ADA_GRAD_L2,
                'dropout_p' : self.dropout_p,
                'hidden_size' : self.hidden_size,
                'MAX_LENGTH' : self.MAX_LENGTH,
                'EMBEDDING_VEC_SIZE' : self.EMBEDDING_VEC_SIZE,
                'EVALUATION_SET':self.EVALUATION_SET}

            self.logger.set_repeated_context_dict(self.run_tag , repeated_context_dict)

        self.logger.write_log(level, message, context_dict)

    def store_file(self,file,tar_path):
        if self.dbx is None:
            self.dbx = dropbox.Dropbox('7j6m2s1jYC0AAAAAAAHy69fu0OxDAU3fPbIjjarqr_1zalj8Mvypf8U71BoLT-AD')
        if type(file) == str:
            with open(file, "rb") as f:
                self.dbx.files_upload(f.read(), '/' + tar_path, mode = dropbox.files.WriteMode.overwrite)
        else:
            pass



config = Config()

