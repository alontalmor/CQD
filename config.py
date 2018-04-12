
import time
import os
import pandas as pd
import numpy as np
import json
import unicodedata
import random
import traceback
from common.elastic_logger import ElasticLogger
import dropbox
import torch
import pickle
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as weight_init
import socket


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Config:
    def __init__(self):
        # add per computer profiles here
        print('computer name:' + socket.gethostname())
        if socket.gethostname() == 'Alons-MacBook-Pro-2.local':
            self.base_dir = '/Users/alontalmor/Dropbox/Apps/WebKB/webkb_dev_data/'
            #self.base_dir = 'webkb_dev_data/'
            self.USE_CLOUD_STORAGE = False
        else:
            self.base_dir = 'webkb_dev_data/'
            self.USE_CLOUD_STORAGE = True

        self.init()

    def init(self):
        # default are "test"
        self.name = 'test'
        self.out_subdir = 'test/'
        self.input_data = ''
        self.input_model = ''

        # Neural param
        self.LR = 0.007
        self.ADA_GRAD_LR_DECAY = 1e-6
        self.ADA_GRAD_L2 = 3e-4
        self.dropout_p = 0.25
        self.hidden_size = 512
        self.MAX_LENGTH = 28
        self.EMBEDDING_VEC_SIZE = 50

        self.print_every = 1000
        self.evaluate_every = 3000
        self.evalset_offset = 0
        self.max_evalset_size = 2000 # used to limit size of dev set when training

        self.MINI_BATCH_SIZE = 10
        self.output_size = 29

        self.use_teacher_forcing = True
        self.teacher_forcing_full_until = 10000
        self.teacher_forcing_partial_until = 30000

        # used to limit size of dev set when training
        self.use_output_masking = True

        # manual seeding for run comparison
        torch.manual_seed(0)
        random.seed(0)

        # used for generated the actual output in run_ptrnet
        self.gen_model_output = True


        # Number of training iteration with no substantial dev accuracy improvement to stop training ("early stopping")
        self.NO_IMPROVEMENT_ITERS_TO_STOP = 50000
        self.MAX_ITER = 200000

        self.SOS_token = 0
        self.EOS_token = 1

        # Neural Options
        self.use_cuda = False
        self.USE_GLOVE = True
        self.WRITE_TO_TENSORBOARD = False
        self.LOAD_SAVED_MODEL = True
        self.PERFORM_TRAINING = False
        self.SAVE_DISTRIBUTIONS = False

        # choose dev or test
        self.EVALUATION_SET = 'dev'
        self.always_save_model = False

        # Stanford NLP
        os.environ["CLASSPATH"] =  "Lib/stanford-ner-2016-10-31"
        os.environ["STANFORD_MODELS"] = "Lib/stanford-ner-2016-10-31/classifiers"
        self.StanfordCoreNLP_Path = 'http://127.0.0.1:9000'

        # Paths
        self.complexwebquestions_dir = self.base_dir + "complex_web_questions/"
        self.noisy_supervision_dir = self.base_dir + 'noisy_supervision/'
        self.neural_model_dir = self.base_dir + "ptrnet_model/"
        self.split_points_dir = self.base_dir + "split_points/"
        self.rc_answer_cache_dir = self.base_dir + "rc_answer_cache/"
        self.rl_train_data = self.base_dir + 'RL_train_data/'
        self.rl_dev_data = self.base_dir + 'RL_dev_data/'
        self.rl_preproc_data = self.base_dir + 'RL_preproc_data/'

        self.logger = None
        if self.USE_CLOUD_STORAGE:
            print('Using Cloud Storage')
            self.dbx = dropbox.Dropbox('7j6m2s1jYC0AAAAAAAHy69fu0OxDAU3fPbIjjarqr_1zalj8Mvypf8U71BoLT-AD')
        else:
            print('Using Local Storage')
            self.dbx = None

        self.run_start_time = time.strftime("%m-%d_%H-%M-%S")

        # Data
        self.glove_50d = self.base_dir + "embeddings/glove/glove.6B.50d.txt.zip"
        self.glove_300d_sample = self.base_dir + "embeddings/glove/glove.sample.300d.txt.zip"

        ####  Reinforcement Learning ###
        self.RL_Training = False

        self.MIN_REWARD_TRESH = 0.1

        #  data generation
        self.gen_trajectories = False
        self.skip_limit = 0
        self.generate_all_skips = False
        if self.gen_trajectories:
            print(' -------- Generating trajecotires! ------------')

    def create_run_dir(self,run_tag, operation):
        self.run_tag = run_tag

        # training requires a new output dir for each training session
        if operation == 'train_model':
            self.neural_model_dir += run_tag + '/'
                                     #+ "_" + str(self.LR) + '_' + str(self.ADA_GRAD_L2) + \
                                     #'_' + str(self.hidden_size) + \
                                     #'_' + str(self.dropout_p) + \
                                     #'_' + self.run_start_time + '/'

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

            self.logger.set_repeated_context_dict(self.name , repeated_context_dict)

        self.logger.write_log(level, message, context_dict)

    def store_on_cloud(self, cloud_path, input_str=None, from_file=False, local_path=None):
        try:
            # data may be a file path (which torch.save is used)
            if from_file:
                with open(local_path, "rb") as f:
                    self.dbx.files_upload(f.read(), '/' + cloud_path, mode = dropbox.files.WriteMode.overwrite)
            else:
                self.dbx.files_upload(input_str, '/' + cloud_path, mode=dropbox.files.WriteMode.overwrite)
        except:
            config.write_log('ERROR', "save to dropbox failed", {'error_message': traceback.format_exc()})

    def load_from_cloud(self, cloud_path, to_file=False , local_path=None):
        try:
            if to_file:
                self.dbx.files_download_to_file(local_path, '/' + cloud_path)
                #with open(local_path, 'wb') as f:
                #    f.write(res.content)
            else:
                md, res = self.dbx.files_download('/' + cloud_path)
                return res.content
        except:
            config.write_log('ERROR', "load from dropbox failed", {'error_message': traceback.format_exc()})

    def store_pytorch_model(self, model, dirname, filename):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        torch.save(model, dirname + filename)
        if config.USE_CLOUD_STORAGE:
            config.store_on_cloud(dirname + filename + '.pkl', from_file=True, local_path= dirname + filename + '.pkl')

    def load_pytorch_model(self, dirname, filename):
        start_time = datetime.datetime.now()
        if config.USE_CLOUD_STORAGE:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            config.load_from_cloud(dirname + filename + '.pkl', local_path= dirname + filename +'.pkl', to_file=True)

        model =  torch.load(dirname + filename + '.pkl')
        config.write_log('INFO', "load_pytorch_model one", {'time it took': str(datetime.datetime.now() - start_time), 'path':dirname + filename})
        return model

    def store_csv(self, data, dirname, filename):
        start_time = datetime.datetime.now()
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        pd.DataFrame(data).to_csv(dirname + filename + '.csv', encoding="utf-8", index=False)
        if config.USE_CLOUD_STORAGE:
            config.store_on_cloud(dirname + filename + '.csv', from_file=True, local_path= dirname + filename + '.csv')
        config.write_log('INFO', "store_csv",{'time it took': str(datetime.datetime.now() - start_time), 'path': dirname + filename})

    def load_csv(self):
        # TODO
        pass

    def store_pkl(self, data, dirname, filename):
        start_time = datetime.datetime.now()
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        with open(dirname + filename + '.pkl', 'wb') as outfile:
            pickle.dump(data, outfile)
        if config.USE_CLOUD_STORAGE:
            config.store_on_cloud(dirname + filename + '.pkl', from_file=True, local_path= dirname + filename + '.pkl')
        config.write_log('INFO', "store_pkl",
                     {'time it took': str(datetime.datetime.now() - start_time), 'path': dirname + filename})

    def load_pkl(self, dirname, filename):
        start_time = datetime.datetime.now()
        if config.USE_CLOUD_STORAGE:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            config.load_from_cloud(dirname + filename + '.pkl', to_file=True, local_path=dirname + filename + '.pkl')
        with open(dirname + filename + '.pkl', 'rb') as outfile:
            data = pickle.load(outfile)
        config.write_log('INFO', "load_pkl",
                         {'time it took': str(datetime.datetime.now() - start_time), 'path': dirname + filename})
        return data

    def store_json(self, data, dirname, filename, pretty=False):
        start_time = datetime.datetime.now()
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(dirname + filename + '.json', 'w') as outfile:
            if pretty:
                outfile.write(json.dumps(data, sort_keys=True, indent=4))
            else:
                outfile.write(json.dumps(data))

        if config.USE_CLOUD_STORAGE:
            config.store_on_cloud(dirname + filename + '.json', from_file=True,
                                  local_path=dirname + filename + '.json')
        config.write_log('INFO', "store_json",
                         {'time it took': str(datetime.datetime.now() - start_time), 'path': dirname + filename})

    def load_json(self,dirname ,filename):
        start_time = datetime.datetime.now()
        if config.USE_CLOUD_STORAGE:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            config.load_from_cloud(dirname + filename + '.json', to_file=True, local_path=dirname + filename + '.json')
        with open(dirname + filename + '.json', 'rb') as outfile:
            data = json.load(outfile)
        config.write_log('INFO', "load_json",
                         {'time it took': str(datetime.datetime.now() - start_time), 'path': dirname + filename})
        return data

config = Config()

