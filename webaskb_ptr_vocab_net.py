from config import config
from io import open
import numpy as np
import json

import torch
from torch.autograd import Variable

from Net.run import NNRun
from common.embeddings import Lang
from config import Config
from common.embeddings import embeddings
from Models.webaskb_ptr_vocab_net import WebAsKB_PtrVocabNet_Model

class WebAsKB_PtrVocabNet():
    def __init__(self):
        # Embeddings
        self.embed = embeddings()
        self.embed.load_vocabulary_word_vectors(config.glove_50d,'glove.6B.50d.txt',50)

    # Load Data
    def prepareData(self, filename, is_training_set, input_lang=None, output_lang=None):
        if input_lang is None:
            input_lang = Lang('input')
            output_lang = Lang('output')

        with open(filename, 'r') as outfile:
            split_dataset = json.load(outfile)

        print("Read %s sentence pairs" % len(split_dataset))
        print("Counting words...")

        input_lang.addWord('None')
        pairs = []
        for question in split_dataset:
            # training is done using only composition and conjunction examples
            if is_training_set and question['comp'] != 'composition' and question['comp'] != 'conjunction':
                continue

            x = []
            y = []
            aux_data = question

            if len(question['sorted_annotations'])>config.MAX_LENGTH-4:
                continue

            for token in question['sorted_annotations']:
                x.append([token['dependentGloss'],token['dep']])
                input_lang.addWord(token['dependentGloss'])
                input_lang.addWord(token['dep'])
            # returns embeded data (also converts to Variables tokens that were not found in Glove)
            x = self.embed.sentence_to_embeddings(input_lang, x)

            # adding actions to ouptuts
            y=[]
            # if supervision is nan don't add a y target variable ( happens in comperative and superlative or dev set)
            if question['pointer_ind'] == question['pointer_ind']:
                for pointer_ind, seq2seq_output in zip(question['pointer_ind'], question['seq2seq_output']):
                    output_lang.addWord(seq2seq_output)
                    # if 'copy' than point to the input index, else point to the embeded output vocbulary
                    if pointer_ind is not None:
                        y.append(pointer_ind)
                    else:
                        y.append(config.MAX_LENGTH + output_lang.word2index[seq2seq_output])


            if config.use_cuda:
                y = Variable(torch.LongTensor(y).view(-1, 1)).cuda()
            else:
                y = Variable(torch.LongTensor(y).view(-1, 1))

            pairs.append({'x':x,'y':y,'aux_data':aux_data})

        # shuffling the X,Y pairs
        print ("total number of pair:" + str(len(pairs)))
        np.random.seed(5)
        pairs = [pairs[i] for i in np.random.permutation(len(pairs))]

        return input_lang, output_lang, pairs

    def load_data(self):
        # we always read the training data - to create the language index in the same order.
        self.input_lang, self.output_lang, self.pairs_train = \
            self.prepareData(config.noisy_supervision_dir + 'train.json',is_training_set=True)
        self.input_lang, self.output_lang, self.pairs_dev = \
            self.prepareData(config.noisy_supervision_dir + config.EVALUATION_SET + '.json', \
                                                           is_training_set=False , input_lang=self.input_lang , \
                                                            output_lang=self.output_lang)

    def init(self):
        # define batch training scheme
        model = WebAsKB_PtrVocabNet_Model(self.input_lang)

        # train using training scheme
        self.net = NNRun(model, self.pairs_train, self.pairs_dev)

    def train(self):
        self.net.run_training()

    def eval(self):
        model_output = self.net.evaluate()
        with open(config.split_points_dir + config.EVALUATION_SET + '.json', 'w') as outfile:
            outfile.write(json.dumps(model_output))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("operation",
                        help='available operations: "run_ptrnet" ,"train_ptrnet"')
    parser.add_argument("--eval_set", help='available eval sets: "dev","test"')
    args = parser.parse_args()

    if args.eval_set is not None:
        config.EVALUATION_SET = args.eval_set

    if args.operation == 'run_ptrnet':
        ptrnet = WebAsKB_PtrNet()
        ptrnet.load_data()
        ptrnet.init()
        ptrnet.eval()
    elif args.operation == 'train_ptrnet':
        config.PERFORM_TRAINING = True
        config.LOAD_SAVED_MODEL = False
        ptrnet = WebAsKB_PtrNet()
        ptrnet.load_data()
        ptrnet.init()
        ptrnet.train()


