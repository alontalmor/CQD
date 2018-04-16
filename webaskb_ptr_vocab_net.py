from config import *
from io import open
from operator import itemgetter, attrgetter

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
    def prepareData(self, split_dataset, is_training_set, input_lang=None, output_lang=None):
        if input_lang is None:
            input_lang = Lang('input')
            output_lang = Lang('output')

        pairs_index = {}

        print("Read %s sentence pairs" % len(split_dataset))
        print("Counting words...")

        split_dataset = [split_dataset[i] for i in  np.random.permutation(len(split_dataset))]

        input_lang.addWord('None')
        pairs = []
        for ind,question in enumerate(split_dataset):

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
            if 'output_seq' in question and question['output_seq'] == question['output_seq']:
                y = question['output_seq'] # RL training, always take the actual trajectory as target.
            else:
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

            # building index
            if aux_data['question'] in pairs_index:
                pairs_index[aux_data['question']].append(len(pairs))
            else:
                pairs_index[aux_data['question']] = [len(pairs)]

            pairs.append({'x':x,'y':y,'aux_data':aux_data})

        # shuffling the X,Y pairs
        print ("total number of pair:" + str(len(pairs)))

        return input_lang, output_lang, pairs, pairs_index

    def load_data(self, data_dir, train_file, eval_file):
        if config.LOAD_SAVED_MODEL:
            self.input_lang = config.load_pkl(config.neural_model_dir + config.modeldir,'input_lang')
            self.output_lang = config.load_pkl(config.neural_model_dir + config.modeldir, 'output_lang')
        else:
            self.input_lang = None
            self.output_lang = None

        noisy_sup_train = config.load_json(data_dir, train_file)
        noisy_sup_eval = config.load_json(data_dir, eval_file)

        # we always read the training data - to create the language index in the same order.
        # if model is loaded, input lang will be loaded as well
        self.input_lang, self.output_lang, self.pairs_train, self.pairs_trian_index = \
            self.prepareData(noisy_sup_train,is_training_set=True , input_lang=self.input_lang, \
                                                            output_lang=self.output_lang)
        self.input_lang, self.output_lang, self.pairs_dev, pairs_dev_index =  \
            self.prepareData(noisy_sup_eval, is_training_set=False , input_lang=self.input_lang , \
                                                            output_lang=self.output_lang)

        # saving language model (only input, output should be always the same)
        if config.PERFORM_TRAINING:
            config.store_pkl(self.input_lang, config.neural_model_dir + config.out_subdir, 'input_lang.pkl', )
            config.store_pkl(self.output_lang, config.neural_model_dir + config.out_subdir, 'output_lang.pkl', )

    def preproc_rl_data(self):
        # checking which files exist:
        rl_input_df = pd.DataFrame()
        for dirname, dirnames, filenames in os.walk(config.rl_train_data + config.datadir):
            print('pre-processing the following files: ' +   str(filenames))

            # making sure noisy sup is added first (because of the default MIN_REWARD_TRESH values
            filenames.remove('noisy_sup.json')
            filenames = ['noisy_sup.json'] + filenames

            for filename in filenames:
                if filename.find('.json')>-1:
                    curr_batch = pd.DataFrame(config.load_json(config.rl_train_data + config.datadir,filename))
                    curr_batch = curr_batch[(curr_batch[['split_part1', 'split_part2']].isnull() * 1.0).sum(axis=1) == 0] # removing null values
                    curr_batch['traj_id'] = curr_batch['ID'] + curr_batch['split_part1'].str.replace(" ","") + ',' + curr_batch['split_part2'].str.replace(" ","")
                    if len(rl_input_df)>0:
                        len_before_filter = len(curr_batch)
                        curr_batch = curr_batch[~curr_batch['traj_id'].isin(rl_input_df['traj_id'])]
                        if len(curr_batch)!= len_before_filter:
                            config.store_json(curr_batch.to_dict(orient='rows'), config.rl_train_data + config.datadir, filename)
                    curr_batch['filename'] = filename
                    rl_input_df = rl_input_df.append(curr_batch, ignore_index=True)

        start = datetime.datetime.now()

        # dropping exact duplicate splits
        rl_input_df = rl_input_df.drop_duplicates(['ID', 'comp', 'split_part1', 'split_part2'])

        print('# rewards below tresh: {:}'.format((((rl_input_df['Reward_MRR'] < config.MIN_REWARD_TRESH) & \
                                                    (rl_input_df['Reward_MRR'] > 0)) * 1.0).sum()))

        # all cases of reward under tresh will be zero
        rl_input_df.loc[rl_input_df['Reward_MRR'] < config.MIN_REWARD_TRESH, 'Reward_MRR'] = 0

        # all noisy supervision samples recieve reward of 0.1 if there previous reward is 0
        rl_input_df.loc[((rl_input_df['filename'] == 'noisy_sup.json') & \
                        (rl_input_df['Reward_MRR'] == 0)), 'Reward_MRR']  = config.MIN_REWARD_TRESH

        # filtering zeros
        rl_input_df = rl_input_df[rl_input_df['Reward_MRR'] != 0]

        print('# rewards above tresh: {:}'.format(((rl_input_df['Reward_MRR'] > config.MIN_REWARD_TRESH) * 1.0).sum()))
        print('# rewards equal tresh: {:}'.format(((rl_input_df['Reward_MRR'] == config.MIN_REWARD_TRESH) * 1.0).sum()))

        # def normalize(data):
        # x = data['Reward_MRR'].as_matrix()
        # e_x = np.exp(x)
        # data['Reward_MRR'] = e_x / e_x.sum(axis=0)
        #    data['Reward_MRR'] -= data['Reward_MRR'].mean()
        #    return data
        # rl_input_df = rl_input_df.groupby('ID').apply(normalize)
        print('Processing time: ' + str(datetime.datetime.now() - start))
        print('Total number of stored samples: ' + str(len(rl_input_df)))

        config.store_json(rl_input_df.to_dict(orient='rows'),config.rl_preproc_data + config.out_subdir,  config.eval_set)

    def init(self, criterion = None):
        # define batch training scheme
        model = WebAsKB_PtrVocabNet_Model(self.input_lang, self.output_lang, criterion)

        # train using training scheme
        self.net = NNRun(model, self.pairs_train, self.pairs_dev, self.pairs_trian_index)

    def train(self):
        self.net.run_training()

    def eval(self):
        model_output = self.net.evaluate()
        config.store_json(model_output , config.split_points_dir + config.out_subdir ,config.eval_set )
        config.store_csv(model_output , config.split_points_dir + config.out_subdir ,config.eval_set)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("operation",
                        help='available operations: "run_ptrnet" ,"train_ptrnet"')
    parser.add_argument("--eval_set", help='available eval sets: "dev","test"')
    args = parser.parse_args()

    if args.eval_set is not None:
        config.eval_set = args.eval_set

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


