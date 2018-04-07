from config import *
from SplitQA import SplitQA
from noisy_supervision import NoisySupervision
#from webaskb_ptrnet import WebAsKB_PtrNet
from webaskb_ptr_vocab_net import WebAsKB_PtrVocabNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("operation", help='available operations: "gen_noisy_sup","run_ptrnet" ,"train_ptrnet", "splitqa"')
parser.add_argument("--eval_set", help='available eval sets: "dev","test"')
parser.add_argument("--run_tag", help='')
parser.add_argument("--input_dir", help='define which directory to load input from (model dir for instance)')

args = parser.parse_args()

# run dir will be created depending on the operation
config.create_run_dir(args.run_tag,args.operation)

if args.eval_set is not None:
    config.EVALUATION_SET = args.eval_set

if args.operation == 'gen_noisy_sup':
    noisy_sup = NoisySupervision()
    noisy_sup.gen_noisy_supervision()
elif args.operation == 'run_model':
    ptrnet = WebAsKB_PtrVocabNet()
    ptrnet.load_data()
    ptrnet.init()
    ptrnet.eval()
elif args.operation == 'train_model':
    config.PERFORM_TRAINING = True
    config.LOAD_SAVED_MODEL = False
    config.max_evalset_size = 2000
    ptrnet = WebAsKB_PtrVocabNet()
    ptrnet.load_data()
    ptrnet.init()
    ptrnet.train()
elif args.operation == 'splitqa':
    config.PERFORM_TRAINING = False
    splitqa = SplitQA()
    splitqa.run_executors()
    splitqa.compute_final_results()