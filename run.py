from config import *
#from SplitQA import SplitQA
from noisy_supervision import NoisySupervision
#from webaskb_ptrnet import WebAsKB_PtrNet
from webaskb_ptr_vocab_net import WebAsKB_PtrVocabNet

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("operation", help='available operations: "gen_noisy_sup","run_ptrnet" ,"train_ptrnet", "splitqa"')

# adding all config attributes
for member in inspect.getmembers(config):
    if not inspect.ismethod(member[1]) and not member[0].startswith(("__")):
        if type(member[1]) == bool:
            # bool options if added will automatically considered true if not state otherwise
            parser.add_argument('--' + member[0], type=str2bool, default=member[1], nargs='?', const=True , help=str(member[1]))
        else:
            parser.add_argument('--' + member[0], type=type(member[1]), default=member[1], help=str(member[1]))

args = parser.parse_args()

for arg in inspect.getmembers(args):
    if not inspect.ismethod(arg[1]) and not arg[0].startswith(("__")):
        if getattr(config, arg[0], None) !=  arg[1]:
            print(arg[0] + " = " + str(arg[1]))
            setattr(config, arg[0], arg[1])

config.init()

if args.operation == 'gen_noisy_sup':
    noisy_sup = NoisySupervision()
    noisy_sup.gen_noisy_supervision()
elif args.operation == 'run_model':
    if config.parent_data_dir == '':
        config.parent_data_dir = config.noisy_supervision_dir
    else:
        config.parent_data_dir = config.base_dir + config.parent_data_dir + '/'

    if config.eval_set == '*':
        for dirname, dirnames, filenames in os.walk(config.parent_data_dir + config.datadir):
            for filename in filenames:
                if filename.find('json.zip') > -1:
                    print('running on: ' + filename)
                    config.eval_set = filename
                    ptrnet = WebAsKB_PtrVocabNet()
                    ptrnet.load_data(config.parent_data_dir + config.datadir, '', config.eval_set)
                    ptrnet.init()
                    ptrnet.eval()
    else:
        ptrnet = WebAsKB_PtrVocabNet()
        ptrnet.load_data(config.parent_data_dir + config.datadir ,'', config.eval_set)
        ptrnet.init()
        ptrnet.eval()

elif args.operation == 'sample_trajectories':
    config.gen_trajectories = True
    config.sample_output_dist = True
    ptrnet = WebAsKB_PtrVocabNet()
    ptrnet.load_data(config.noisy_supervision_dir + config.datadir ,'train', config.eval_set)
    ptrnet.init()
    ptrnet.eval()

elif args.operation == 'train_supervised':
    config.PERFORM_TRAINING = True
    config.LOAD_SAVED_MODEL = False
    config.max_evalset_size = 2000
    ptrnet = WebAsKB_PtrVocabNet()
    ptrnet.load_data(config.noisy_supervision_dir ,'train', config.eval_set)
    ptrnet.init()
    ptrnet.train()

elif args.operation == 'preproc_RL':
    ptrnet = WebAsKB_PtrVocabNet()
    ptrnet.preproc_rl_data()

elif args.operation == 'train_RL':
    config.PERFORM_TRAINING = True
    config.LOAD_SAVED_MODEL = True
    config.RL_Training = True
    config.always_save_model = True

    ptrnet = WebAsKB_PtrVocabNet()
    ptrnet.load_data(config.rl_preproc_data + config.datadir ,'train', config.eval_set)
    ptrnet.init()
    ptrnet.train_rl()

elif args.operation == 'splitqa':
    config.PERFORM_TRAINING = False
    splitqa = SplitQA()
    splitqa.run_executors()
    splitqa.compute_final_results()