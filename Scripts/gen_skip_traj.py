from config import *
# initializing specific config params
config.data_dir = '../Data/'
config.init()
config.EVALUATION_SET = 'train'
config.name = 'ptr_vocab_1skip_samp'
config.out_subdir = config.name + '/'
config.input_model = 'ptr_vocab' + '/'
from webaskb_ptr_vocab_net import WebAsKB_PtrVocabNet
random.seed(1)
ptrnet = WebAsKB_PtrVocabNet()
ptrnet.load_data(config.noisy_supervision_dir + 'train.json',\
                config.noisy_supervision_dir + config.EVALUATION_SET + '.json')
ptrnet.init()

model_output = []
model_format_errors = {}
pairs_dev = [ptrnet.net.pairs_dev[i] for i in range(len(ptrnet.net.pairs_dev))]
sample_size = min(config.max_evalset_size,len(ptrnet.net.pairs_dev))
ptrnet.net.model.init_stats()

ptrnet.net.test_loss = 0

accuracy_avg = 0
for test_iter in range(0, sample_size):
    if test_iter % 200 == 0:
        print(test_iter)
    testing_pair = pairs_dev[test_iter]

    test_loss , output_seq, loss, output_dists, output_masks  = ptrnet.net.model.forward(testing_pair['x'], testing_pair['y'])
    ptrnet.net.test_loss += test_loss

    # generating model output
    try:
        model_output +=  ptrnet.net.model.format_model_output(testing_pair, output_seq, output_dists, output_masks)
    except Exception as inst:
        if inst.args[0] == 'format_model_output_error':
            if inst.args[1] in model_format_errors:
                model_format_errors[inst.args[1]] += 1
            else:
                model_format_errors[inst.args[1]] = 1
        else:
            print(traceback.format_exc())
        # adding empty output
        #model_output += [{'ID': testing_pair['aux_data']['ID'], 'question': testing_pair['aux_data']['question'], \
        #       'answers': testing_pair['aux_data']['answers']}]

with open(config.split_points_dir + config.out_subdir + config.EVALUATION_SET + '.json', 'w') as outfile:
    outfile.write(json.dumps(model_output))
pd.DataFrame(model_output).to_csv(config.split_points_dir + config.out_subdir + config.EVALUATION_SET + '.csv',encoding="utf-8",index=False)
    