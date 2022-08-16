import torch
# import numpy as np
from utils import questionize, contextualize, gen_answer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk  # , load_dataset
from rtpt import RTPT
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='XIL on Retriever')
parser.add_argument('-m', '--mode', default='self', type=str, choices=['self', 'RAG'],
                    help='self implemented or RAG implementation')
parser.add_argument('--filter', default=False, type=bool,
                    help='only use true statements from dataset for context')
parser.add_argument('--type', default='qna', type=str, choices=['plain', 'qna'],
                    help='context type, whether to use plain context or context in question answer style')
parser.add_argument('--unprocessed', default=False, type=bool,
                    help='whether to use a dataset where sentence embeddings for inputs is already available')
parser.add_argument('--nn', default=1, type=int,
                    help='how many neighbors to use a context?')
parser.add_argument('--set', default='test', type=str, choices=['validation', 'test'],
                    help='which set to evaluate?')
parser.add_argument('--thresh', default=0.9, type=float,
                    help='threshold for retrieving neighbors')
parser.add_argument('--baseline', default=False, type=bool,
                    help='compute results also for baseline? i.e. data without context')


# parser.add_argument('--dataset', default='norms', type=str,
#                     choices=['norms', 'squadv2', 'msmarco', 'nq', 'trivia', 'arc', 'searchqa'],
#                     help='which dataset?')


def run_generation():
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
    rtpt = RTPT(name_initials='FF', experiment_name='XIL_RET', max_iterations=len(dataloader))
    rtpt.start()

    # suppress the generation of certain tokens, e.g. the underscore
    # bad_words = ['????', '________', '__', '_____', '__________________________________________________', '??????',
    #              '_________________________________________________', '____', '_______', 'Â', '¡', '¬',
    #              '___________________________________________________', '_________________',
    #              '______________________________________________________',
    #              '________________________________________________', '________________________________________________']
    # bad_words = ['Question']
    # bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    # force_words_ids = [tokenizer(['Yes', 'No'], add_special_tokens=False).input_ids]

    outputs = []
    for data in tqdm(dataloader):
        rtpt.step()

        data['text'] = data['text'][0].capitalize()
        data['text_label'] = data['text_label'][0]
        output = []

        if args.baseline:
            # no context, i.e. directly questionize text
            inp = questionize(data['text'])
            output = gen_answer(inp, output, tokenizer, model)
        else:
            # contextualize input data
            inp = contextualize(data, args, model_sentence, contexts, retriever, thresh=0.9)
            if not inp:
                output.extend([inp, ''])
                continue
            output = gen_answer(inp, output, tokenizer, model)

        output.append(data['text_label'])
        outputs.append(output)
        # TODO: combine query and answer to a new query and ask for an explanation (why?)

    df = pd.DataFrame(outputs, columns=['Input', 'Prediction', 'Ground Truth'])
    if args.baseline:
        df.to_csv(f'results/{args.set}/results_baseline.csv', sep='\t')
    else:
        df.to_csv(f'results/{args.set}/results_{args.filter}_{args.type}_knn{args.nn}.csv', sep='\t')


if __name__ == "__main__":
    args = parser.parse_args()

    torch.set_num_threads(60)

    fname = f'/workspace/repositories/norms/retriever_processed_norms'
    if not args.filter:
        filt = '_unfiltered'
    else:
        filt = ''

    # # split train set in retriever and train data
    # database = load_from_disk(f'/workspace/repositories/norms/retriever_processed_norms_train')
    # database = database.train_test_split(test_size=0.5)
    # retriever, train_data = database['train'], database['test']

    # use train and test/validation split
    retriever = load_from_disk(fname + '_train' + filt)
    # train_data = load_from_disk(fname + f'_{args.set}' + filt)
    train_data = load_from_disk(fname + f'_{args.set}_unfiltered')

    contexts = torch.tensor(retriever['embedding']).cuda()
    # contexts = np.array(database['embedding'])

    # load models
    model_names = ["bigscience/T0_3B", "bigscience/T0pp", "google/t5-11b-ssm-tqa", "EleutherAI/gpt-j-6B",
                   "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neox-20b", "t5-3b", "t5-11b"]
    model_name = model_names[0]
    model_name = 'models/' + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.pad_token = tokenizer.eos_token

    if "gpt" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.cuda()

    if args.unprocessed:
        model_names = ["sentence-transformers/sentence-t5-xl", "sentence-transformers/sentence-t5-xxl",
                       "all-mpnet-base-v2",
                       "gtr-t5-xxl", "gtr-t5-xl"]
        model_name = model_names[0]
        model_name = 'models/' + model_name
        model_sentence = SentenceTransformer(model_name)
        model_sentence.cuda()
    else:
        model_sentence = ''

    run_generation()
