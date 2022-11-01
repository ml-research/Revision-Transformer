import torch
from utils import questionize, questionize_cl, contextualize, gen_answer, load_models
from datasets import load_from_disk  # , load_dataset
from rtpt import RTPT
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='XIL on Retriever')
parser.add_argument('--filter', default=False, type=bool,
                    help='use only true statements from dataset for context')
parser.add_argument('--type', default='qna', type=str, choices=['plain', 'qna'],
                    help='context type, whether to use plain context or context in question answer style')
parser.add_argument('--unprocessed', default=False, type=bool,
                    help='whether to use a dataset where sentence embeddings for inputs is already available')
parser.add_argument('--nn', default=1, type=int,
                    help='how many neighbors to use as context?')
parser.add_argument('--thresh', default=0.9, type=float,
                    help='threshold for retrieving neighbors')
parser.add_argument('--baseline', default=False, type=bool,
                    help='compute results for baseline, i.e. data without context, or RiT?')
parser.add_argument('--version', default='agreement', type=str, choices=['agreement', 'acceptability', 'comparison'],
                    help='which moral type in norms dataset?')
parser.add_argument('--simulate', default=False, type=bool,
                    help='compute results with validation-simulated feedback?')
parser.add_argument('--lang', default='', type=str, choices=['de', 'en', 'fr', 'ro', ''],
                    help='for which language?')
parser.add_argument('--batch_size', default=250, type=int)
parser.add_argument('--model_name', default='bloom-3b', type=str,
                    choices=["T0-3B", "T0pp", "t5-11b-ssm-tqa", "t5-3b", "t5-11b", "bloom-1b", "bloom-3b", "bloom-7b",
                             "bloom"])
parser.add_argument('--model_name_sent', default='t5-xl', type=str,
                    choices=["t5-xl", "t5-xxl", "all-mpnet-base-v2", "gtr-t5-xxl", "gtr-t5-xl", "multilingual"],
                    help='model for context retrieval via sentence similarity')


def run_generation():
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=0)
    rtpt = RTPT(name_initials='FF', experiment_name='RiT_generate', max_iterations=len(dataloader))
    rtpt.start()

    generate_kwargs = dict(max_new_tokens=8, do_sample=False, min_length=4, pad_token_id=tokenizer.eos_token_id)
    outputs = []
    for data in tqdm(dataloader):
        rtpt.step()
        output = gen_answer(data, tokenizer, model, generate_kwargs)
        outputs.extend(output)

    # TODO: combine query and answer to a new query and ask for an explanation (why?)
    df = pd.DataFrame(outputs, columns=['Input', 'Prediction', 'Ground Truth'])
    if args.lang:
        pth = f'results/crosslingual/{args.version}'
    else:
        pth = f'results/{args.version}'
    if args.baseline:
        df.to_csv(pth + f'/results_baseline_lang={args.lang}_{args.model_name}.csv', sep='\t')
    else:
        if args.simulate:
            df.to_csv(
                pth + f'/results_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_simulate{args.simulate}_{args.model_name}.csv',
                sep='\t')
        else:
            df.to_csv(
                pth + f'/results_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_{args.model_name}.csv',
                sep='\t')


if __name__ == "__main__":
    args = parser.parse_args()
    torch.set_num_threads(60)

    if not args.filter:
        filt = '_unfiltered'
    else:
        filt = ''

    if args.lang:
        fname = f'/workspace/repositories/norms/crosslingual/{args.version}/retriever_processed_norms_cl'
        retriever = load_from_disk(fname + f'_train_{args.lang}{filt}')
        test_data = load_from_disk(fname + f'_test_{args.lang}_unfiltered')
    else:
        fname = f'/workspace/repositories/norms/{args.version}/retriever_processed_norms'
        retriever = load_from_disk(fname + '_train' + filt)
        test_data = load_from_disk(fname + f'_test_unfiltered')

    if args.simulate:
        # user simulation with pre-selected contexts
        ids = np.loadtxt('results/simulation_baseFalse.txt')
        retriever = retriever.select(ids)

    contexts = torch.tensor(retriever['embedding']).cuda()

    # load models
    model, tokenizer, model_sentence = load_models(args)

    # apply RiT before generation for speedup
    if args.baseline:
        # no context, i.e. directly questionize text
        if args.lang:
            test_data = test_data.map(lambda x: {"text": questionize_cl(x['text'], args.lang)}, remove_columns=["text"])
        else:
            test_data = test_data.map(lambda x: {"text": questionize(x['text'], args.lang)}, remove_columns=["text"])
    else:
        # contextualize input data
        test_data = test_data.map(
            lambda x: {"text": contextualize(x, args, model_sentence, contexts, retriever, thresh=args.thresh)[0]},
            remove_columns=["text"])

    del retriever, contexts
    torch.cuda.empty_cache()

    run_generation()
