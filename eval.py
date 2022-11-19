from sentence_transformers import SentenceTransformer
import pandas as pd
import argparse
from nlgeval import NLGEval
from tqdm import tqdm
import torch
from rtpt import RTPT
from misc.text2class import normalize_label
from utils import acc_yes_no, get_moral_agreement_text_accuracy

from transformers import (
    AutoTokenizer, FSMTForConditionalGeneration, MarianMTModel,
    MarianTokenizer)

class Translator:
    def __init__(self, model_name, max_batch_tokens=1000):
        self.max_batch_tokens = max_batch_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name.startswith("facebook"):
            self.model = FSMTForConditionalGeneration.from_pretrained(model_name)
        elif model_name.startswith("Helsinki") or model_name.startswith("gsarti"):
            self.model = MarianMTModel.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def __call__(self, sentences):
        results = []
        batch = []

        def translate_batch():
            max_len = max(len(sent) for sent in batch)
            for sent in batch:
                for _ in range(max_len - len(sent)):
                    sent.append(tokenizer.pad_token)
            id_batch = [tokenizer.convert_tokens_to_ids(sent) for sent in batch]
            input_ids = torch.tensor(id_batch).to(device)
            outputs = model.generate(input_ids)
            for sent_out in outputs:
                decoded = tokenizer.decode(sent_out, skip_special_tokens=True)
                results.append(decoded)

        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            tokens.append("</s>")
            batch.append(tokens)

            if len(batch) * max(len(sent) for sent in batch) > self.max_batch_tokens:
                translate_batch()
                batch = []

        if batch:
            translate_batch()

        return results


nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)

parser = argparse.ArgumentParser(description='XIL on Retriever')
parser.add_argument('--set', default='test', type=str, choices=['validation', 'test'],
                    help='which set to evaluate?')
parser.add_argument('--thresh', default=0.9, type=float,
                    help='threshold for retrieving neighbors')
parser.add_argument('--filter', default=False, type=bool,
                    help='only use true statements from dataset for context')
parser.add_argument('--type', default='qna', type=str, choices=['plain', 'qna'],
                    help='context type, whether to use plain context or context in question answer style')
parser.add_argument('--nn', default=1, type=int,
                    help='how many neighbors to use a context?')
parser.add_argument('--similarity', default=False, type=bool,
                    help='compute similarity?')
parser.add_argument('--normalize', default=False, type=bool,
                    help='normalize labels?')
parser.add_argument('--version', default='agreement', type=str, choices=['agreement', 'acceptability', 'comparison'],
                    help='which moral type in norms dataset?')
parser.add_argument('--lang', default='', type=str, choices=['de', 'en', 'fr', 'ro', ''],
                    help='for which language?')
parser.add_argument('--simulate', default=False, type=bool,
                    help='compute results with validation-simulated feedback?')
parser.add_argument('--model_name', default='bloom-3b', type=str,
                    choices=["T0-3B", "T0pp", "t5-11b-ssm-tqa", "t5-3b", "t5-11b", "bloom-1b", "bloom-3b", "bloom-7b",
                             "bloom"])
parser.add_atugment("--mt_model_name", type=str, default="facebook/wmt19-de-en")
args = parser.parse_args()
torch.set_num_threads(40)

if args.lang:
    pth = f'results/crosslingual/{args.version}'
else:
    pth = f'results/{args.version}'
df = pd.read_csv(pth + f'/results_baseline_lang={args.lang}_{args.model_name}.csv', sep='\t', dtype=str)

if args.simulate:
    nm = f'/results_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_{args.model_name}_simulateTrue.csv'
else:
    nm = f'/results_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_{args.model_name}.csv'
df_gen = pd.read_csv(pth + nm, sep='\t', dtype=str)

if args.lang and args.lang != 'en':
    translator = Translator(args.mt_model_name)
    # TODO: translate answers and GT to english

if args.lang:
    # remove part of output after newline and if NaN replace with empty string
    df['Prediction'] = df['Prediction'].apply(lambda x: x.split('\n', 1)[0] if isinstance(x, str) else '')
    df_gen['Prediction'] = df_gen['Prediction'].apply(lambda x: x.split('\n', 1)[0] if isinstance(x, str) else '')

if args.similarity:
    model_names = ["sentence-transformers/sentence-t5-xl", "sentence-transformers/sentence-t5-xxl",
                   "all-mpnet-base-v2", "gtr-t5-xxl", "gtr-t5-xl"]
    model_name = model_names[0]
    model_name = 'models/' + model_name
    model_sentence = SentenceTransformer(model_name)
    model_sentence.cuda()
    similarities = []
    similarities_gen = []

# # overlap of predicition with answer from context: ~72%
# if args.type == 'qna' and args.nn == 1:
#     # TODO: implemented correctly? distinguish (non-)contextualized examples, also check if Question/Answer split works
#     if args.lang == 'en' or not args.lang:
#         context_prediction = df_gen['Input'].apply(
#             lambda x: x.split('Answer: ')[1].split(' Question: ')[0] if pd.notna(x) else 0).copy()
#     elif args.lang == 'de':
#         context_prediction = df_gen['Input'].apply(
#             lambda x: x.split('Antwort: ')[1].split(' Frage: ')[0] if pd.notna(x) else 0).copy()
#     else:
#         raise ValueError('Not yet implemented for other langues')
#     print(f"overlap: {sum(context_prediction == df_gen['Prediction']) / df_gen.shape[0]:.4f}")

data_ = zip(df['Ground Truth'], df['Prediction'], df_gen['Prediction'], df_gen['Input'])

rtpt = RTPT(name_initials='FF', experiment_name='RiT_eval', max_iterations=len(df))
rtpt.start()

scores = []
scores_ = []
scores_gen = []

# compute N-gram metrics and sentence embedding similarity
for gt, pred, pred_gen, inp in tqdm(data_, total=len(df)):
    rtpt.step()
    score_baseline = nlgeval.compute_individual_metrics([gt], pred)
    scores.append(score_baseline)
    if args.similarity:
        emb = model_sentence.encode([gt, pred, pred_gen], convert_to_tensor=True,
                                    device='cuda')
        sim_baseline = torch.cosine_similarity(emb[0], emb[1], dim=-1)
        similarities.append(sim_baseline)
    if pd.notna(inp) and inp:
        scores_gen.append(nlgeval.compute_individual_metrics([gt], pred_gen))
        if args.similarity:
            similarities_gen.append(torch.cosine_similarity(emb[0], emb[2], dim=-1))

###### split data ############
# locate empty entries
idx = df_gen['Input'].isna()

# split up performance in baseline and ctxt
df_ctxt_base = df.copy()
df_ctxt_base[~idx] = df_gen.loc[~idx].copy()
df_ctxt = df_gen.loc[~idx].copy().reset_index(drop=True)
df_base_res = df.loc[idx].copy().reset_index(drop=True)

# TODO add _base to baseline scores

cols = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
# split performance, remove CIDEr as we how only one ground truth answer per prediction
n_gram = pd.DataFrame(scores, columns=cols).drop('CIDEr', axis=1)
n_gram_gen = pd.DataFrame(scores_gen, columns=cols).drop('CIDEr', axis=1)
n_gram_ctxt_base = n_gram.copy()
n_gram_ctxt_base[~idx] = n_gram_gen.copy().reset_index(drop=True)
n_gram_base_res = n_gram[idx].copy().reset_index(drop=True)

# compute mean n-gram scores
n_gram = n_gram.mean(axis=0).round(4)
n_gram_ctxt_base = n_gram_ctxt_base.mean(axis=0).round(4)
n_gram_ctxt = n_gram_gen.mean(axis=0).round(4)
n_gram_base_res = n_gram_base_res.mean(axis=0).round(4)

if args.similarity:
    mean_sim = torch.mean(torch.stack(similarities))
    mean_std = torch.std(torch.stack(similarities))
    mean_sim_gen = torch.mean(torch.stack(similarities_gen))
    mean_std_gen = torch.std(torch.stack(similarities_gen))

# compute yes no accuracy
acc_yn = acc_yes_no(df['Ground Truth'].copy(), df['Prediction'].copy())
acc_yn_ctxt_base = acc_yes_no(df_ctxt_base['Ground Truth'].copy(), df_ctxt_base['Prediction'].copy())
acc_yn_ctxt = acc_yes_no(df_ctxt['Ground Truth'].copy(), df_ctxt['Prediction'].copy())
acc_yn_base_res = acc_yes_no(df_base_res['Ground Truth'].copy(), df_base_res['Prediction'].copy())

if args.normalize:
    # TODO: converts NaNs to empty, thats a problem. Also, the other delphi methods normalize again, do check!
    df['Ground Truth'] = df['Ground Truth'].apply(normalize_label)
    df['Prediction'] = df['Prediction'].apply(normalize_label),
    df_ctxt_base['Ground Truth'] = df_ctxt_base['Ground Truth'].apply(normalize_label)
    df_ctxt_base['Prediction'] = df_ctxt_base['Prediction'].apply(normalize_label)
    df_ctxt['Ground Truth'] = df_ctxt['Ground Truth'].apply(normalize_label)
    df_ctxt['Prediction'] = df_ctxt['Prediction'].apply(normalize_label)
    df_base_res['Ground Truth'] = df_base_res['Ground Truth'].apply(normalize_label)
    df_base_res['Prediction'] = df_base_res['Prediction'].apply(normalize_label)

# compute polarity accuracy
acc_pol = get_moral_agreement_text_accuracy(df['Ground Truth'], df['Prediction'])
acc_pol_ctxt_base = get_moral_agreement_text_accuracy(df_ctxt_base['Ground Truth'], df_ctxt_base['Prediction'])
acc_pol_ctxt = get_moral_agreement_text_accuracy(df_ctxt['Ground Truth'], df_ctxt['Prediction'])
if sum(idx):
    acc_pol_base_res = get_moral_agreement_text_accuracy(df_base_res['Ground Truth'], df_base_res['Prediction'])
else:
    acc_pol_base_res = -1.

types = ['baseline', 'context with base', 'context only', 'residual baseline']
yns = [acc_yn, acc_yn_ctxt_base, acc_yn_ctxt, acc_yn_base_res]
pols = [acc_pol, acc_pol_ctxt_base, acc_pol_ctxt, acc_pol_base_res]
ngrams = [n_gram, n_gram_ctxt_base, n_gram_ctxt, n_gram_base_res]
# sims = [similarities, similarities, similarities_gen, similarities]

if args.simulate:
    nm = f'/eval_{args.filter}_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_{args.model_name}_simulateTrue.txt'
else:
    nm = f'/eval_{args.filter}_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_{args.model_name}.txt'

txt_file = open(pth + nm, 'w+')
txt_file.write(f'NLG evaluation:\n')
txt_file.write(f"num_ctxtd: {sum(~idx)}, total: {idx.shape[0]}\n")
for i in range(len(types)):
    txt_file.write(f'\n------------------------------------\n')
    txt_file.write(f'{types[i]}: \n')
    txt_file.write(f'acc_yn: {yns[i]:.4f}\n')
    txt_file.write(f'acc_pol: {pols[i]:.4f}\n')
    txt_file.write(f'n_gram: \n{ngrams[i]} \n')
    if args.similarity:
        txt_file.write(f'similarity: \n{sims[i]:.4f} \n')

txt_file.close()
