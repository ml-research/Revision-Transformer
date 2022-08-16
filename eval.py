from sentence_transformers import SentenceTransformer
import pandas as pd
import argparse
from nlgeval import NLGEval
from tqdm import tqdm
import torch
from rtpt import RTPT
from delphi import convert_moral_acceptability_text_to_class_wild_v11, get_moral_agreement_text_accuracy, \
    convert_moral_acceptability_text_to_class

nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)

parser = argparse.ArgumentParser(description='XIL on Retriever')
# parser.add_argument('--file', default='True_qna_knn1', type=str,
#                     choices=['True_qna_knn1', 'True_qna_knn2', 'False_qna_knn1', 'False_qna_knn2', 'True_plain_knn1',
#                              'True_plain_knn2', 'False_plain_knn1', 'False_plain_knn2'],
#                     help='which combination to evaluate')
parser.add_argument('--all', default=False, type=bool,
                    help='compute on all or only on contextualized ones')
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
args = parser.parse_args()
torch.set_num_threads(40)

df = pd.read_csv(f'results/{args.set}/results_baseline.csv', sep='\t')
df_ = pd.read_csv(f'results/{args.set}/results_{args.filter}_{args.type}_knn{args.nn}.csv', sep='\t')

if args.similarity:
    model_names = ["sentence-transformers/sentence-t5-xl", "sentence-transformers/sentence-t5-xxl",
                   "all-mpnet-base-v2",
                   "gtr-t5-xxl", "gtr-t5-xl"]
    model_name = model_names[0]
    model_name = 'models/' + model_name
    model_sentence = SentenceTransformer(model_name)
    model_sentence.cuda()
    similarities = []
    similarities_ = []

rtpt = RTPT(name_initials='FF', experiment_name='XIL_RET', max_iterations=len(df))
rtpt.start()

scores = []
scores_ = []
if args.all:
    # 1) compute both on all samples even on the ones where the model did not contextualize
    for gt, pred, gt_, pred_ in tqdm(zip(df['Ground Truth'], df['Prediction'], df_['Ground Truth'], df_['Prediction']),
                                     total=len(df)):
        rtpt.step()
        score_baseline = nlgeval.compute_individual_metrics([gt], pred)
        scores.append(score_baseline)
        if args.similarity:
            emb = model_sentence.encode([gt, pred, gt_, pred_], convert_to_tensor=True, device='cuda')
            sim_baseline = torch.cosine_similarity(emb[0], emb[1], dim=-1)
            similarities.append(sim_baseline)
        if ~pred_.isnull().any():
            scores_.append(nlgeval.compute_individual_metrics([gt_], pred_))
            if args.similarity:
                similarities_.append(torch.cosine_similarity(emb[2], emb[3], dim=-1))
        else:
            scores_.append(score_baseline)
            if args.similarity:
                similarities_.append(sim_baseline)

else:
    # 2) or compute both only on samples that were contextualization happened
    # for row, row_ in tqdm(zip(df.iterrows(), df_.iterrows()), total=len(df)):
    for gt, pred, gt_, pred_ in tqdm(zip(df['Ground Truth'], df['Prediction'], df_['Ground Truth'], df_['Prediction']),
                                     total=len(df)):
        rtpt.step()
        if pred_:
            scores.append(nlgeval.compute_individual_metrics([gt], pred))
            scores_.append(nlgeval.compute_individual_metrics([gt_], pred_))
            if args.similarity:
                emb = model_sentence.encode([gt, pred, gt_, pred_], convert_to_tensor=True, device='cuda')
                similarities.append(torch.cosine_similarity(emb[0], emb[1], dim=-1))
                similarities_.append(torch.cosine_similarity(emb[2], emb[3], dim=-1))

df_scores = pd.DataFrame(scores, columns=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'])
mean_score = df_scores.mean(axis=0).round(4)
df_scores_ = pd.DataFrame(scores_, columns=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'])
mean_score_ = df_scores_.mean(axis=0).round(4)

if args.similarity:
    mean_sim = torch.mean(torch.stack(similarities))
    mean_std = torch.std(torch.stack(similarities))
    mean_sim_ = torch.mean(torch.stack(similarities_))
    mean_std_ = torch.std(torch.stack(similarities_))

_, _, delphi_score = convert_moral_acceptability_text_to_class_wild_v11(df['Ground Truth'], df['Prediction'])
accuracy = get_moral_agreement_text_accuracy(df['Ground Truth'], df['Prediction'])

_, _, delphi_score_ = convert_moral_acceptability_text_to_class_wild_v11(df_['Ground Truth'], df_['Prediction'])
accuracy_ = get_moral_agreement_text_accuracy(df_['Ground Truth'], df_['Prediction'])

txt_file = open(f'results/{args.set}/eval_{args.filter}_{args.type}_knn{args.nn}.txt', 'w+')
txt_file.write(f'NLG score:\n')
txt_file.write(f'baseline: \n{mean_score} \n\n')
txt_file.write(f'contextualized: \n{mean_score_} \n')

txt_file.write(f'------------------------------------\n')
txt_file.write(f'Delphi\n')
txt_file.write(f'baseline: \ncov: {delphi_score:.3f} \n'
               f'acc_exact: {accuracy[0]:.3f} acc_polarity: {accuracy[1]:.3f}\n\n')
txt_file.write(f'contextualized: \ncov: {delphi_score_:.3f} \n'
               f'acc_exact: {accuracy_[0]:.3f} acc_polarity: {accuracy_[1]:.3f}\n')

if args.similarity:
    txt_file.write(f'------------------------------------\n')
    txt_file.write(f'Similarity\n')
    txt_file.write(f'baseline: {mean_sim:.3f} +- {mean_std:.3f}\n')
    txt_file.write(f'contextualized: {mean_sim_:.3f}  +- {mean_std_:.3f}\n')
txt_file.close()
