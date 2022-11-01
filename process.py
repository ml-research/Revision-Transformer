from datasets import Features, Sequence, Value, load_dataset
from functools import partial
from sentence_transformers import SentenceTransformer
import argparse
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk

detokenizer = TreebankWordDetokenizer()

parser = argparse.ArgumentParser(description='XIL on Retriever')
parser.add_argument('--filter', default=False, type=bool,
                    help='only use true statements from dataset')
parser.add_argument('--dataset', default='norms', type=str,
                    choices=['norms', 'norms_cl', 'squadv2', 'msmarco', 'nq', 'trivia', 'arc', 'searchqa'],
                    help='which dataset?')
parser.add_argument('--version', default='agreement', type=str, choices=['agreement', 'acceptability', 'comparison'],
                    help='which moral type in norms dataset?')

args = parser.parse_args()
pth = '/workspace/repositories/norms'

if not args.version:
    args.version = args.dataset

if args.dataset == 'norms':
    dataset_ = load_dataset("csv",
                            data_files={'train': pth + f"/{args.version}/train.moral_{args.version}.tsv",
                                        'validation': pth + f"/{args.version}/validation.moral_{args.version}.tsv",
                                        'test': pth + f"/{args.version}/test.moral_{args.version}.tsv"},
                            delimiter="\t")
if args.dataset == 'norms_cl':
    dataset_ = load_dataset("csv",
                            data_files={'train': pth + f"/crosslingual/{args.version}/train.moral_{args.version}.tsv",
                                        'validation': pth + f"/crosslingual/{args.version}/validation.moral_{args.version}.tsv",
                                        'test': pth + f"/crosslingual/{args.version}/test.moral_{args.version}.tsv"},
                            delimiter="\t")
elif args.dataset == 'squadv2':
    dataset_ = load_dataset("squad_v2")
elif args.dataset == 'msmarco':
    dataset_ = load_dataset("ms_marco", "v2.1")
elif args.dataset == 'nq':
    dataset_ = load_dataset("natural_questions")
elif args.dataset == 'trivia':
    dataset_ = load_dataset("trivia_qa", "rc")
elif args.dataset == 'arc':
    dataset_ = load_dataset("ai2_arc", "ARC-Challenge")
elif args.dataset == 'searchqa':
    dataset_ = load_dataset("search_qa", "raw_jeopardy")


def embed(documents: dict, model_sentence: SentenceTransformer) -> dict:
    """Compute the embeddings of document passages"""
    embeddings = model_sentence.encode(documents["text"], convert_to_tensor=True, device='cuda')
    return {"embedding": embeddings.detach().cpu().numpy()}


def invert_statement(text):
    string_tknzd = nltk.word_tokenize(text)
    breakpoint()
    # tags = nltk.pos_tag(string_tknzd)
    # if ('VB' not in tags[0][1] or 'MD' not in tags[0][1]) and ('VB' in tags[1][1] or 'MD' in tags[1][1]):
    if string_tknzd[2] == "not" or string_tknzd[2] == "n't":
        string_tknzd.pop(2)
    else:
        string_tknzd[2] = "not"
    text = detokenizer.detokenize(string_tknzd).capitalize()
    return text


# sentence-transformers/stsb-xlm-r-multilingual
model_sentence = SentenceTransformer('models/' + 'sentence-transformers/sentence-t5-xl')
model_sentence.cuda()
# iterate over all dataset splits
for set_ in dataset_:
    dataset = dataset_[set_]
    fname = pth + f'/{args.version}/retriever_processed_{args.dataset}_{set_}'
    # preprocess
    if args.dataset == 'norms':
        if args.version == 'comparison':
            print('not implemented')
        else:
            # 1) remove unwanted cols
            dataset = dataset.remove_columns(['Unnamed: 0', 'input_type', 'source'])
            if args.version == 'acceptability':
                dataset = dataset.remove_columns(['pattern'])
            dataset = dataset.rename_column('input_sequence', 'text')
            # 2) pop item if label==-1
            if args.filter:
                # Todo: fix filtering, instead of removing the example it could simple be negated
                # dataset = dataset.map(lambda x: {'text': invert_statement(x['text']) if x['class_label'] == -1})
                dataset = dataset.filter(lambda example: example['class_label'] == 1)
            else:
                fname += '_unfiltered'
            dataset = dataset.remove_columns('class_label')

    elif args.dataset == 'squadv2':
        # 1) remove unwanted cols
        dataset = dataset.remove_columns(['id', 'title', 'context'])
        dataset = dataset.rename_column('question', 'text')
        dataset = dataset.map(lambda x: {'answers': x['answers']['text'][0] if x['answers']['text'] else ''})
        dataset = dataset.rename_column('answers', 'text_label')

    elif args.dataset == 'trivia':
        # 1) remove unwanted cols
        dataset = dataset.remove_columns(['question_id', 'question_source', 'entity_pages', 'search_results'])
        dataset = dataset.rename_column('question', 'text')
        dataset = dataset.map(lambda x: {'answer': x['answer']['value']})
        dataset = dataset.rename_column('answer', 'text_label')

    elif args.dataset == 'searchqa':
        # 1) remove unwanted cols
        dataset = dataset.remove_columns(
            ['category', 'air_date', 'value', 'round', 'show_number', 'search_results'])
        dataset = dataset.rename_column('question', 'text')
        dataset = dataset.rename_column('answer', 'text_label')

    # And compute the embeddings
    new_features = Features(
        {"text": Value("string"), "text_label": Value("string"), "embedding": Sequence(Value("float32"))}
    )
    dataset = dataset.map(
        partial(embed, model_sentence=model_sentence),
        batched=True,
        batch_size=32,
        features=new_features,
    )

    # And finally save dataset
    dataset.save_to_disk(fname)
