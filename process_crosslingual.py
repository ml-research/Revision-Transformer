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
parser.add_argument('--dataset', default='norms_cl', type=str,
                    choices=['norms', 'norms_cl', 'squadv2', 'msmarco', 'nq', 'trivia', 'arc', 'searchqa'],
                    help='which dataset?')
parser.add_argument('--version', default='agreement', type=str, choices=['agreement', 'acceptability', 'comparison'],
                    help='which moral type in norms dataset?')

args = parser.parse_args()
pth = '/workspace/repositories/norms'

if not args.version:
    args.version = args.dataset

dataset_ = load_dataset("csv",
                        data_files={'train': pth + f"/crosslingual/{args.version}/train.yes_no.tsv",
                                    'validation': pth + f"/crosslingual/{args.version}/validation.yes_no.tsv",
                                    'test': pth + f"/crosslingual/{args.version}/test.yes_no.tsv"},
                        delimiter="\t")


def embed(documents: dict, model_sentence: SentenceTransformer) -> dict:
    """Compute the embeddings of document passages"""
    embeddings = model_sentence.encode(documents["text"], convert_to_tensor=True, device='cuda')
    return {"embedding": embeddings.detach().cpu().numpy()}


# sentence-transformers/stsb-xlm-r-multilingual
model_sentence = SentenceTransformer('models/' + 'sentence-transformers/stsb-xlm-r-multilingual')
model_sentence.cuda()
# iterate over all dataset splits
for lang in ["de", "en", "fr", "ro"]:
    for set_ in dataset_:
        dataset = dataset_[set_]
        fname = pth + f'/crosslingual/{args.version}/retriever_processed_{args.dataset}_{set_}_{lang}'

        # replace with other language
        prompt_pth = pth + f"/crosslingual/{args.version}/{set_}.yes_no.prompt.{lang}"
        answer_pth = pth + f"/crosslingual/{args.version}/{set_}.yes_no.answer.{lang}"
        with open(prompt_pth) as file:
            prompt = file.readlines()
            prompt = [line.rstrip() for line in prompt]
        with open(answer_pth) as file:
            answer = file.readlines()
            answer = [line.rstrip() for line in answer]

        dataset = dataset.add_column("text", prompt[1:])
        dataset = dataset.add_column("text_label_", answer[1:])

        # preprocess
        # 1) remove unwanted cols
        dataset = dataset.remove_columns(['Unnamed: 0', 'input_type', 'source', 'input_sequence'])
        # 2) pop item if label==-1
        if args.filter:
            # Todo: fix filtering, instead of removing the example it could simple be negated
            # dataset = dataset.map(lambda x: {'text': invert_statement(x['text']) if x['class_label'] == -1})
            dataset = dataset.filter(lambda example: example['class_label'] == 1)
        else:
            fname += '_unfiltered'
        dataset = dataset.remove_columns('class_label')

        dataset = dataset.remove_columns('text_label')
        dataset = dataset.rename_column('text_label_', 'text_label')
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
