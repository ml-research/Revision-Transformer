from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import torch

detokenizer = TreebankWordDetokenizer()


def questionize(string):
    string_tknzd = nltk.word_tokenize(string)
    tags = nltk.pos_tag(string_tknzd)
    if ('VB' not in tags[0][1] or 'MD' not in tags[0][1]) and ('VB' in tags[1][1] or 'MD' in tags[1][1]):
        if "'s" in string_tknzd[1]:
            string_tknzd[1] = string_tknzd[1].replace("'s", "is")
        elif "'re" in string_tknzd[1]:
            string_tknzd[1] = string_tknzd[1].replace("'re", "are")
        if string_tknzd[2] == "n't":
            string_tknzd[2], string_tknzd[0], string_tknzd[1] = string_tknzd[0], string_tknzd[1], string_tknzd[2]
        else:
            string_tknzd[1], string_tknzd[0] = string_tknzd[0], string_tknzd[1]
        string = detokenizer.detokenize(string_tknzd).capitalize()

    # correct punctuation
    if string[-1] in '.!':
        string = string[:-1] + '?'
    else:
        string = string + '?'
    # design input type
    # string = f'Q: {string} A: '
    string = f'Question: {string} Answer: '
    # string = f'{string} '
    return string


def contextualize(inp, args, model_sentence, contexts, retriever, thresh=0.9):
    if args.unprocessed:
        input_emb = model_sentence.encode(inp['text'], convert_to_tensor=True, device='cuda')
    else:
        if torch.is_tensor(inp['embedding']):
            input_emb = torch.tensor(inp['embedding']).cuda()
        else:
            input_emb = torch.stack(inp['embedding']).cuda()
            input_emb = input_emb.permute(1, 0)
    sim = torch.cosine_similarity(input_emb.unsqueeze(0), contexts.unsqueeze(1), dim=-1)
    # sim = - torch.dist(input_emb, contexts, p=2)
    # sim = np.einsum("ij,kj->ik", contexts, input_emb)
    if args.nn == 1:
        context_val, context_idx = torch.topk(sim, dim=0, k=args.nn)
        # mask = (context_val > thresh)[0]
        # context = retriever[context_idx.tolist()[0]]
        # context = list(context.values())
        # for idx, ctx in enumerate(context):
        #     ctx[ctx == (~mask)] = False
        #     context[idx] = ctx
        # if context_val > thresh:
        #     context = retriever[context_idx.tolist()[0]]
        if context_val > thresh:
            context = retriever[context_idx.item()]
        else:
            return False
            # return questionize(inp['text'])
        if args.type == 'qna':
            context['text'] = questionize(context['text'])
            input_contexted = context['text'] + context['text_label'] + ' ' + questionize(inp['text'])
        elif args.type == 'plain':
            input_contexted = context['text'] + ' ' + questionize(inp['text'])
    else:
        input_contexted = []
        context_vals, context_idxs = torch.topk(sim, dim=0, k=args.nn)
        for context_val, context_idx in zip(context_vals, context_idxs):
            # only append if context is similar enough
            if context_val > thresh:
                context = retriever[context_idx.item()]
            else:
                continue
            if args.type == 'qna':
                context['text'] = questionize(context['text'])
                ctxt = context['text'] + context['text_label'] + ' '  # + questionize(inp['text'])
            elif args.type == 'plain':
                ctxt = context['text'] + ' '  # + questionize(inp['text'])
            input_contexted.append(ctxt)
        if not input_contexted:
            return False
        input_contexted = ''.join(input_contexted) + questionize(inp['text'])
    return input_contexted


def filter_output(text_out, inp):
    """
    filter generated text samples
    """
    for j, t in enumerate(text_out):
        # 1) remove inp from generated sample
        # [:-1] as the last element is space, if now the first prediction element is punctuation then it doesn't replace
        t = t.replace(inp[:-1], '')
        # # 2) remove part of generated sample if 'Question' is generated
        # t = t.split('Question', 1)[0]
        # # 3) remove generated sample if bad_words are generated
        # bad_words = ['??', '__', '¡', '¬', '**', '##']  # 'Â'
        # if any(s in t for s in bad_words):
        #     t = ''
        # # 4) remove generated sample if it does not contain force_words
        # force_words = ['yes', 'no']
        # if not any(s in t for s in force_words):
        #     t = ''
        text_out[j] = t
    # remove empty ones
    # text_output = list(filter(None, text_output))
    return text_out


def prepare_models():
    """
    load all models and store them locally, such that a server crash does not requires downloading all models again
    """

    model_names = ["bigscience/T0_3B", "bigscience/T0pp", "google/t5-11b-ssm-tqa", "EleutherAI/gpt-j-6B",
                   "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neox-20b", "t5-3b", "t5-11b"]
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        if "gpt" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_name = 'models/' + model_name
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)

    model_names = ["sentence-transformers/sentence-t5-xl", "sentence-transformers/sentence-t5-xxl", "all-mpnet-base-v2",
                   "gtr-t5-xxl", "gtr-t5-xl"]
    for model_name in model_names:
        model_sentence = SentenceTransformer(model_name)
        model_name = 'models/' + model_name
        model_sentence.save(model_name)


def gen_answer(inp, output, tokenizer, model):
    input_ids = tokenizer(inp, return_tensors="pt", truncation=True, padding=True,
                          add_special_tokens=False).input_ids.cuda()
    prediction = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=30, min_length=4,
                                do_sample=True, top_k=tokenizer.vocab_size // 10, temperature=0.1, )

    text_out = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    # Todo: [0] currently only for batchsize=1, not parallelized yet
    text_out = filter_output(text_out, inp)[0]
    output.extend([inp, text_out])
    return output

# # prediction with beam_search
# prediction = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=30, min_length=4,
#                             do_sample=True, top_k=tokenizer.vocab_size // 10, temperature=0.1, )
#                             # top_p=0.9,
#                             # repetition_penalty=0.7,
#                             # early_stopping=True,
#                             # force_words_ids=force_words_ids,
#                             # bad_words_ids=bad_words_ids,
#                             # no_repeat_ngram_size=3,  # 3
#                             # num_beam_groups=2,
#                             # diversity_penalty=0.5,
#                             # num_beams=10,)  # 5
#                             # num_return_sequences=5, )
