import torch
from utils import questionize, questionize_cl, contextualize, gen_answer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk  # , load_dataset
from rtpt import RTPT
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np

import io
import json
import os
from pathlib import Path

import torch.distributed as dist

import deepspeed
from huggingface_hub import snapshot_download
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode

parser = argparse.ArgumentParser(description='XIL on Retriever')
parser.add_argument('--filter', default=False, type=bool,
                    help='use only true statements from dataset for context')
parser.add_argument('--type', default='qna', type=str, choices=['plain', 'qna'],
                    help='context type, whether to use plain context or context in question answer style')
parser.add_argument('--unprocessed', default=False, type=bool,
                    help='whether to use a dataset where sentence embeddings for inputs is already available')
parser.add_argument('--nn', default=1, type=int,
                    help='how many neighbors to use as context?')
parser.add_argument('--thresh', default=0.45, type=float,
                    help='threshold for retrieving neighbors')
parser.add_argument('--baseline', default=False, type=bool,
                    help='compute results for baseline, i.e. data without context, or RiT?')
parser.add_argument('--version', default='agreement', type=str, choices=['agreement', 'acceptability', 'comparison'],
                    help='which moral type in norms dataset?')
parser.add_argument('--simulate', default=False, type=bool,
                    help='compute results with validation-simulated feedback?')
parser.add_argument('--lang', default='', type=str, choices=['de', 'en', 'fr', 'ro', ''],
                    help='for which language?')
parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--model_name', default='bloom-3b', type=str,
                    choices=["T0-3B", "T0pp", "t5-11b-ssm-tqa", "t5-3b", "t5-11b", "bloom-1b", "bloom-3b", "bloom-7b",
                             "bloom"])
parser.add_argument('--model_name_sent', default='t5-xl', type=str,
                    choices=["t5-xl", "t5-xxl", "all-mpnet-base-v2", "gtr-t5-xxl", "gtr-t5-xl", "multilingual"],
                    help='model for context retrieval via sentence similarity')

def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


def get_repo_root(model_name_or_path, revision=None):
    # checks if online or not
    if is_offline_mode():

        print_rank0("Offline mode: forcing local_files_only=True")
        local_files_only = True
    else:
        local_files_only = False

    # loads files from hub
    cached_repo_dir = snapshot_download(
        model_name_or_path, allow_patterns=["*"], local_files_only=local_files_only, revision=revision
    )

    return cached_repo_dir


def get_checkpoint_files(model_name_or_path, revision=None):
    # checks if online or not
    if is_offline_mode():
        print_rank0("Offline mode: forcing local_files_only=True")
        local_files_only = True
    else:
        local_files_only = False

    # loads files from hub
    cached_repo_dir = snapshot_download(
        model_name_or_path, allow_patterns=["*"], local_files_only=local_files_only, revision=revision
    )

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


def write_checkponts_json():
    with io.open(checkpoints_json, "w", encoding="utf-8") as f:
        # checkpoint_files = glob.glob(f"{checkpoint_dir}/*bin")
        checkpoint_files = get_checkpoint_files(model_name)

        # print("Checkpoint files:", checkpoint_files)

        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}

        json.dump(data, f)


def run_generation():
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=0)
    rtpt = RTPT(name_initials='FF', experiment_name='RiT_generate', max_iterations=len(dataloader))
    rtpt.start()


    print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False, min_length=4, pad_token_id=tokenizer.eos_token_id)

    print_rank0(f"Generate args {generate_kwargs}")

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
        df.to_csv(pth + f'/results_baseline_lang={args.lang}_bloom.csv', sep='\t')
    else:
        if args.simulate:
            df.to_csv(
                pth + f'/results_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_simulate{args.simulate}_bloom.csv',
                sep='\t')
        else:
            df.to_csv(
                pth + f'/results_{args.type}_knn{args.nn}_thresh{args.thresh}_lang={args.lang}_bloom.csv',
                sep='\t')


if __name__ == "__main__":
    args = parser.parse_args()

    tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

    num_tokens = 8

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()

    model_name = args.name
    infer_dtype = args.dtype

    tp_presharded_mode = True if model_name in tp_presharded_models else False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    kernel_inject = True

    dtype = torch.float16

    with deepspeed.OnDevice(dtype=dtype, device="meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    model = model.eval()

    checkpoints_json = "checkpoints.json"

    if kernel_inject:
        kwargs = dict(replace_with_kernel_inject=True)
    else:
        kwargs = dict(injection_policy={BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")})

    repo_root = get_repo_root(model_name)
    if tp_presharded_mode:
        checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
    else:
        if rank == 0:
            write_checkponts_json()
        dist.barrier()

    # checkpoints_json=None
    model = deepspeed.init_inference(
        model,
        mp_size=world_size,
        base_dir=repo_root,
        dtype=getattr(torch, infer_dtype),
        checkpoint=checkpoints_json,
        **kwargs,
    )

    model = model.module

    fname = f'/workspace/repositories/norms/crosslingual/{args.version}/retriever_processed_norms_cl'
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

    if args.unprocessed:
        model_names = ["sentence-transformers/stsb-xlm-r-multilingual"]
        model_name = model_names[0]
        model_name = 'models/' + model_name
        model_sentence = SentenceTransformer(model_name)
        model_sentence.cuda()
    else:
        model_sentence = ''

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
