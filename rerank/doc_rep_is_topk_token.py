import torch
from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import os
import pytrec_eval
from collections import OrderedDict, defaultdict
from statistics import mean
import json
import sys
import numpy as np
import torch.nn.functional as F
import random

def quick_sort_iterative(arr):
    stack = [(0, len(arr) - 1)]
    result = arr[:]
    
    while stack:
        start, end = stack.pop()
        if start >= end:
            continue
        pivot = partition(result, start, end)
        stack.append((start, pivot - 1))
        stack.append((pivot + 1, end))
    
    return result

def partition(array, start, end):
    pivot = array[(start + end) // 2]
    i = start - 1
    j = end + 1
    while True:
        i += 1
        while array[i] < pivot:
            i += 1
        j -= 1
        while array[j] > pivot:
            j -= 1
        if i >= j:
            return j
        array[i], array[j] = array[j], array[i]

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[random.randint(0, len(arr) - 1)]
        left = [x for x in arr if x[0] > pivot[0]]
        middle = [x for x in arr if x[0] == pivot[0]]
        right = [x for x in arr if x[0] < pivot[0]]
        return quick_sort(left) + middle + quick_sort(right)

def generate_token_scores(model, tokenizer, text, max_length=20, top_k=32000, top_n=150):
    input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=1,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    torch.cuda.empty_cache()
    scores = outputs.scores
    max_scores_map = {}

    for score_distribution in scores:
        normalized_scores = F.softmax(score_distribution, dim=-1)
        topk_scores, topk_indices = torch.topk(normalized_scores, k=top_k, dim=-1)
        
        for idx, score in zip(topk_indices.squeeze(), topk_scores.squeeze()):
            token_id = idx.item()
            score = score.item()
            if score > 0 and (token_id not in max_scores_map or score > max_scores_map[token_id]):
                max_scores_map[token_id] = score
    
    return max_scores_map

def generate_token_scores_non_autoregresive(model, tokenizer, text, max_length=10, top_k=32000):
    input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    max_scores_map = {}

    decoder_input_ids = torch.full((1, 1), tokenizer.pad_token_id, dtype=torch.long, device='cuda')

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

            logits = outputs.logits[:, -1, :]
            normalized_scores = F.softmax(logits, dim=-1)
            topk_scores, topk_indices = torch.topk(normalized_scores, k=top_k, dim=-1)

            for idx, score in zip(topk_indices.squeeze(), topk_scores.squeeze()):
                token_id = idx.item()
                score = score.item()
                if score > 0 and (token_id not in max_scores_map or score > max_scores_map[token_id]):
                    max_scores_map[token_id] = score

    torch.cuda.empty_cache()

    return max_scores_map

def get_top_token_ids(max_scores_map, top_n=150):
    score_list = [(score, token_id) for token_id, score in max_scores_map.items() if score > 0]
    sorted_scores = quick_sort(score_list)
    top_token_ids = [token_id for score, token_id in sorted_scores[:top_n]]
    
    return top_token_ids

def get_top_tokens(max_scores_map, top_n=50):
    score_list = [(score, token_id) for token_id, score in max_scores_map.items() if score > 0]
    sorted_scores = quick_sort(score_list)
    top_tokens = {token_id: score for score, token_id in sorted_scores[:top_n]}
    
    return top_tokens

def filter_qtext_token_scores(max_scores_map, tokenizer, qtext):
    input_ids = tokenizer.encode(qtext, add_special_tokens=False)
    return {token_id: max_scores_map.get(token_id, 0) for token_id in input_ids}

model_type_or_dir = sys.argv[1] #"castorini/doc2query-t5-base-msmarco"

topk = open(sys.argv[2]) #open("index-sqhd-54-refcorrect-top100.trec.tsv")

VALIDATION_METRIC = 'recip_rank'   #'recip_rank' #'ndcg_cut_10' 

qrel_file = sys.argv[3] #"dev_qrels.tsv"

qrels = defaultdict(dict)
with open(qrel_file) as f:
    for line in f:
        try:
            qid, _, did, rel = line.strip().split("\t")
        except:
            qid, _, did, rel = line.strip().split(" ")
        if int(rel) > 0:
            qrels[qid][did] = int(rel)

model = T5ForConditionalGeneration.from_pretrained(model_type_or_dir).to("cuda").half()
model.eval()
tokenizer = T5Tokenizer.from_pretrained(model_type_or_dir)
torch.manual_seed(42)

run = defaultdict(dict)

fo = open(sys.argv[4], 'w')

cur_qid = None
cur_qtext = None

dtexts = []
dids = []
qids = []
num=100
bsize=16
read_query = {}

for idx, line in enumerate(tqdm(topk)):
    qid, didt, qtext, dtext = line.strip().split("\t")

    qid_int = int(qid)

    if qid_int not in read_query:
        query_ids = tokenizer.encode(qtext, add_special_tokens=False)
        read_query[qid_int] = query_ids

    if (idx + 1) % num == 0:
        with torch.no_grad():
            all_scores = []
            i = 0
            for dtext in dtexts:
                max_scores_map = generate_token_scores(model, tokenizer, dtext, max_length=20, top_k=32000)
                doc_ids = get_top_token_ids(max_scores_map, top_n=150)

                if len(qrels[qid]) == 0:
                    continue

                score = 0
                for k in read_query[int(qid)]:
                    if k in doc_ids:
                        score += max_scores_map.get(k, 0) ** 2
                all_scores.append(score)

        for qid, did, score in zip(qids, dids, all_scores):
            fo.write(f"{qid}\t{did}\t{score}\n")
            fo.flush()
            run[qid][did] = score

        dtexts = []
        dids = []
        qids = []
        
    qids.append(qid)
    dtexts.append(dtext)
    dids.append(didt)

torch.cuda.empty_cache()                         
fo.close()
topk.close()

for VALIDATION_METRIC in ['recip_rank','ndcg_cut_10', 'ndcg_cut_20', 'P_20']:
    for top_k in [5,10,20,100]:
        top_run = defaultdict(dict)
        for q in run:
            docs = sorted(run[q].items(), key=lambda x: -x[1])
            for item in docs[:top_k]:
                top_run[q][item[0]] = item[1]
        trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
        eval_scores = trec_eval.evaluate(top_run)
        print(VALIDATION_METRIC, top_k, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))