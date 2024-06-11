import pytrec_eval
from statistics import mean
from collections import defaultdict

run = defaultdict(dict)
with open('result.tsv', 'r') as f: # Put your file path here
    for line_number, line in enumerate(f, 1):
        try:
            qid, did, score = line.strip().split('\t')
            run[qid][did] = float(score)
        except ValueError:
            print(f"Skipping malformed line {line_number}: {line.strip()}")

qrels = defaultdict(dict)
with open('dev_qrels.tsv', 'r') as f:
    for line in f:
        qid, _, did, rel = line.strip().split('\t')
        qrels[qid][did] = int(rel)

for VALIDATION_METRIC in ['recip_rank', 'ndcg_cut_10', 'ndcg_cut_20', 'P_20']:
    for top_k in [5, 10, 20, 100]:
        top_run = defaultdict(dict)
        for q in run:
            docs = sorted(run[q].items(), key=lambda x: -x[1])
            for item in docs[:top_k]:
                top_run[q][item[0]] = item[1]
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
        eval_scores = evaluator.evaluate(top_run)
        print(VALIDATION_METRIC, top_k, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))