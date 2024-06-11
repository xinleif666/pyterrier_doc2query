import pytrec_eval
from statistics import mean
from collections import defaultdict

run = defaultdict(dict)
with open('modify0_non.tsv', 'r') as f:
# with open('test_quicksort_query_add_top50_1127.tsv', 'r') as f:
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

'''
result_full_ori
recip_rank 5 0.30888968481375356
recip_rank 10 0.2580874494019193
recip_rank 20 0.1837278468535288
recip_rank 100 0.050449305837780324
ndcg_cut_10 5 0.36826857093848087
ndcg_cut_10 10 0.3552193766497411
ndcg_cut_10 20 0.2904480618889735
ndcg_cut_10 100 0.02193953383392664
ndcg_cut_20 5 0.36826857093848087
ndcg_cut_20 10 0.3552193766497411
ndcg_cut_20 20 0.3161615544149717
ndcg_cut_20 100 0.11386934734897264
P_20 5 0.029183381088825216
P_20 10 0.035193409742120346
P_20 20 0.04042263610315187
P_20 100 0.022915472779369627
'''