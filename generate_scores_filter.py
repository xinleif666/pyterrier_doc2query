import pyterrier as pt
import pandas as pd
from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
from pyterrier_doc2query import Doc2QueryStore, QueryScoreStore, QueryFilter, QueryScorer, Doc2Query
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_dr import ElectraScorer
from pyterrier_dr import TctColBert

import json
import re

def parse_contents(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    passages = []
    all_queries = []

    query_start_pattern = re.compile(r'\.\s+(?=What|Why|Who|Where|When|Which|How)', flags=re.IGNORECASE)
    query_split_pattern = re.compile(r'\s+(?=(?:What|Why|Who|Where|When|Which|How)\s)', flags=re.IGNORECASE)

    for entry in data["entries"]:
        contents = entry["contents"]

        first_query_match = re.search(query_start_pattern, contents)
        if first_query_match:
            passage_text = contents[:first_query_match.start()+1].strip()
            queries_text = contents[first_query_match.start()+2:].strip()
        else:
            passage_text = contents
            queries_text = ''

        queries_list = []
        if queries_text:
            potential_queries = re.split(query_split_pattern, queries_text)
            for query in potential_queries:
                query = query.strip()
                if not query.endswith('?'):
                    query += '?'
                queries_list.append(query)

        # Ensure unique queries only
        unique_queries = list(dict.fromkeys(queries_list))
        passages.append((entry["id"], passage_text))
        all_queries.append(unique_queries)

    return passages, all_queries

def score_queries_with_t5(passages, all_queries):
    # Initialize MonoT5 scorer
    scorer = MonoT5ReRanker()

    for (pid, passage), queries in zip(passages, all_queries):
        print(f'Passage: {passage}')
        for query in queries:
            df = pd.DataFrame([{'qid': pid, 'query': query, 'docno': '0', 'text': passage}])

            res = scorer(df)
            print(f'Query: {query}, Score: {res.iloc[0]["score"]}')

def score_queries_with_t5_and_filter(passages, all_queries, threshold=-0.05):
    scorer = MonoT5ReRanker()
    
    filtered_queries = []

    for (pid, passage), queries in zip(passages, all_queries):
        print(f'Passage: {passage}')
        
        scores = []
        
        for query in queries:
            df = pd.DataFrame([{'qid': pid, 'query': query, 'docno': '0', 'text': passage}])
            res = scorer(df)
            score = res.iloc[0]["score"]
            scores.append((query, score))
            print(f'Query: {query}, Score: {score}')
        
        filtered_queries_for_passage = [query for query, score in scores if score > threshold]
        filtered_queries.append(filtered_queries_for_passage)
    
    return filtered_queries

def score_queries_with_electra_and_filter(passages, all_queries, threshold=1):
    scorer = ElectraScorer('crystina-z/monoELECTRA_LCE_nneg31')

    filtered_queries = []

    for (pid, passage), queries in zip(passages, all_queries):
        print(f'Passage: {passage}')
        
        scores = []
        scores1 = []

        for query in queries:
            df = pd.DataFrame([{'qid': pid, 'query': query, 'docno': '0', 'text': passage}])
            res = scorer(df)
            score = res.iloc[0]["score"]
            scores.append((query, score))

            print(f'Query: {query}, Score: {score}')
        
        filtered_queries_for_passage = [query for query, score in scores if score > threshold]
        filtered_queries.append(filtered_queries_for_passage)
    
    return filtered_queries

def expand_passage_with_filtered_queries(passages, all_queries, threshold=-0.05):
    scorer = MonoT5ReRanker()
    
    expanded_passages = []

    for (pid, passage_text), queries in zip(passages, all_queries):
        print(f'Processing Passage ID: {pid}')
        
        scores = []
        for query in queries:
            df = pd.DataFrame([{'qid': pid, 'query': query, 'docno': '0', 'text': passage_text}])
            res = scorer(df)
            score = res.iloc[0]["score"]
            scores.append((query, score))
        
        filtered_queries_for_passage = [query for query, score in scores if score > threshold]

        print(f"Filtered Queries: {filtered_queries_for_passage}")

        if filtered_queries_for_passage:
            expanded_text = passage_text + " " + " ".join(filtered_queries_for_passage)
        else:
            expanded_text = passage_text  # No queries passed the threshold

        expanded_passages.append(expanded_text)  # Storing only the expanded text
    
    return expanded_passages


def add_comma_to_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        lines = input_file.readlines()
        total_lines = len(lines)
        for i, line in enumerate(lines):
            if i < total_lines - 1:
                if not line.endswith(',\n') and not line.endswith(','):
                    output_file.write(line.rstrip('\n') + ',\n')
                else:
                    output_file.write(line)
            else:
                output_file.write(line)

if __name__ == "__main__":
    file_path = './test_docs00.json'
    # file_path = './docs00.json'
    # run this function only when we want to modify
    # the structure of input docsxx.json files
    # output_file_path = './modified_docs00.json'
    # add_comma_to_file(file_path, output_file_path)

    passages, all_queries = parse_contents(file_path)
    threshold = 1

    # score_queries_with_t5(passages, all_queries)
    # filtered_queries = score_queries_with_t5_and_filter(passages, all_queries, threshold)
    filtered_queries = score_queries_with_electra_and_filter(passages, all_queries, threshold)
    expanded_passages = expand_passage_with_filtered_queries(passages, all_queries, threshold)
    for (pid, passage), queries in zip(passages, filtered_queries):
        print(f'Show_passage: {passage}')
        print('Filtered queries based on score threshold:')
        for query in queries:
            print(query)
    for passage in expanded_passages:
        print(passage)