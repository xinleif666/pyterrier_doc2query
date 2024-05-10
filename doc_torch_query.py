import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
new_tokens = ['autonomous', 'Manhattan', 'manhattan', 'seinhardt', 'produses', 'seinmann']
num_added_tokens = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

doc_text = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."
input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
outputs = model.generate(
    input_ids=input_ids,
    max_length=64,
    do_sample=True,
    top_k=32000,
    num_return_sequences=1,
    output_scores=True,
    return_dict_in_generate=True
)

sequences = outputs.sequences
scores = outputs.scores

generated_query = tokenizer.decode(sequences[0], skip_special_tokens=True)

result = []

for i, score_distribution in enumerate(scores):
    topk_scores, topk_indices = torch.topk(score_distribution, k=32000, dim=-1)
    topk_indices_list = topk_indices.squeeze().tolist()
    topk_scores_list = topk_scores.squeeze().tolist()

    selected_token = tokenizer.convert_ids_to_tokens([sequences[0][i + 1]])[0]
    print(f"Selected Token: {selected_token}")

    candidates = [
        {"token": tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens([idx])[0]]), "score": score}
        for idx, score in zip(topk_indices_list, topk_scores_list)
    ]

    max_score = max(topk_scores_list)
    total_score = sum(topk_scores_list)
    average_score = total_score / len(topk_scores_list)

    result.append({
        "position": i,
        "selected_token": tokenizer.convert_tokens_to_string([selected_token]),
        "candidates": candidates,
        "statistics": {
            "max_score": max_score,
            "average_score": average_score,
            "total_score": total_score
        }
    })

print(f"Generated Query: {generated_query}\n")
for res in result:
    print(f"Position {res['position'] + 1}: Selected Token = '{res['selected_token']}'")
    print(f"  Max Score: {res['statistics']['max_score']:.4f}")
    print(f"  Average Score: {res['statistics']['average_score']:.4f}")
    print(f"  Total Score: {res['statistics']['total_score']:.4f}")
    for candidate in res['candidates']:
        print(f"  Candidate Token = '{candidate['token']}', Score = {candidate['score']:.4f}")
    print()
