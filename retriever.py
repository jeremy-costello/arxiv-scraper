from sentence_transformers import SentenceTransformer

from indexer import get_index

# QUERY
input_query = 'What is a Transformer?'
top_k_hits = 4

# SQL inputs
text_column = 'abstract'

# S-BERT inputs
model_name = 'multi-qa-MiniLM-L6-cos-v1'
embedding_path = f'{text_column}__{model_name}.pkl'

# hnswlib inputs
index_path = './hnswlib.index'

index, arxiv_id_list = get_index(embedding_path, index_path)

model = SentenceTransformer(model_name)
query_embedding = model.encode(input_query)

corpus_ids, distances = index.knn_query(query_embedding, k=top_k_hits)

hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
hits = sorted(hits, key=lambda x: x['score'], reverse=True)

for hit in hits[:top_k_hits]:
    print(f"score: {round(hit['score'], 3)} | arxiv id: {arxiv_id_list[hit['corpus_id']]}")