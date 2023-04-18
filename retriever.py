import pickle
import hnswlib

from sentence_transformers import SentenceTransformer, CrossEncoder


def main():
    # QUERY
    input_query = 'What is a Transformer?'
    top_k_hits = 8
    rerank = True

    # SQL inputs
    text_column = 'abstract'

    # S-BERT inputs
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    embedding_path = f'{text_column}__{model_name}.pkl'
    cross_encoder_name = 'ms-marco-MiniLM-L-6-v2'

    # hnswlib inputs
    index_path = './hnswlib.index'


    index, arxiv_id_list, corpus_list = get_index(embedding_path, index_path, rerank)

    index.set_ef(50)  # ef should always be > top_k_hits

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(input_query)

    corpus_ids, distances = index.knn_query(query_embedding, k=top_k_hits)

    hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    if rerank:
        sort_score = 'cross-score'
        cross_encoder_input = [[input_query, corpus_list[hit['corpus_id']]] for hit in hits]
        cross_scores = get_reranking(cross_encoder_input, cross_encoder_name)
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]
    else:
        sort_score = 'score'

    hits = sorted(hits, key=lambda x: x[sort_score], reverse=True)
    for hit in hits[:top_k_hits]:
        print(f"score: {round(hit[sort_score], 3)} | arxiv id: {arxiv_id_list[hit['corpus_id']]}")


def get_reranking(cross_encoder_input, cross_encoder_name):
    cross_encoder = CrossEncoder(f'cross-encoder/{cross_encoder_name}')

    cross_scores = cross_encoder.predict(cross_encoder_input)
    return cross_scores


def get_index(embedding_path, index_path, rerank):
    with open(embedding_path, 'rb') as f:
        cache_data = pickle.load(f)

    arxiv_id_list = cache_data['arxiv_ids']
    corpus_embeddings = cache_data['embeddings']
    if rerank:
        corpus_list = cache_data['corpus_list']
    else:
        corpus_list = None

    index = hnswlib.Index(space='cosine', dim=corpus_embeddings.shape[1])
    index.load_index(index_path)

    return index, arxiv_id_list, corpus_list


if __name__ == "__main__":
    main()
