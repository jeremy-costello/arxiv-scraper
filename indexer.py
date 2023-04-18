import os
import pickle
import hnswlib


def get_index(embedding_path, index_path):
    with open(embedding_path, 'rb') as f:
        cache_data = pickle.load(f)

    arxiv_id_list = cache_data['arxiv_ids']
    corpus_list = cache_data['corpus_list']
    corpus_embeddings = cache_data['embeddings']

    index = hnswlib.Index(space='cosine', dim=corpus_embeddings.shape[1])

    if os.path.exists(index_path):
        index.load_index(index_path)
    else:
        index.init_index(max_elements=corpus_embeddings.shape[0], ef_construction=400, M=64)
        index.add_items(corpus_embeddings, list(range(corpus_embeddings.shape[0])))
        index.save_index(index_path)

    return index, arxiv_id_list, corpus_list
