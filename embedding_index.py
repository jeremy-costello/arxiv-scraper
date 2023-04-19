import os
import pickle
import sqlite3

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer


# SQL inputs
database_name = 'arxiv_database.db'
table_name = 'arxiv_papers'
text_column = 'abstract'

# S-BERT inputs
model_name = 'sentence-transformers/allenai-specter'
embedding_path = f'{text_column}__{model_name}.pkl'
max_seq_length = 512

# hnswlib inputs
index_path = './hnswlib.index'
hnsw_ef_construction = 400
hnsw_M = 64


conn = sqlite3.connect(database_name)

cursor = conn.execute(f"SELECT arxiv_id, {text_column} FROM {table_name} WHERE {text_column}_embedding_model IS NULL OR {text_column}_embedding_model != ?", (model_name,))

rows = cursor.fetchall()

arxiv_id_list = [row[0] for row in rows]
corpus_list = [row[1] for row in rows]

if arxiv_id_list and corpus_list:
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    corpus_embeddings = model.encode(corpus_list, show_progress_bar=True, convert_to_numpy=True)

    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            cache_data = pickle.load(f)
                
        arxiv_id_list = cache_data['arxiv_ids'].extend(arxiv_id_list)
        corpus_embeddings = np.vstack((cache_data['embeddings'], corpus_embeddings))

    embedding_dict = {
        'arxiv_ids': arxiv_id_list,
        'corpus_list': corpus_list,
        'embeddings': corpus_embeddings
    }

    with open(embedding_path, 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    index = hnswlib.Index(space='cosine', dim=corpus_embeddings.shape[1])
    index.init_index(max_elements=corpus_embeddings.shape[0],
                     ef_construction=hnsw_ef_construction,
                     M=hnsw_M)
    index.add_items(corpus_embeddings,
                    list(range(corpus_embeddings.shape[0])))
    index.save_index(index_path)

    conn.execute(f"UPDATE {table_name} SET {text_column}_embedding_model = ? WHERE {text_column}_embedding_model IS NULL OR {text_column}_embedding_model != ?",
                (model_name, model_name))

    conn.commit()

conn.close()
