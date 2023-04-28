import os
import pickle
import sqlite3

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer


'''
- remove from lists and embeddings where model was changed
'''


# SQL inputs
database_name = 'arxiv_database.db'
table_name = 'arxiv_papers'
text_column = 'abstract'

# S-BERT inputs
model_name = 'sentence-transformers/allenai-specter'
embedding_path = f"{text_column}__{model_name.replace('/', '_')}.pkl"
max_seq_length = None
sep_token = '[SEP]'

# hnswlib inputs
index_path = './hnswlib.index'
hnsw_ef_construction = 400
hnsw_M = 64


conn = sqlite3.connect(database_name)

c = conn.cursor()

c.execute(f"SELECT arxiv_id, title, {text_column} FROM {table_name} WHERE {text_column}_embedding_model IS NULL OR {text_column}_embedding_model != ?", (model_name,))

rows = c.fetchall()

new_arxiv_id_list = [row[0] for row in rows]
title_list = [row[1] for row in rows]
text_list = [row[2] for row in rows]

if new_arxiv_id_list and title_list and text_list:
    # way to make sure there isn't a mix of these two in the SQL table
    if text_column == 'abstract':
        new_corpus_list = [title + sep_token + text for title, text in zip(title_list, text_list)]
    elif text_column == 'full_text':
        new_corpus_list = text_list
    else:
        raise ValueError('Invalid text column!')

    model = SentenceTransformer(model_name)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    new_corpus_embeddings = model.encode(new_corpus_list, show_progress_bar=True, convert_to_numpy=True)

    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        old_arxiv_id_list = cache_data['arxiv_ids']
        keep_arxiv_id_list = [x for x in old_arxiv_id_list if x not in new_arxiv_id_list]
        removed_indices = [i for i, x in enumerate(old_arxiv_id_list) if x in new_arxiv_id_list]

        arxiv_id_list = keep_arxiv_id_list.extend(new_arxiv_id_list)

        old_corpus_list = cache_data['corpus_list']
        keep_corpus_list = np.delete(old_corpus_list, removed_indices).tolist()

        corpus_list = keep_corpus_list.extend(new_corpus_list)
        
        old_corpus_embeddings = cache_data['embeddings']
        keep_corpus_embeddings = np.delete(old_corpus_embeddings, removed_indices, axis=0)

        corpus_embeddings = np.vstack((keep_corpus_embeddings, new_corpus_embeddings))

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

    c.execute(f"UPDATE {table_name} SET {text_column}_embedding_model = ? WHERE {text_column}_embedding_model IS NULL OR {text_column}_embedding_model != ?",
                (model_name, model_name))

    conn.commit()

conn.close()
