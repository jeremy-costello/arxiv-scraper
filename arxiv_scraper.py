import gc
import json
import sqlite3

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from arxiv_dl import get_paper_dict


def main():
    # SQL inputs
    database_name = 'arxiv_database.db'
    table_name = 'arxiv_papers'

    # arxiv inputs
    max_results = 10000
    start_date = '2023-01-01'
    end_date = '2023-04-30'
    categories = {
        'cs': ['AI', 'CL', 'CV', 'LG', 'NE'],
        'stat': ['ML']
    }

    # other inputs
    get_full_text = False

    conn = sqlite3.connect(database_name)

    c = conn.cursor()

    c.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    )
    result = c.fetchone()

    if result is None:
        c.execute(
            f'''
                CREATE TABLE {table_name}
                    (
                        arxiv_id text,
                        categories text,
                        title text,
                        authors text,
                        submission_date text,
                        revision_date text NULL,
                        abstract text,
                        abstract_embedding_model text NULL,
                        full_text text NULL,
                        full_text_embedding_model text NULL,
                        ss_id text NULL
                    )
            '''
        )

    api_url = 'https://export.arxiv.org/api/query'
    date_range = f"{start_date.replace('-', '')}000000 TO {end_date.replace('-', '')}235959"

    c.execute(f"SELECT arxiv_id FROM {table_name}")
    rows = c.fetchall()
    arxiv_id_list = [row[0] for row in rows]
    arxiv_id_set = set(arxiv_id_list)
    assert len(arxiv_id_list) == len(arxiv_id_set)

    for cat, subcat_list in categories.items():
        for subcat in subcat_list:
            print(f'Category: {cat}.{subcat}')

            search_query = f"submittedDate:[{date_range}] AND cat:{cat}.{subcat}"
            api_params = {
                'search_query': search_query,
                'start': '0',
                'max_results': f'{str(max_results)}',
                'sortBy': 'submittedDate',
                'sortOrder': 'ascending',
            }

            r = requests.get(api_url, params=api_params)

            soup = BeautifulSoup(r.content, 'xml')

            for entry in tqdm(soup.find_all('entry')):
                link = entry.find('link', {'type': 'text/html'})
                paper_id = link['href'].split('/')[-1].split('v')[0]

                if paper_id not in arxiv_id_set:
                    categories_dict = {
                        'categories': []
                    }
                    categories = entry.find_all('category')
                    for category in categories:
                        categories_dict['categories'].append(category['term'])
                    
                    data_dict = get_paper_dict(paper_id, get_full_text)

                    full_text = data_dict['full_text']
                    if full_text is not None:
                        full_text = json.dumps(full_text)

                    c.execute(
                        f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            paper_id,
                            json.dumps(categories_dict),
                            data_dict['title'],
                            json.dumps(data_dict['authors']),
                            data_dict['date_submitted'],
                            data_dict['date_revised'],
                            data_dict['abstract'],
                            None,
                            full_text,
                            None,
                            data_dict['ss_id']
                        )
                    )

                    conn.commit()

    conn.close()


if __name__ == '__main__':
    main()
