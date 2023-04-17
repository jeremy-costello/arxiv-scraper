# imports
import json
import sqlite3
import requests
from bs4 import BeautifulSoup
from arxiv_dl import get_paper_dict


def main():
    # SQL inputs
    database_name = 'arxiv_database.db'
    table_name = 'arxiv_papers'

    # arxiv inputs
    max_results = 10
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    categories = {
        'cs': ['AI']
    }

    # other inputs
    get_full_text = True

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
                        date text,
                        abstract text,
                        full_text text NULL,
                        ss_id text NULL
                    )
            '''
        )

    api_url = 'https://export.arxiv.org/api/query'

    for cat, subcat_list in categories.items():
        for subcat in subcat_list:
            api_params = {
                'search_query': f'cat:{cat}.{subcat}',
                'start': '0',
                'max_results': f'{str(max_results)}',
                'sortBy': 'submittedDate',
                'sortOrder': 'descending',
                'submittedDate': f'{start_date},{end_date}'
            }

            r = requests.get(api_url, params=api_params)

            soup = BeautifulSoup(r.content, 'xml')

            for entry in soup.find_all('entry'):
                categories_dict = {
                    'categories': []
                }
                categories = entry.find_all('category')
                for category in categories:
                    categories_dict['categories'].append(category['term'])

                link = entry.find('link', {'type': 'text/html'})
                paper_id = link['href'].split('/')[-1].split('v')[0]

                c.execute(f"SELECT arxiv_id FROM {table_name} WHERE arxiv_id=?", (paper_id,))
                result = c.fetchone()

                if result is None:
                    data_dict = get_paper_dict(paper_id, get_full_text)

                    full_text = data_dict['full_text']
                    if full_text is not None:
                        full_text = json.dumps(full_text)

                    c.execute(
                        f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            paper_id,
                            json.dumps(categories_dict),
                            data_dict['title'],
                            json.dumps(data_dict['authors']),
                            data_dict['date'],
                            data_dict['abstract'],
                            full_text,
                            data_dict['ss_id']
                        )
                    )

                    conn.commit()

    conn.close()


if __name__ == '__main__':
    main()
