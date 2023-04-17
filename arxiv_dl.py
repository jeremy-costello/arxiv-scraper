import os
import shutil
import requests
from datetime import datetime

from bs4 import BeautifulSoup

try:
    from PyPDF2 import PdfReader
    HAVE_PDF_READER = True
except ImportError:
    HAVE_PDF_READER = False


def main():
    paper_id = '2110.12764'
    get_full_text = True

    return_dict = get_paper_dict(paper_id, get_full_text)
    print(return_dict)


def get_paper_dict(paper_id, get_full_text):
    return_dict = dict()
    get_full_text = get_full_text and HAVE_PDF_READER

    url = 'https://export.arxiv.org'

    abs_url = f'{url}/abs/{paper_id}'

    r = requests.get(abs_url)
    html = r.content
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('h1', {'class': 'title'}).text.lstrip('Title:').strip()

    authors_dict = {
        'authors': []
    }
    authors = soup.find('div', {'class': 'authors'})

    for a in authors.find_all('a'):
        author_dict = dict()
        author_dict['name'] = a.text
        author_dict['id'] = a['href']
        authors_dict['authors'].append(author_dict)

    date_string = soup.find('div', {'class': 'dateline'}).text[1:-1].lstrip('Submitted on').strip()
    date_full = datetime.strptime(date_string, '%d %b %Y')
    date = datetime.strftime(date_full, '%Y-%m-%d')

    abstract = soup.find('blockquote', {'class': 'abstract'}).text.strip().lstrip('Abstract:').strip()

    return_dict['title'] = title
    return_dict['authors'] = authors_dict
    return_dict['date'] = date
    return_dict['abstract'] = abstract

    if get_full_text:
        pdf_url = f'{url}/pdf/{paper_id}'
        pdf_name = f'{paper_id}.pdf'

        with requests.get(pdf_url, stream=True) as r:
            with open(pdf_name, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        pdf_text_dict = dict()
        reader = PdfReader(pdf_name)
        for num, page in enumerate(reader.pages):
            pdf_text_dict[num] = page.extract_text()

        os.remove(pdf_name)

        return_dict['full_text'] = pdf_text_dict
    else:
        return_dict['full_text'] = None

    # semantic scholar
    search_url = 'https://api.semanticscholar.org/graph/v1/paper/search'

    search_params = {
        'query': title.lower()
    }

    search = requests.get(search_url, params=search_params)
    search_json = search.json()
    if 'data' in search_json.keys():
        paper_data = search.json()['data'][0]
        paper_title = paper_data['title']
        if title.lower().strip() == paper_title.lower().strip():
            paper_id = paper_data['paperId']
        else:
            paper_id = None
    else:
        paper_id = None

    return_dict['ss_id'] = paper_id

    return return_dict


if __name__ == "__main__":
    main()
