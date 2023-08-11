import requests
from requests.utils import requote_uri


def replace_imath(d):
    d = d.replace(r'[imath]', '$')
    d = d.replace(r'[/imath]', '$')
    d = d.replace(r'</em>', '')
    d = d.replace(r'<em class="hl">', '')
    return d


def search_api(keywords=['$x+y=xy$', 'why'], topk=3):
    keywords = list(map(lambda x: 'OR content:' + x, keywords))

    response = requests.get('https://approach0.xyz/search-relay/', params={
        'p': 1,
        'q': ','.join(keywords)
    })

    data = response.json()
    if data['ret_code'] == 0:
        hits = data['hits'][:topk]
        results = []
        for hit in hits:
            url = hit['field_url']
            title = replace_imath(hit['field_title'])
            content = replace_imath(hit['field_content'])
            result = url + '\n\n' + title + '\n\n' + content
            results.append(result)
        return results
    else:
        return f'Error: {data["ret_str"]}'


if __name__ == '__main__':
    keywords = [r'$(\sin x)^7 = a \sin 7x + b \sin 5x + c \sin 3x + d \sin x$']
    results = search_api(keywords, topk=4)
    for res in results:
        print(res)
        print('\n\n')
