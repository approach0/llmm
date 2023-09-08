import requests


def search(uri, question, keywords, docid=0,
    full_topk=30, topk=3, no_mapper=False):

    json = {
        'docid': docid,
        'topk': full_topk
    }

    if question is not None:
        json['question'] = question
    if keywords is not None:
        json['keywords'] = keywords

    url='http://tuna.cs.uwaterloo.ca:8080/' + uri
    res = requests.post(url, json=json)
    if res.ok:
        res = res.json()
        res = res[:topk]
        def mapper(item):
            content, post_id, score = item
            if uri == 'mabowdor':
                url = ('https://math.stackexchange.com/' +
                    f'questions/{post_id}')
            else:
                url = post_id
            return f'URL: {url}\n\n' + content
        if no_mapper:
            return res
        else:
            return list(map(mapper, res))
    else:
        return []


if __name__ == '__main__':
    # res = search('mabowdor', 'How to evaluate $e^i$?', ['e^i'])
    # print(res)
    # res = search('MATH', 'How to evaluate $e^i$?', ['e^i'])
    # print(res)
    res = search('dups', None, ['$\\frac{1}{2}$', '$x^2$', 'projection', 'formula'], docid=1096, no_mapper=True)
    print(res)
