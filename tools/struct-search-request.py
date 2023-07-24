import json
import pickle
import requests


url='http://tuna.cs.uwaterloo.ca:8080/struct_search'
corpus_lookup='data/arqmath3-ColBERT-docdict-only/docdict.pkl'

query = [
    {'str': r'\tan x + \sec x = 2 \cos x.', 'type': 'tex'},
]

print('loading corpus lookup pickle ...')
with open(corpus_lookup, 'rb') as fh:
    corpus = pickle.load(fh)

res = requests.post(url, json={
    'topk': 3,
    'query': query
})
if res.ok:
    j = json.loads(res.json())
    if j['ret_code'] != 0:
        print(j)
        quit(1)
    for hit in j['hits']:
        docid = hit['docid']
        answer_url = hit['field_url']
        answer_id = hit['field_title']
        snippet = hit['field_content']
        document = corpus[answer_id]
        d = document[1]
        d = d.replace(r'[imath]', '$')
        d = d.replace(r'[/imath]', '$')

        print('-' * 50)
        print(d)
