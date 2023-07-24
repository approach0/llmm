import requests
url='http://tuna.cs.uwaterloo.ca:8080/struct_search'
query = [
    {'str': r'\tan x + \sec x = 2 \cos x.', 'type': 'tex'},
]

res = requests.post(url, json={
    'topk': 3,
    'query': query
})
if res.ok:
    print(res.json())
