import sys
sys.path.insert(0, './pya0')

import pya0
from pya0.index_manager import from_prebuilt_index

import os
import json
from flask import Flask, request, jsonify


prebuilt_index = 'arqmath-task1'

index_path = from_prebuilt_index(prebuilt_index)
ix = pya0.index_open(index_path, option="r")
if ix is None:
    print('ERR: Cannot open index!')
    quit()


app = Flask('struct_search')

@app.route('/struct_search', methods=['GET', 'POST'])
def server_handler():
    j = request.json
    if 'topk' not in j or 'query' not in j:
        return jsonify({'error': 'no topk or query key!'})
    topk = j['topk']
    print(f'Searching (topk={topk}) ...')
    JSON = pya0.search(ix, j['query'], verbose=False, topk=topk)
    results = json.loads(JSON)
    print(json.dumps(results, indent=4))
    return jsonify(JSON)


if __name__ == '__main__':
  app.run(debug=True, port=8080, host="0.0.0.0")
