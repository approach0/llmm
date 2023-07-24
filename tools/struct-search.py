import os
import json
import sys
sys.path.insert(0, './pya0')
from pya0.index_manager import from_prebuilt_index

prebuilt_index = 'arqmath-task1'

index_path = from_prebuilt_index(prebuilt_index)
ix = pya0.index_open(index_path, option="r")
if ix is None:
    print('ERR: Cannot open index!')
    quit()

print('Searching ...')
JSON = pya0.search(ix, [
    {'str': r'\tan x + \sec x = 2 \cos x.', 'type': 'tex'},
], verbose = False, topk= 10)
results = json.loads(JSON)
print(json.dumps(results, indent=4))
