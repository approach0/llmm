MATH_path = '/home/w32zhong/msr/datasets/MATH/test/precalculus'
default_tokenizer = 'approach0/dpr-cocomae-220'
single_vec_model = 'approach0/dpr-cocomae-220'
prebuilt_index = 'arqmath-task1-dpr-cocomae-220-hnsw'

import os
import json
import sys
sys.path.insert(0, './pya0')

from pya0.index_manager import from_prebuilt_index
from pya0.replace_post_tex import replace_dollar_tex, replace_display_tex, replace_inline_tex
from pya0.transformer_eval import psg_encoder__dpr_default, searcher__docid_vec_flat_faiss
from pya0.visualize import output_html

print('Loading model...')
index_path = from_prebuilt_index(prebuilt_index)
encoder, enc_utils = psg_encoder__dpr_default(default_tokenizer, single_vec_model, 0, 0, 'cpu')
searcher, _ = searcher__docid_vec_flat_faiss(index_path, None, enc_utils, 'cpu')
topic = os.path.basename(MATH_path) 

for filename in os.listdir(MATH_path):
    json_path = os.path.join(MATH_path, filename)
    with open(json_path, 'r') as fh:
        j = json.load(fh)
    query = j['problem']
    query = replace_dollar_tex(query)
    query = replace_display_tex(query)
    query = replace_inline_tex(query)
    print('Retrieving docs ...')
    results = searcher(query, encoder, topk=5, debug=False)

    print('TEST:', json_path)
    print('Q:', j['problem'], end='\n\n')
    print('A:', j['solution'], end='\n\n')
    raw_results = [j['solution']]
    for i, res in enumerate(results):
        d = res[2][1]
        raw_results.append(d)
        d = d.replace(r'[imath]', '$')
        d = d.replace(r'[/imath]', '$')
        print(f'D({1+i}):', d, end='\n\n')

    def html(fh, query, hit, page, idx):
        if page == 0 and idx == 0:
            pass
            fh.write(f'<p><b>Ground Truth</b>: {hit} </p>\n\n')
        else:
            fh.write(f'<p><b>Retrieved#{idx}</b>: {hit} </p>\n\n')

    output_html('./output', 'MATH', f'{topic}-{filename}',
        j['problem'], raw_results, None, False, 10, html)
    #input('Hit Enter for the next...')
