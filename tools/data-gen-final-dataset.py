import os
import json
from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset('approach0/mathy-phase2')
    final_data = []
    for data in dataset['train']:
        response_sect = '### Response:\n'
        response = data['prompt'].split(response_sect)[1]
        srch_query = response.split('\n\n')[0]
        i = response.index('--- BEGIN of API results ---')
        srch_results = response[i:]
        d = {
            'note': 'approach0/mathy-phase2',
            'problem_id': data['problem'].replace(r'../MATH/', ''),
            'problem': data['query'],
            'solution': data['solution'],
            'search_query': srch_query,
            'search_result': srch_results,
            'relevance': data['manual_rating']
        }
        final_data.append(d)

    main_corpus = '/home/w32zhong/llmm/output/tree_collection'
    for dirname, dirs, files in os.walk(main_corpus):
        for fname in files:
            if fname.split('.')[-1] != 'log':
                continue
            fpath = os.path.join(dirname, fname)
            with open(fpath, 'r') as fh:
                j = json.load(fh)
            note = '/'.join(dirname.split('/')[-2:] + [fname])
            problem_id = j['path']
            breakpoint()
            d = {
                'note': note,
                'problem_id': problem_id,
                'problem': ,
                'solution': j['solution'],
                'search_query': ,
                'search_result': ,
                'relevance': 
            }
