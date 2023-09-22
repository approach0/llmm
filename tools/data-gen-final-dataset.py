import os
import json
from datasets import load_dataset

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../Progressive-Hint')
sys.path.insert(0, '../math/modeling')
from rl_mcts import Node
from main_clean import extract_math_answer
from math_equivalence import is_equiv


if __name__ == '__main__':
    dataset = load_dataset('approach0/mathy-phase2', download_mode='force_redownload')
    final_data = []
    for data in dataset['train']:
        response_sect = '### Response:\n'
        response = data['prompt'].split(response_sect)[1]
        srch_query = response.split('\n\n')[0]
        i = response.index('--- BEGIN of API results ---')
        srch_result = response[i:]
        srch_result = srch_result.replace('--- BEGIN of API results ---\n', '')
        srch_result = srch_result.replace('\n--- END of API results ---\n', '')
        srch_result = srch_result.strip('\n')
        d = {
            'note': 'approach0/mathy-phase2',
            'problem_id': data['problem'].replace(r'../MATH/', ''),
            'problem': data['query'],
            'solution': data['solution'],
            'search_query': srch_query,
            'search_result': srch_result,
            'relevance': data['manual_rating']
        }
        final_data.append(d)

    main_corpus = '/home/w32zhong/llmm/output/tree_collection'
    n_pos_samples = 0
    n_neg_samples = 0
    for dirname, dirs, files in os.walk(main_corpus):
        for fname in files:
            if fname.split('.')[-1] != 'log':
                continue
            fpath = os.path.join(dirname, fname)
            with open(fpath, 'r') as fh:
                j = json.load(fh)
            note = '/'.join(dirname.split('/')[-2:] + [fname])
            problem_id = j['path']
            root = Node.from_json(j['json'])
            problem = root.state
            sol = j['solution']
            paths = root.get_all_paths(['Q', 'K', 'R', 'A'])
            paths_QA, paths_QKRA = [], []
            for path in paths:
                path_type = ''.join([n.node_type for n in path])
                if path_type == 'QA':
                    paths_QA.append(path)
                elif path_type == 'QKRA':
                    paths_QKRA.append(path)

            if len(paths_QKRA) == 0:
                continue

            def mark(ans):
                boxed_ans = extract_math_answer(ans)
                boxed_sol = extract_math_answer(sol)
                equiv = is_equiv(boxed_sol, boxed_ans)
                return equiv

            Q_mark = list(map(mark, [p[-1].state for p in paths_QA]))
            R_mark = list(map(mark, [p[-1].state for p in paths_QKRA]))
            ratio = len(R_mark) // len(Q_mark)
            true_positive = sum(R_mark) > ratio * sum(Q_mark)

            if true_positive:
                rel_indicator = (1 + sum(R_mark)) / (ratio * (sum(Q_mark) + 1))
                for correct, p in zip(R_mark, paths_QKRA):
                    if not correct: continue
                    srch_query = p[1].state
                    if not srch_query.startswith('SEARCH'): continue
                    print(srch_query)
                    srch_result = p[2].state
                    srch_result = srch_result.strip('\n')
                    #print(rel_indicator)
                    d = {
                        'note': note,
                        'problem_id': problem_id,
                        'problem': problem,
                        'solution': sol,
                        'search_query': srch_query,
                        'search_result': srch_result,
                        'relevance': int(rel_indicator)
                    }
                    final_data.append(d)
                    n_pos_samples += 1

            elif n_neg_samples < n_pos_samples:
                for correct, p in zip(R_mark, paths_QKRA):
                    if correct: continue
                    srch_query = p[1].state
                    if not srch_query.startswith('SEARCH'): continue
                    print(srch_query)
                    srch_result = p[2].state
                    srch_result = srch_result.strip('\n')
                    d = {
                        'note': note,
                        'problem_id': problem_id,
                        'problem': problem,
                        'solution': sol,
                        'search_query': srch_query,
                        'search_result': srch_result,
                        'relevance': 0
                    }
                    final_data.append(d)
                    n_neg_samples += 1
            else:
                pass

    print(f'Writing {len(final_data)} rows...')
    with open('output/final-dataset.json', 'w') as fh:
        json.dump(final_data, fh)
