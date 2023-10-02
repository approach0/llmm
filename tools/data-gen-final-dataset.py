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


def gen_final_dataset(corpus_dir, output_json='output/final-dataset.json'):
    final_data = []
    n_pos_samples = 0
    n_neg_samples = 0
    for dirname, dirs, files in os.walk(corpus_dir):
        for fname in files:
            if fname.split('.')[-1] != 'log':
                continue
            fpath = os.path.join(dirname, fname)
            try:
                with open(fpath, 'r') as fh:
                    j = json.load(fh)
            except Exception as e:
                print(e)
                continue
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
            true_positive = sum(R_mark) > ratio and sum(Q_mark) == 0

            if true_positive:
                for correct, p in zip(R_mark, paths_QKRA):
                    if not correct: continue
                    srch_query = p[1].state
                    if not 'SEARCH' in srch_query: continue
                    assert srch_query.startswith('SEARCH')
                    srch_query = srch_query.strip('\n')
                    srch_result = p[2].state
                    srch_result = srch_result.strip('\n')
                    answer = p[3].state.strip('\n')
                    d = {
                        'note': note,
                        'problem_id': problem_id,
                        'problem': problem,
                        'solution': sol,
                        'search_query': srch_query,
                        'search_result': srch_result,
                        'answer': answer,
                        'correct': correct,
                        'relevance': sum(R_mark) - ratio
                    }
                    final_data.append(d)
                    n_pos_samples += 1
                    break

            elif n_neg_samples < n_pos_samples:
                for correct, p in zip(R_mark, paths_QKRA):
                    if correct: continue
                    srch_query = p[1].state
                    if not srch_query.startswith('SEARCH'): continue
                    print(srch_query)
                    srch_result = p[2].state
                    srch_result = srch_result.strip('\n')
                    answer = p[3].state.strip('\n')
                    d = {
                        'note': note,
                        'problem_id': problem_id,
                        'problem': problem,
                        'solution': sol,
                        'search_query': srch_query,
                        'search_result': srch_result,
                        'answer': answer,
                        'correct': correct,
                        'relevance': 0
                    }
                    final_data.append(d)
                    n_neg_samples += 1
                    break
            else:
                pass

    print(f'Writing {len(final_data)} rows...')
    with open(output_json, 'w') as fh:
        json.dump(final_data, fh)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(gen_final_dataset)
