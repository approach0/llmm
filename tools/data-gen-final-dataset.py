import os
import json
from datasets import load_dataset
from collections import defaultdict

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
            paths = root.get_all_paths(['Q', 'K', 'R', 'A', 'E', 'C'])
            paths_by_key = defaultdict(list)
            for path in paths:
                path_type = ''.join([n.node_type for n in path])
                if path_type in ['QA', 'QKRA', 'QECA']:
                    paths_by_key[path_type].append(path)

            def mark(ans):
                boxed_ans = extract_math_answer(ans)
                boxed_sol = extract_math_answer(sol)
                equiv = is_equiv(boxed_sol, boxed_ans)
                return equiv

            marks_by_key = {}
            for key, path in paths_by_key.items():
                marks_by_key[key] = map(mark, [p[-1].state for p in path])
                marks_by_key[key] = list(marks_by_key[key])

            for key in paths_by_key.keys():
                if key == 'QA': continue
                base_mark = marks_by_key['QA']
                aug_mark = marks_by_key[key]
                ratio = len(aug_mark) // len(base_mark)
                true_positive = sum(aug_mark) > ratio and sum(base_mark) == 0

                if true_positive:
                    for correct, p in zip(marks_by_key[key], paths_by_key[key]):
                        if not correct: continue
                        aug_query = p[1].state.strip('\n')
                        print(aug_query)
                        aug_result = p[2].state.strip('\n')
                        answer = p[3].state.strip('\n')
                        d = {
                            'note': note,
                            'problem_id': problem_id,
                            'problem': problem,
                            'solution': sol,
                            'aug_query': aug_query,
                            'aug_result': aug_result,
                            'answer': answer,
                            'correct': correct,
                            'relevance': sum(aug_mark) - ratio
                        }
                        final_data.append(d)
                        n_pos_samples += 1
                        break

                elif n_neg_samples < n_pos_samples:
                    for correct, p in zip(marks_by_key[key], paths_by_key[key]):
                        if correct: continue
                        aug_query = p[1].state.strip('\n')
                        if ('COMPUTE' not in aug_query and
                            'SEARCH' not in aug_query):
                            continue
                        print(aug_query)
                        aug_result = p[2].state.strip('\n')
                        answer = p[3].state.strip('\n')
                        d = {
                            'note': note,
                            'problem_id': problem_id,
                            'problem': problem,
                            'solution': sol,
                            'aug_query': aug_query,
                            'aug_result': aug_result,
                            'answer': answer,
                            'correct': correct,
                            'relevance': 0
                        }
                        final_data.append(d)
                        n_neg_samples += 1
                        break

    print(f'Writing {len(final_data)} rows to {output_json}...')
    with open(output_json, 'w') as fh:
        json.dump(final_data, fh)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(gen_final_dataset)
