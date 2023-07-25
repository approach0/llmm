import os
import re
import json
import pickle
from tqdm import tqdm


data_path_prefix = '../datasets/ARQMath-3'
instruction = '''Answer a math question in the input.'''


def load_pickle_file(file):
    with open(file, 'rb') as fh:
        print(f'Loading {file} ...')
        return pickle.load(fh)


def generate_pairs(
    q_dict_file=f'{data_path_prefix}/arqmath-question-dict.pkl',
    a_dict_file=f'{data_path_prefix}/arqmath-answer-dict.pkl',
    answer_bank_file=f'{data_path_prefix}/arqmath-answer-bank.pkl',
    output_file='output/arqmath-pairs.json',
    min_votes=7, answer_topk=2, max_items=float('inf')):

    q_dict = load_pickle_file(q_dict_file)
    a_dict = load_pickle_file(a_dict_file)
    answer_bank = load_pickle_file(answer_bank_file)

    all_questions = q_dict.items()
    output = []
    with tqdm(all_questions) as progress:
        cnt = 0
        for qid, (ac, tags, Q) in progress:
            positives = []
            if ac in a_dict:
                positives.append(a_dict[ac])

            answers = filter(
                lambda x: int(x[1]) >= min_votes, answer_bank[qid]
            )
            answers = sorted(answers, reverse=True, key=lambda x: int(x[1]))
            positives += [a_dict[a[0]] for a in answers if a[0] != ac]

            positives = filter(
                lambda x: not re.search('hint', x, re.IGNORECASE), positives
            )

            for A in list(positives)[:answer_topk]:
                j = {
                    "instruction": instruction,
                    "input": Q,
                    "output": A
                }
                output.append(j)
            cnt += 1
            if cnt >= max_items:
                break

    with open(output_file, 'w') as fh:
        json.dump(output, fh, indent=2)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(generate_pairs)
