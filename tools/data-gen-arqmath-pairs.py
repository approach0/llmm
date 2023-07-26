import os
import re
import json
import pickle
from tqdm import tqdm
from xmlr import xmliter
from collections import defaultdict


data_path_prefix = '../datasets/ARQMath-3'
instruction = '''Answer a math question in the input.'''


def read_linked_posts(postlink_file):
    dups_dict = defaultdict(list)
    print(f'Loading {postlink_file} ...')
    for attrs in xmliter(postlink_file, 'row'):
        if attrs['@LinkTypeId'] != '3':
            continue # skip weakly relevant ones (linked posts)
        a = int(attrs['@PostId'])
        b = int(attrs['@RelatedPostId'])
        dups_dict[a].append(b)
        dups_dict[b].append(a)
    return dups_dict


def load_pickle_file(file):
    with open(file, 'rb') as fh:
        print(f'Loading {file} ...')
        return pickle.load(fh)


def replace_imath_tags(txt):
    txt = txt.replace(r'[imath]',  '$')
    txt = txt.replace(r'[/imath]',  '$')
    return txt


def generate_pairs(
    q_dict_file=f'{data_path_prefix}/arqmath-question-dict.pkl',
    a_dict_file=f'{data_path_prefix}/arqmath-answer-dict.pkl',
    answer_bank_file=f'{data_path_prefix}/arqmath-answer-bank.pkl',
    postlink_file=f'{data_path_prefix}/PostLinks.V1.3.xml',
    output_file='output/arqmath-pairs.json',
    min_votes=7, answer_topk=2, max_items=float('inf')):

    dups_dict = read_linked_posts(postlink_file)
    q_dict = load_pickle_file(q_dict_file)
    a_dict = load_pickle_file(a_dict_file)
    answer_bank = load_pickle_file(answer_bank_file)

    all_questions = q_dict.items()
    output = []
    with tqdm(all_questions) as progress:
        for qid, (ac, tags, Q) in progress:
            # get only high-quality questions
            if ac not in a_dict and qid not in dups_dict:
                continue

            positives = []
            if ac in a_dict:
                positives.append(a_dict[ac])

            answers = filter(
                lambda x: int(x[1]) >= min_votes, answer_bank[qid]
            )
            answers = sorted(answers, reverse=True, key=lambda x: int(x[1]))
            positives += [a_dict[a[0]] for a in answers if a[0] != ac]

            positives = filter(
                lambda x:
                re.search('yield', x, re.IGNORECASE) or
                re.search('we get', x, re.IGNORECASE) or
                re.search('answer is', x, re.IGNORECASE)
                , positives
            )

            positives = filter(
                lambda x: not re.search('hint', x, re.IGNORECASE), positives
            )
            positives = filter(
                lambda x: not re.search('http', x, re.IGNORECASE), positives
            )
            positives = filter(
                lambda x: not re.search('pdf', x, re.IGNORECASE), positives
            )
            positives = filter(
                lambda x: not re.search('book', x, re.IGNORECASE), positives
            )
            positives = filter(
                lambda x: not re.search('paper', x, re.IGNORECASE), positives
            )

            for A in list(positives)[:answer_topk]:
                j = {
                    "instruction": instruction,
                    "input": replace_imath_tags(Q),
                    "output": replace_imath_tags(A)
                }
                output.append(j)
            if len(output) >= max_items:
                break

    print(f'Saving {len(output)} pairs ...')
    with open(output_file, 'w', encoding='utf8', errors='replace') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(generate_pairs)
