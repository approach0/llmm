import os
import re
import json
import random
from rich import print as rich_print


import sys
sys.path.insert(0, '../Progressive-Hint')
sys.path.insert(0, '../math/modeling')
from main_clean import extract_math_answer
from math_equivalence import is_equiv


instruction = r'''Answer a math question in the input.

Indicate your final answer in boxed LaTeX. For example, if the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}}.
'''


def data_read(filepath, lookup, verbose=False):
    filepath = os.path.expanduser(filepath)
    test_problems = []
    train_problems = []
    with open(filepath, 'r') as fh:
        for line in fh:
            j = json.loads(line)
            #if j['is_quality_control_question']:
            #    continue
            generator = j['generation']
            question = j['question']
            problem = question['problem']
            label = j['label']
            solution = question['ground_truth_answer']
            finish = label['finish_reason']

            if problem not in lookup:
                print('Warning: not in lookup:', problem)
                continue
            if finish == 'found_error':
                continue
            elif finish == 'give_up':
                continue
            elif finish == 'bad_problem':
                continue
            elif problem is None:
                continue
            elif '[asy]' in problem:
                continue

            if verbose:
                rich_print(f'[red]{problem}[/red]')
                rich_print(f'[green]{solution}[/green]')
            steps = []
            for j, step in enumerate(label['steps']):
                chosen = step["chosen_completion"]
                completions = step['completions']
                human_compl = step['human_completion']
                if chosen is None:
                    if human_compl is None:
                        continue
                    if verbose:
                        print('@', human_compl['text'])
                    steps.append(human_compl['text'])
                else:
                    chosen_compl = completions[chosen]
                    step_txt = chosen_compl['text']
                    step_rat = chosen_compl['rating']
                    if verbose:
                        print(step_rat, step_txt)
                    steps.append(step_txt)

            if len(steps) == 0:
                continue
            last_step = steps[-1]
            regex = r"# Answer\n\n(.+)"
            m = re.search(regex, last_step)
            if not m:
                continue
            last_step = re.sub(regex, '', last_step)
            steps[-1] = last_step
            agent_answer = m.group(1)
            steps.append(r'The answer is \boxed{'
                + agent_answer + '}.')

            if solution is not None and not is_equiv(agent_answer, solution):
                steps.append('In case you need a different boxed format, '
                + r'the answer is, equivalently, \boxed{' + solution + '}')
                if verbose:
                    print(steps[-2:])
                    print(solution, end='\n\n')
            steps = list(map(lambda x: x.strip(), steps))
            j_instruct = {
                "instruction": instruction,
                "input": problem,
                "output": '\n\n'.join(steps)
            }

            path = lookup[problem]
            if path.startswith('test/'):
                test_problems.append(j_instruct)
            elif path.startswith('train/'):
                train_problems.append(j_instruct)
            else:
                print(path)
                raise ValueError
    return train_problems, test_problems


def MATH_read(filepath, lookup):
    filepath = os.path.expanduser(filepath)
    with open(filepath, 'r') as fh:
        for line in fh:
            j = json.loads(line)
            question = j['problem']
            solution = j['solution']
            answer = j['answer']
            subject = j['subject']
            unique_id = j['unique_id']
            lookup[question] = unique_id


def main(prm_dir='../prm800k/prm800k', purpose='train'):
    lookup = dict()
    MATH_train_jsonl = os.path.join(prm_dir,
        'math_splits/train.jsonl')
    MATH_test_jsonl = os.path.join(prm_dir,
        'math_splits/test.jsonl')
    MATH_read(MATH_train_jsonl, lookup)
    MATH_read(MATH_test_jsonl, lookup)

    output = []
    PRM_train_jsonl = os.path.join(prm_dir,
        'data/phase1_train.jsonl')
    train_problems, test_problems = data_read(PRM_train_jsonl, lookup)
    if purpose == 'train':
        output += train_problems
    else:
        output += test_problems

    PRM_train_jsonl = os.path.join(prm_dir,
        'data/phase2_train.jsonl')
    train_problems, test_problems = data_read(PRM_train_jsonl, lookup)
    if purpose == 'train':
        output += train_problems
    else:
        output += test_problems

    print(f'Saving {len(output)} data ...')
    output_file = f'output/PRM-{purpose}.json'
    with open(output_file, 'w', encoding='utf8', errors='replace') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
