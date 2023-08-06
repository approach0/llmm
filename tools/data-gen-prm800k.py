import os
import json
import random
from rich import print as rich_print


instruction = 'Answer a math question in the input.'


def data_read(filepath, lookup, verbose=False):
    filepath = os.path.expanduser(filepath)
    problems = []
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
            if problem.startswith('test/'):
                continue
            elif finish == 'found_error':
                continue
            elif finish == 'give_up':
                continue
            elif finish == 'bad_problem':
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

            j_instruct = {
                "instruction": instruction,
                "input": problem,
                "output": '\n\n'.join(steps)
            }
            problems.append(j_instruct)
    return problems


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


def main(prm_dir='../prm800k/prm800k'):
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
    problems = data_read(PRM_train_jsonl, lookup)
    output += problems

    PRM_train_jsonl = os.path.join(prm_dir,
        'data/phase2_train.jsonl')
    problems = data_read(PRM_train_jsonl, lookup)
    output += problems

    print(f'Saving {len(output)} data ...')
    output_file = 'output/PRM-train.json'
    with open(output_file, 'w', encoding='utf8', errors='replace') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
