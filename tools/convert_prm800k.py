import os
import json
import random
from rich import print as rich_print


def data_read(filepath):
    filepath = os.path.expanduser(filepath)
    problems = []
    with open(filepath, 'r') as fh:
        for line in fh:
            j = json.loads(line)
            if j['is_quality_control_question']:
                continue
            if j['is_initial_screening_question']:
                continue
            generator = j['generation']
            question = j['question']
            problem = question['problem']
            label = j['label']
            solution = question['ground_truth_answer']
            rich_print(f'[red]{problem}[/red]')
            rich_print(f'[green]{solution}[/green]')
            for j, step in enumerate(label['steps']):
                chosen = step["chosen_completion"]
                completions = step['completions']
                human_compl = step['human_completion']
                if chosen is None:
                    print('@', human_compl)
                else:
                    chosen_compl = completions[chosen]
                    step_txt = chosen_compl['text']
                    step_rat = chosen_compl['rating']
                    print(step_rat, step_txt)


def MATH_read(filepath):
    filepath = os.path.expanduser(filepath)
    with open(filepath, 'r') as fh:
        for line in fh:
            j = json.loads(line)
            question = j['problem']
            solution = j['solution']
            answer = j['answer']
            subject = j['subject']
            unique_id = j['unique_id']
            #print(unique_id, question)
            if 'What is the least prime factor of $7^4 - 7^3$?' in question:
                print(unique_id)
                print(solution)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'read_data': data_read,
        'read_MATH': MATH_read
    })
