import os
import json


dataset_path = '../datasets/MATH'
instruction = r'''Answer a math question in the input.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} at the very end of your output.
'''


def generate_pairs(
    dataset_dir=f'{dataset_path}/train',
    output_file='output/MATH-pairs.json',
    max_items=float('inf')):

    output = []
    for dirname in os.listdir(dataset_dir):
        data_dir = os.path.join(dataset_dir, dirname)
        assert os.path.isdir(data_dir)
        for fname in os.listdir(data_dir):
            json_path = os.path.join(data_dir, fname)
            print(json_path)
            with open(json_path, 'r') as fh:
                j_original = json.load(fh)
                j_instruct = {
                    "instruction": instruction,
                    "input": j_original['problem'],
                    "output": j_original['solution']
                }
                output.append(j_instruct)
            if len(output) >= max_items:
                break

    print(f'Saving {len(output)} pairs ...')
    with open(output_file, 'w', encoding='utf8', errors='replace') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(generate_pairs)
