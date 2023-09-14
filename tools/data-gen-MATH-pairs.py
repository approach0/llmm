import os
import json


dataset_path = '../MATH'
instruction = r'''Answer a math question in the input.

Indicate your final answer in boxed LaTeX. For example, if the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}}.
'''


def generate_pairs(purpose='train', max_items=float('inf'), skip_asy=False):
    dataset_dir=f'{dataset_path}/{purpose}'
    output_file=f'output/MATH-{purpose}.json'

    output = []
    for dirname in os.listdir(dataset_dir):
        data_dir = os.path.join(dataset_dir, dirname)
        assert os.path.isdir(data_dir)
        for fname in os.listdir(data_dir):
            json_path = os.path.join(data_dir, fname)
            print(json_path)
            with open(json_path, 'r') as fh:
                j_original = json.load(fh)
            problem = j_original['problem']
            solution = j_original['solution']
            topic = j_original['type']

            #if topic != 'Precalculus': continue

            # skip those containing ASY graph!!!
            if skip_asy and ('[asy]' in problem or '[asy]' in solution):
                continue

            assert topic in [
                'Algebra', 'Number Theory',
                'Precalculus', 'Geometry',
                'Intermediate Algebra',
                'Counting & Probability',
                'Prealgebra']
            j_instruct = {
                "src_path": json_path.replace(dataset_path + '/', ''),
                "instruction": instruction,
                "input": problem,
                "output": solution
            }
            output.append(j_instruct)
            if len(output) >= max_items:
                break

    print(f'Saving {len(output)} pairs (purpose={purpose})...')
    with open(output_file, 'w', encoding='utf8', errors='replace') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(generate_pairs)
