import os
import json
import random


def merge_pairs(*json_paths, output_file='output/finetune-pairs.json',
    shuffle=True, seed=72, append='auto'):

    output = []
    for json_path in json_paths:
        with open(json_path, 'r') as fh:
            try:
                j = json.load(fh)
            except json.decoder.JSONDecodeError as e:
                continue
            if append == 'auto':
                if isinstance(j, list):
                    output += j
                else:
                    output.append(j)
            elif append:
                output.append(j)
            else:
                output += j

    if shuffle:
        random.seed(seed)
        random.shuffle(output)

    print(f'Saving {len(output)} pairs ...', output_file)
    with open(output_file, 'w', encoding='utf8', errors='replace') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


def sanity_check(json_path):
    with open(json_path, 'r') as fh:
        j = json.load(fh)
    for i, item in enumerate(j):
        if 'boxed' in item['instruction']:
            print(i)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'merge': merge_pairs,
        'check': sanity_check
    })
