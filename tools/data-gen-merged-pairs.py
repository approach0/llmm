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


def data_generator_json(json_file):
    with open(json_file, 'r') as fh:
        j = json.load(fh)
    for item in j:
        yield item


def push_data(train_path, test_path, repo):
    from datasets import Dataset
    from datasets.dataset_dict import DatasetDict

    train_ds = Dataset.from_generator(data_generator_json,
        gen_kwargs={'json_file': train_path})
    test_ds = Dataset.from_generator(data_generator_json,
        gen_kwargs={'json_file': test_path})

    # sanity check
    train_ds = train_ds.filter(lambda x: 'train' in x['src_path'])
    test_ds = test_ds.filter(lambda x: 'test' in x['src_path'])

    dataset = DatasetDict({'train': train_ds, 'test': test_ds})
    dataset.push_to_hub(repo)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'merge': merge_pairs,
        'check': sanity_check,
        'push': push_data,
    })
