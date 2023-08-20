import json
from datasets import Dataset


def data_generator(json_file):
    with open(json_file, 'r') as fh:
        j = json.load(fh)

    for item in j:
        if item['problem'] is None:
            continue
        yield item


def collate_prompts(batch_tok_fn, batch_data):
    prompts = [d['prompt'] for d in batch_data]
    return batch_tok_fn(prompts), batch_data


def collate_ask_relevance(batch_tok_fn, batch_data):
    from tools.prompt_factory import ask_relevance
    prompts = [
        ask_relevance(d['prompt'].split('### Input:\n')[1])
        for d in batch_data
    ]
    return batch_tok_fn(prompts), batch_data


if __name__ == '__main__':
    json_file = './output/merged_test.json'
    ds = Dataset.from_generator(data_generator,
        gen_kwargs={'json_file': json_file})
    print(ds[0])
    ds.push_to_hub("approach0/MATH-picky-test")
