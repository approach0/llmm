import json
from datasets import Dataset


def generator(json_file):
    with open(json_file, 'r') as fh:
        j = json.load(fh)

    for item in j:
        if item['problem'] is None:
            continue
        yield item


if __name__ == '__main__':
    json_file = './output/merged_test.json'
    ds = Dataset.from_generator(generator, gen_kwargs={'json_file': json_file})
    print(ds[0])
    ds.push_to_hub("approach0/MATH-picky-test")
