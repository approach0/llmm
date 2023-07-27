import re
import json
import requests
from bs4 import BeautifulSoup

url = 'http://tuna.cs.uwaterloo.ca:8080/test/precalculus/logname__cot-chatgpt-2023-03-15/'
res = requests.get(url)
html_doc = res.content

soup = BeautifulSoup(html_doc, 'html.parser')

all_problems = dict()
for s in soup.find_all('a'):
    print(s.text)
    m = re.search(r"([0-9]+).json", s.text)
    problem = m.group(1)
    all_problems[problem] = True

keys = list(all_problems.keys())
output_json_path = 'output/problem_keys.json'
with open(output_json_path, 'w') as fh:
    json.dump(keys, fh)
print('Output:', output_json_path)
#import pdb; pdb.set_trace()
