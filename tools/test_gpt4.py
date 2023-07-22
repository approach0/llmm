import os
import time
import openai
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def gpt4_complete(prompt, **kargs):
    openai.api_key = os.environ.get('GPT4KEY')
    while True:
        try:
            completion = openai.ChatCompletion.create(
              model="gpt-4",
              messages=[{"role": "user", "content": prompt}],
              max_tokens=1024
            )
        except Exception as e:
            print('Err', e)
            time.sleep(10)
            continue
        break
    return completion.choices[0].message.content
