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


def gpt4_msr(prompt, **kargs):
    openai.api_key = os.environ.get('GPT4KEY')
    openai.api_type = 'azure'
    openai.api_version = "2023-05-15"
    openai.api_base = 'https://gcrgpt4aoai9c.openai.azure.com/'
    response = openai.Completion.create(
        engine='gpt-4',
        prompt=prompt,
        temperature=0,
        max_tokens=2048,
    )


if __name__ == '__main__':
    #print(gpt4_complete('what is 7 mod 5?'))
    print(gpt4_msr('what is 7 mod 5?'))
