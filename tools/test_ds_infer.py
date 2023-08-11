import os
import requests


def test(prompt='Can you count to 10?',
    url='http://127.0.0.1:8988/generate',
    **kargs):
    if 'args' in kargs:
        url = kargs['args']

    try:
        res = requests.post(url, json={
            'prompt': prompt
        })

        if res.ok:
            return res.text
        else:
            return f'Error: {res.reason}'

    except Exception as e:
        return f'Error: {str(e)}'


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(test)
