import os
import time
import openai
from tools.timeout import timeout
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class OpenAI_API():
    def __init__(self, **kwargs):
        openai.api_key = os.environ.get(kwargs['env_token_key'])
        openai.api_type = kwargs['api_type']
        openai.api_version = kwargs['api_version']
        openai.api_base = kwargs['api_base']
        self.engine = kwargs['engine']
        self.abort = None

    @timeout(seconds=30)
    def generate(self, prompt, **kwargs):
        return openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            stream=True,
            **kwargs
        )

    def streamout(self, response):
        res_txt = ''
        for chunk in response:
            choices = chunk['choices']
            if len(choices) > 0:
                choice = choices[0]
                if 'text' in choice:
                    delta = choice['text']
                    if len(delta) == 0:
                        continue
                    res_txt += delta
                    if self.abort and self.abort(res_txt):
                        break
        return res_txt

    def complete(self, prompt, gen_kwargs, stream_kwargs):
        sleep_time = 1
        while True:
            try:
                response = self.generate(prompt, **gen_kwargs)
                res_txt = self.streamout(response)
                break
            except Exception as e:
                print(str(e), f'sleep {sleep_time} secs.')
                time.sleep(sleep_time)
                sleep_time *= 2
                continue
        return res_txt


if __name__ == '__main__':
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read('rl.ini')
    config = cfg['open_ai_gpt3_5']

    from rl import get_cfg_json
    kwargs = get_cfg_json(config, 'openai_init', {})
    api = OpenAI_API(**kwargs)

    gen_kwargs = get_cfg_json(config, 'openai_generate', {})
    stream_kwargs = get_cfg_json(config, 'openai_stream', {})
    r = api.complete('count to 10', gen_kwargs, stream_kwargs)
    print(r)
