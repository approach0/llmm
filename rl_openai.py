import os
import time
import openai
from tools.timeout import timeout
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class OpenAI_API():
    def __init__(self, **kwargs):
        self.api_key = os.environ.get(kwargs['env_token_key'])
        self.api_type = kwargs['api_type']
        self.api_version = kwargs['api_version']
        self.api_base = kwargs['api_base']

        self.engine = kwargs['engine']
        self.abort = None

    @timeout(seconds=30)
    def generate(self, prompt, **kwargs):
        openai.api_key = self.api_key
        openai.api_type = self.api_type
        openai.api_version = self.api_version
        openai.api_base = self.api_base
        return openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            stream=True,
            **kwargs
        )

    def streamout(self, response, bs, stream):
        res_txt = [''] * bs
        for chunk in response:
            choices = chunk['choices']
            if len(choices) > 0:
                choice = choices[0]
                if 'text' in choice:
                    delta = choice['text']
                    if len(delta) == 0:
                        continue
                    index = choice['index']
                    res_txt[index] += delta
                    if stream:
                        os.system('clear')
                        print(res_txt[index])
                    if self.abort and self.abort(res_txt):
                        break
        return res_txt

    def complete(self, prompts, kwargs):
        assert isinstance(prompts, list)
        sleep_time = kwargs.pop('sleep_time')
        stream = kwargs.pop('stream')
        while True:
            time.sleep(sleep_time)
            try:
                bs = len(prompts)
                response = self.generate(prompts, **kwargs)
                res_txt = self.streamout(response, bs, stream)
                break
            except Exception as e:
                sleep_time *= 2
                print(str(e), f'Sleep {sleep_time} secs.')
                continue
        return res_txt


if __name__ == '__main__':
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read('rl.ini')
    config = cfg['open_ai_gpt3_5_judge_relevance']

    from rl import get_cfg_json
    kwargs = get_cfg_json(config, 'openai_init', {})
    api = OpenAI_API(**kwargs)

    gen_kwargs = get_cfg_json(config, 'openai_gen', {})
    responses = api.complete([
        'find the root: $x^2=4$',
        'count to 10.'
    ], gen_kwargs)
    print(responses)
