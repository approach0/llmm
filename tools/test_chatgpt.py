import os
import time
import requests
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class OAI_API():
    def __init__(self):
        self.name = "corby"
        self.oaikey = os.environ.get('OAIKey')
        self.engine = "gpt-35-turbo"
        self.deployments = {
            "text-davinci-002" : "text-davinci-002",
            "text-davinci-003" : "text-davinci-003",
            "gpt-35-turbo" : "gpt-35-turbo"
        }
        #self.api_version = "2022-06-01-preview"
        #self.api_version = "2023-03-15-preview"
        self.api_version = "2023-07-01-preview"

    def get_completion(self, prompt, num_tokens = 2048, num_samples = 1,
                    stop = None, include_log_probs = 0,
                    temperature = 0.0, get_probs_of = [], **kargs):
        get_probs_of = set(get_probs_of)
        request_url = (f"https://{self.name}.openai.azure.com/openai/deployments/" +
            f"{self.deployments[self.engine]}/completions")

        headers = {
            'api-key': self.oaikey,
        }
        params = {
            'api-version': self.api_version,
        }
        tries = 0
        max_tries = 5
        output = {}
        sleep_time = 1
        while True:
            try:
                json_data = {
                    'prompt': prompt,
                    'max_tokens': num_tokens,
                    'temperature' : temperature,
                    'stop' : stop,
                    'n' : num_samples,
                }
                print('requesting: ', request_url)
                response = requests.post(request_url, params=params, headers=headers, json=json_data)
                output = response.json()
                valid_output = output.copy()
                valid_output["choices"] = [x['text'].strip() for x in output['choices']]
                if include_log_probs > 0:
                    valid_output["top_logprobs"] = [x['logprobs'] for x in output['choices']]
                if get_probs_of:
                    valid_output["token_log_prob"] = {}
                    for token, prob in zip(
                        output['choices'][0]["logprobs"]["tokens"],
                        output['choices'][0]["logprobs"]["token_logprobs"]):
                        if token in get_probs_of:
                            valid_output["token_log_prob"][token] = prob
                sleep_time = 2
            except Exception as e:
                print(output)
                tries += 1
                if tries > max_tries:
                    return output
                if "error" in output:
                    valid_output = None
                    if ("code" in output and "content_filter" in output["code"]) or "filter" in output["error"].get("message", ""):
                        return None
                    if "maximum context length" in output["error"].get("message", ""):
                        print(f"Error: {output['error']['message']}, shortening prompt.")
                sleep_time *= 2
                time.sleep(sleep_time)
                continue
            time.sleep(sleep_time)
            break
        return valid_output['choices'][0]


import openai


class ChatGPT_Agent():
    def __init__(self):
        self.messages = []

    def reset(self):
        self.messages = []

    def complete(self, prompt='count to 10', stream=True, **kargs):
        openai.api_key = os.environ.get('OAIKey')
        openai.api_type = 'azure'
        #openai.api_version = "2023-07-01-preview"
        openai.api_version = "2023-03-15-preview"
        openai.api_base = 'https://corby.openai.azure.com'

        self.messages.append({
            'role': 'user',
            'content': prompt
        })

        #response = openai.ChatCompletion.create(
        #    engine='gpt-35-turbo',
        #    messages=self.messages,
        #    temperature=0,
        #    stream=True
        #)

        response = openai.Completion.create(
            engine='gpt-35-turbo',
            prompt=prompt,
            temperature=0,
            stream=True
        )

        response_text = ''
        #for chunk in response:
        #    choices = chunk['choices']
        #    if len(choices) > 0:
        #        delta = choices[0]['delta']
        #        if 'content' in delta:
        #            delta = delta['content']
        #            if stream: print(delta, end="")
        #            response_text += delta
        #if stream: print()

        for chunk in response:
            choices = chunk['choices']
            if len(choices) > 0:
                choice = choices[0]
                if 'text' in choice:
                    delta = choice['text']
                    if stream: print(delta, end="")
                    response_text += delta
        if stream: print()

        self.messages.append({
            'role': 'assistant',
            'content': response_text
        })
        return response_text


agent = ChatGPT_Agent()

if __name__ == '__main__':
    #api = OAI_API()
    #out = api.get_completion('Hello! How many languages do you speak?')
    #print(out['choices'][0])

    agent.complete('Hello! How many languages do you speak? Tell me a short story for each of the languages you can speak.')
    #agent.reset() 
    agent.complete("which one is Chinese? I don't see it.")
