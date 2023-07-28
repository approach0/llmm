import os
import wolframalpha

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

wolfram_alpha_appid = os.environ.get('WolframKey')

def WolframAlphaCalculator(input_query: str):
    wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
    res = wolfram_client.query(input_query)
    assumption = next(res.pods).text
    answer = next(res.results).text
    return f"Assumption: {assumption} \nAnswer: {answer}"


if __name__ == '__main__':
    #print(WolframAlphaCalculator('what is a browser?'))
    #print(WolframAlphaCalculator('what is 7 mod 5?'))
    #print(WolframAlphaCalculator('solve $\sin(x) = \cos(2x)$'))
    print(WolframAlphaCalculator('solve x = 1 - 2 x^2'))
