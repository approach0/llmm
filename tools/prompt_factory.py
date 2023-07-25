def example(Q):
    with open(f'prompt_{Q}.txt', 'r') as fh:
        return fh.read()


#########################
####### version 1 #######
#########################


def direct1(Q):
    prompt = r'''Please solve the math problem below:

--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---
'''.format(Q)

    prompt += r'''

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} at the very end of your output.
'''
    return prompt


def cot1(Q):
    prompt = r'''You are a mathematician, here is a math problem that I need you to solve:

--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---
'''.format(Q)

    prompt += r'''Let's think step by step, and derive the final answer.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} at the very end of your output.

Keep your answer concise, you have 2048 tokens to finish answering it!
'''
    return prompt


def ia1(Q, *P):
    prompt = r'''You are a mathematician, here is a math problem that I need you to solve:

--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---

To assist you, I have found some potentially relevant passages about this problem:
'''.format(Q)

    for p in P:
        prompt += f'''
--- PASSAGE BEGIN ---
{p}
--- PASSAGE END   ---
'''

    prompt += r'''

Utilize above passages as hints, but feel free to tell me which one(s) is/are useful or not useful to this problem.

Then, let's think step by step and derive the final answer.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} at the very end of your output.

Keep your answer concise, you have 2048 tokens to finish answering it!
'''
    return prompt


#########################
####### version 2 #######
#########################


def direct2(Q):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Answer a math question in the input.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.
'''

    prompt += '''
### Input:
--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---

'''.format(Q)

    prompt += r'''
### Response:
'''
    return prompt


def cot2(Q):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Answer a math question in the input.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.

Let's think step by step, and derive the final answer.
'''

    prompt += '''
### Input:
--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---

'''.format(Q)

    prompt += r'''
### Response:
'''
    return prompt


def ia2(Q, *P):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Answer a math question in the input.
The input is also followed by some potentially relevant passages to assist you.

If you find any passage(s) helpful or not helpful, feel free to tell me.
Utilize them to guide your answer as much as possible.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.

Let's think step by step, and derive the final answer.
'''

    prompt += '''
### Input:
--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---

'''.format(Q)

    for p in P:
        prompt += f'''
--- PASSAGE BEGIN ---
{p}
--- PASSAGE END   ---

'''

    prompt += r'''
### Response:
'''
    return prompt


if __name__ == '__main__':
    print(cot1('Why are universal models difficult to serve at web search scale?'))
    print(ia1('Why are universal models difficult to serve at web search scale?', 'universal models are more general and they require more compute resources'))
