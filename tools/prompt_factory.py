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


#########################
####### version 3 #######
#########################


def multihop1(Q):
    # example is extracted from MATH precalculus/1183.json 
    prompt = r'''Below is an Instruction section that describes a task, paired with an Input section that provides further context.
Write in the Response section that appropriately completes the request.

### Instruction:
Answer a math question in the input. You can invoke a math-aware search engine API (SEARCH[*]) or a computation API (COMPUTE[*]) as you like, and I will insert the returned API results for you right after SEARCH or COMPUTE calls.

At the end, indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.

Now, let me walk you through just one example first.

--- PROBLEM BEGIN ---
Find the number of solutions in the interval $[0,2\pi]$ to equation $\tan x + \sec x = 2 \cos x.$
--- PROBLEM END   ---

I feel uncertain about his problem, let me search the key formula first...

SEARCH[$\tan x + \sec x = 2 \cos x.$]

Here are the results:
--- RESULTS BEGIN ---
I tried changing it all to sin and cos

$$
\frac{\sin x}{\cos x} + \frac{1}{\cos x} = 2\cos x
$$

then I made it to one fraction

$$
\frac{\sin x + 1}{\cos x} = 2 \cos x
$$

Then I don't know where to go from there. Please help!
--- NEXT RESULT ---
That gives you: $\sin x= 2\cos^2x-1 = \cos(2x).$
It should be easy from here.
--- RESULTS END   ---

Okay, the second results look very relevant. Now what left to be solved is

$$
\sin x = \cos(2x)
$$

Let me see if this equation has been solved before...

SEARCH[$\sin x = \cos(2x)$]

Here are the results:
--- RESULTS BEGIN ---
Hint: Use the fact that $\cos(2x) = 1 - 2\sin^2x$.
Then you will have a quadratic for $\sin x$.
--- NEXT RESULT ---
I'm guessing that you mean either for the integral to run from 0 to π/2, or from −π/2 to 0.
--- RESULTS END   ---

Okay, the 2nd result is not relevant, but the first one is a very good idea. If we plug into the formula using $\cos(2x) = 1 - 2\sin^2x$, then

$$
\sin x = 1 - 2\sin^2x
$$

if we let $\sin x = y$, then above is essentially

$$
y = 1 - 2 y^2
$$

this easy equation should be solvable by computation API...

COMPUTE[solve $y = 1 - 2 y^2$ for y]

Here are the results:
--- RESULTS BEGIN ---
[-1, 1/2]
--- RESULTS END ---

Okay, that means $\sin(x)$ is either $-1$ or $1/2$. Let me invoke computation API again.

For $-1$ case...

COMPUTE[solve $\sin x = -1$ for x]

Here are the results:
--- RESULTS BEGIN ---
[-pi/2, 3*pi/2]
--- RESULTS END ---

Okay, for the other $1/2$ case...

COMPUTE[solve $\sin x = 1/2$ for x]

Here are the results:
--- RESULTS BEGIN ---
[pi/6, 5*pi/6]
--- RESULTS END ---

Okay, by collecting these results, and knowning that they have to be in the interval $[0,2\pi]$, we get

[3*pi/2, pi/6, 5*pi/6]

So the number of solutions is $\boxed{2}$.

You get the idea? Now it is your turn!
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


if __name__ == '__main__':
    print(multihop1(r'Let $x$ be a real number such that $\sec x - \tan x = 2$. Find $\sec x + \tan x.$'))
