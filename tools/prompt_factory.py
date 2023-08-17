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
Answer a math question in the input directly without any thought process.

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
Answer a math question in the input. You can invoke a math-aware search engine API (SEARCH) or a computation API (COMPUTE) as you like, and I will insert the returned API results for you right after SEARCH or COMPUTE calls.

The SEARCH API is followed by its parameters which are a list of keywords in JSON format (DO NOT use "and" to separate keywords, use comma!), for example:

SEARCH["apple", "banana"]

Note that the SEARCH API supports math keywords, but you need to indicate math keywords by wrapping it with dollar signs, for example:

SEARCH["$x^2 = -1$", "imaginary numbers"]

DO NOT mix text and math in one JSON item, i.e. instead of writing:

SEARCH['$what kind of curve is defined by $x^2 - y^2 = 4$']

write keyword by keyword with only one type in each:

SEARCH["curve", "defined by", "$x^2 - y^2 = 4$"]

For the COMPUTE API, it is also followed by its parameters in JSON. The first parameter `mode' is chosen from `calculate', `simplify' or `solve *', whereas the second parameter is the symbolic expression in LaTeX.

For example, to calculate 3 / 2, you can do:

COMPUTE["calculate", "$\\frac{3}{2}$"]

To simplify $x + y - 3 + 2 + x$, you can do:

COMPUTE["simplify", "$x + y - 3 + 2 + x$"]

And to solve "$y = 1 - 2 y^2$" for y, you can do:

COMPUTE["solve y", "$y = 1 - 2 y^2$"]

For the SEARCH API, only consider helpful API results for your goal, ignore irrelevant ones.
For the COMPUTE API, remember it is limited to simple tasks. It does not support linear algebra, nor matrix manipulations.

When API returns any error, exam your query and check whether your argument is a valid JSON, if you find an error, call the same API with corrected argument(s) again!
When results are not helpful, do explore alternative ways. You do not have to rely on the previous result(s)!

At the end, indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.

Now, let me walk you through just one example first.

--- PROBLEM BEGIN ---
Find the number of solutions in the interval $[0,2\pi]$ to equation $\tan x + \sec x = 2 \cos x.$
--- PROBLEM END   ---

You might feel uncertain about his problem, let us search the key formula first...

SEARCH["$\\tan x +\\sec x =2\\cos x$"]

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
--- RESULTS END ---

Okay, the second result looks very relevant.
If we follow that thought, what left to be solved is

$$
\sin x = \cos(2x)
$$

Let us see if this equation has been solved before...

SEARCH["$\\sin x = \\cos(2x)$"]

Here are the results:
--- RESULTS BEGIN ---
Hint: Use the fact that $\cos(2x) = 1 - 2\sin^2x$.
Then you will have a quadratic for $\sin x$.
--- NEXT RESULT ---
I'm guessing that you mean either for the integral to run from 0 to π/2, or from −π/2 to 0.
--- RESULTS END ---

Okay, the 2nd result is not relevant, but the first one is a very good idea.
If we plug into that formula using $\cos(2x) = 1 - 2\sin^2x$, we get

$$
\sin x = 1 - 2\sin^2x
$$

now, if we let $\sin x = y$, then above is essentially

$$
y = 1 - 2 y^2
$$

this easy equation should be solvable by computation API...

COMPUTE["solve y", "$y = 1 - 2 y^2$"]

Here are the results:
--- RESULTS BEGIN ---
\left[ -1, \  \frac{1}{2}\right]
--- RESULTS END ---

Okay, that means $\sin(x)$ is either $-1$ or $1/2$. Let me invoke computation API again.

For $\sin(x) = -1$ case...

COMPUTE["solve x", "$\\sin x = -1$"]

Here are the results:
--- RESULTS BEGIN ---
\left[ - \frac{\pi}{2}, \  \frac{3 \pi}{2}\right]
--- RESULTS END ---

Okay, for the other case when $\sin(x) = 1/2$...

COMPUTE["solve x", "$\\sin x = 1/2$"]

Here are the results:
--- RESULTS BEGIN ---
\left[ \frac{\pi}{6}, \  \frac{5 \pi}{6}\right]
--- RESULTS END ---

Okay, by collecting these results, and knowing that they have to be in the interval $[0,2\pi]$, we get

[3*pi/2, pi/6, 5*pi/6]

So the number of solutions is $\boxed{3}$.

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


def multihop_err1(msg='API error!'):
    return '''
Here are the results:
--- RESULTS BEGIN ---
{}
--- RESULTS END ---
'''.format(msg)


def multihop_results1(results):
    prompt = '''
Here are the results:
--- RESULTS BEGIN ---
'''
    if isinstance(results, str):
        prompt += results + '\n'
    else:
        for i, res in enumerate(results):
            prompt += res + '\n'
            if i != len(results) - 1:
                prompt += '--- NEXT RESULT ---\n'
    prompt += '--- RESULTS END ---\n\n'
    return prompt


def ask_identity_formula_logits(Q):
    prompt = r'''Below is an Instruction section that describes a task, paired with an Input section that provides further context.
Write in the Response section that appropriately completes the request.

### Instruction
Given a math problem, tell me if you need to look up a formula to better answer this question. The criteria is, imagine you have a math-aware search engine where you can search with formula(s), if you think there is any formula (do not consider a matrix) can be very helpful finding a relevant answer to this question by matching a structurally similar formula occurred in the relevant answer, then it meets the requirement of being a key formula.

Do write your answer in either [yes] or [no] to indicate whether there is such key formula for solving this problem. Do not include extra output or values!

Example 1
--- PROBLEM BEGIN ---
If $\tan^{-1} x + \tan^{-1} y = \frac{\pi}{4},$ then compute $xy + x + y.$
--- PROBLEM END   ---

The first formula occurred in this problem looks very unique to this question, it also looks like a good identity as it is beautiful and likely to be popular.
On the other hand, $xy + x + y.$ looks like a very common expression, as a result, searching it may not recall this question compared to searching for the first formula.
I am pretty confident $\tan^{-1} x + \tan^{-1} y = \frac{\pi}{4}$ is the key formula, my answer is [yes].

Example 2
--- PROBLEM BEGIN ---
Three vertices of a cube in space have coordinates $A = (2,3,0),$ $B = (0,5,4),$ and $C = (4,1,8).$
Compute the coordinates of the center of the cube.
--- PROBLEM END   ---

For this one, the three formulas $A = (2,3,0),$ $B = (0,5,4),$ and $C = (4,1,8).$ all look very common, I am afraid using any of them as key formula(s) will restrict the search results too much.
As a result, it is highly unlikely they are key formulas for this problem. My answer is [no].

'''

    prompt += '''
### Input
--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---

### Response

'''.format(Q)

    return prompt


def ask_identity_formula(Q):
    prompt = r'''Below is an Instruction section that describes a task, paired with an Input section that provides further context.
Write a response in the Response section that appropriately completes the request.

### Instruction
Given a math problem, tell me if it requires a key formula to answer. A key formula is a expression that is relevant for solving this problem, and can be used in a math-aware search engine to look up similar expressions and the solutions.

Do not consider a matrix in formulas.

Please indicate, on a scale of 1-10 and in the format of "prob[x]", the likelihood that there is such a key formula for solving this problem.
If you do not think this problem has a key formula, indicate it with a low score.

Example 1
--- PROBLEM BEGIN ---
If $\tan^{-1} x + \tan^{-1} y = \frac{\pi}{4},$ then compute $xy + x + y.$
--- PROBLEM END   ---

There is a structure in the first expression that can be looked up by a math search engine, it could be useful because it looks like an identity, and is unique to this problem.

For this question, my rating for having such key formula is prob[10].

Example 2
--- PROBLEM BEGIN ---
Three vertices of a cube in space have coordinates $A = (2,3,0),$ $B = (0,5,4),$ and $C = (4,1,8).$
Compute the coordinates of the center of the cube.
--- PROBLEM END   ---

There are 3 expressions in this problem, but none of them would be informative about how to solve the problem. Instead, the key information is in the textual components.

As a result, for this question, my rating for having such key formula is prob[1].

Example 3
--- PROBLEM BEGIN ---
Find $\cos \frac{\pi}{3}.$
--- PROBLEM END   ---

The only expression in this problem might be looked up by a search engine, with potentially the correct answer, but it can also be solved directly. I am not too sure if it is useful to use the math search engine.
As a result, for this question, my rating for having such a key formula is prob[5].

'''

    prompt += '''
### Input
--- PROBLEM BEGIN ---
{}
--- PROBLEM END   ---

### Response

'''.format(Q)

    return prompt


#########################
####### version 3 #######
#########################

def cot_wizard(Q):
    prompt = r'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{Q}
'''.format(Q=Q)

    prompt += r'''
### Response:
'''
    return prompt

def cot_mytrain(Q):
    prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer a math question in the input.\n\nIndicate your final answer in boxed LaTeX. For example, if the final answer is \\sqrt{3}, write it as \\boxed{\\sqrt{3}}.\n

### Input:
'''
    prompt += Q + '\n'

    prompt += r'''
### Response:
'''
    return prompt


def multihop_simple(Q):
    prompt = r'''Answer a math question in the input, with the help of a SEARCH engine API.

Given an example math question:

Q: Find the number of solutions in the interval $[0,2\pi]$ to equation $\tan x + \sec x = 2 \cos x.$

You might want to invoke the search API like this:

SEARCH["$\\tan x +\\sec x =2\\cos x$"]

Now, answer the following math question:
'''

    prompt += '''
Q: {}

'''.format(Q)

    return prompt


if __name__ == '__main__':
    # precalculus/1185.json
    prompt = multihop1(r'Let $x$ be a real number such that $\sec x - \tan x = 2$. Find $\sec x + \tan x.$')
    print(prompt)
    print('length:', len(prompt))
