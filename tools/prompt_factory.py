import json


#########################
####### version 1 #######
#########################


def example(Q):
    with open(f'prompt_{Q}.txt', 'r') as fh:
        return fh.read()


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

SEARCH['$what kind of curve is defined by x^2 - y^2 = 4$']

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


def final_tool_augment_prompt1(Q):
    prompt = r'''Below is an Instruction section that describes a task, paired with an Input section that provides further context.
Write in the Response section that appropriately completes the request.

### Instruction:
Answer a math question in the input.

To assist you, you can invoke a math-aware search API (i.e., SEARCH) or a computation API (COMPUTE), and I will insert the returned API results for you right after each valid SEARCH or COMPUTE calls.

The SEARCH API is followed by its parameters which are a list of keywords in JSON format, for example:

SEARCH["$x^2 = -1$", "imaginary numbers"]

DO NOT mix text and math in one JSON item, i.e. instead of writing:

SEARCH['$what kind of curve is defined by x^2 - y^2 = 4$']

write keyword by keyword with only one type in each:

SEARCH["curve", "defined by", "$x^2 - y^2 = 4$"]

For the COMPUTE API, it is also followed by its parameters in JSON. The first parameter `mode' is chosen from `calculate', `simplify' or `solve *', whereas the second parameter is the symbolic expression in LaTeX.

For example, to calculate sine of 270 degree, you can do:

COMPUTE["calculate", "\\sin(270 \\times \\frac{\\pi}{180})"]

To simplify $\sin^2 x + \cos^2 x$, you can do:

COMPUTE["simplify", "\\sin^2(x) + \\cos^2(x)"]

And to solve $y = 1 - 2 y^2$ for y, you can do:

COMPUTE["solve y", "y = 1 - 2 y^2"]

For the SEARCH API, only consider helpful API results for your goal, ignore irrelevant ones.
For the COMPUTE API, remember it is limited to simple tasks. It does not support linear algebra, nor matrix manipulations.

When the API result is helpful, you can just rely on the result or extract the final answer from it directly, in such case, there is no need to answer from the begining and redo any existing derivations in the result.

When API results are not helpful, ignore the results and answer the given math question directly!

At the end, indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.

Take a deep breath and now I will hand the math question to you!
'''

    prompt += '''
### Input:
{}
'''.format(Q)

    prompt += r'''
### Response:
'''
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


def ask_relevance(after_input_sect):
    prompt = r'''Below is an Instruction section that describes a task, paired with an Input section that provides further context.
Write a response in the Response section that appropriately completes the request.

### Instruction
Given a math problem, and a provided search result potentially relevant to this problem.

Please indicate, on a scale of 0 to 2 and in the format of "rate[x]", the relevance of this search result.

The criteria of judging the relevance of a provided search result passage is:

rate[0]: The provided passage is not helpful in solving the given math problem;

rate[1]: The provided passage is relevant, but it requires some further derivations and efforts in solving the given math problem;

rate[2]: The provided passage is fully relevant, we can follow the process and easily derive the answer, or we can even find the solution directly in the provided passage.

Here is an example math problem:

Show $\frac{a_1^2}{a_1+a_2}+\frac{a_2^2}{a_2+a_3}+ \cdots \frac{a_n^2}{a_n+a_1} \geq \frac12$.

I will give a judgement for each of search results below for your reference:

--- BEGIN of API results ---
URL: https://math.stackexchange.com/questions/zzz

#### User Answer (Upvotes: 8)
AM-GM gives: $\frac{\left(\frac{s-a_1}{n-1}\right)a_1 + \left(\frac{s-a_2}{n-1}\right)a_2+ \cdots +\left(\frac{s- a_n}{n-1}\right)a_n}{a_1 + a_2 +\cdots+a_n} \ge \left(\left(\frac{s-a_1}{n-1}\right)^{a_1} \cdot \left(\frac{s-a_2}{n-1}\right)^{a_2}\cdots \left(\frac{s-a_n}{n-1}\right)^{a_n}\right)^{\frac{1}{s}}$

We can show AM of the numbers:
$$
\frac{s}{n} \ge \frac {2\sum\limits_{1\le i\lt j\le n} a_ia_j}{(n-1)s}
$$
$\implies (n-1)\sum\limits_{i=1}^n a_i^2 \ge 2\sum\limits_{1\le i\lt j\le n} a_ia_j$.
--- END of API results ---

This result is indeed showing a proof for an inequality, but I do not see any relation between it and the given math question.
Without noticing any relevant part that is truly helpful, I will give a relevance of rate[0].

--- BEGIN of API results ---
URL: https://math.stackexchange.com/questions/xxx

#### User Answer (Upvotes: 8)
By the Cauchy-Schwarz inequality we have:
$$
\frac{a_1^2}{a_1+a_2}+\frac{a_2^2}{a_2+a_3}+ \cdots \frac{a_n^2}{a_n+a_1}=\frac{a_1^2}{(\sqrt{a_1+a_2})^2}+\frac{a_2^2}{(\sqrt{a_2+a_3})^2}+ \cdots+ \frac{a_n^2}{(\sqrt{a_n+a_1})^2} \geq \frac{1}{a_1+\cdots + a_n+a_1+ \cdots + a_n}\left(\frac{a_1 \cdot \sqrt{a_1+a_2}}{\sqrt{a_1+a_2}} + \frac{a_2 \cdot \sqrt{a_2+a_3}}{\sqrt{a_2+a_3}}+ \cdots + \frac{a_n \cdot \sqrt{a_n+a_1}}{\sqrt{a_n+a_1}}\right)\\=\frac{a_1+a_2+a_3+ \cdots a_n}{{2(a_1+a_2+a_3+ \cdots a_n)}}=...
$$

Can you handle the rest?
--- END of API results ---

This one is an fully relevant answer to an identical math question, although it require one more step to derive, it is easy to see the following step will complete the proof. So the relevance should be rate[2].

--- BEGIN of API results ---
URL: https://math.stackexchange.com/questions/yyy

#### User Answer (Upvotes: 8)
Use the CBS inequality:
$\frac{x_{1}^{2}}{a_{1}} + \cdots + \frac{x_{n}^{2}}{a_{n}} \geq \frac{(x_{1} + \cdots + x_{n})^{2}}{a_{1}+\cdots+a_{n}}.$
--- END of API results ---

This result might be answering the same question, but it may take a few more steps to check if the CBS inequality is indeed useful for our proof. At least the proof cannot be followed directly and clearly from this inequality, so the relevance should be rate[1].

Got it? Now, it is your turn!

### Input:
'''
    return prompt + after_input_sect


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


def cot_wizard_asking_for_boxed(Q):
    prompt = r'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.
'''
    prompt += '''
### Instruction:
{Q}
'''.format(Q=Q)

    prompt += r'''
### Response:
'''
    return prompt


def ia_wizard(Q, *P):
    prompt = r'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

The instruction is also followed by some potentially relevant passages to assist you.
Utilize them to guide your answer as much as possible.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.
'''
    prompt += r'''
### Instruction:
{Q}

Here are some relevant passages:
'''.format(Q=Q)

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


def adapt_wizard(Q, *P):
    if len(P) == 0:
        return cot_wizard_asking_for_boxed(Q)
    else:
        return ia_wizard(Q, *P)


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


def ia_mytrain(Q, Qry, *P):
    prompt = tool_prompt1(Q)
    if isinstance(Qry, str):
        prompt += r'''

### Response:
{QryJson}
'''.format(QryJson=Qry)
    else:
        prompt += r'''

### Response:
SEARCH{QryJson}
'''.format(QryJson=json.dumps(Qry))

    prompt += f'\n--- BEGIN of API results ---\n'
    for i, p in enumerate(P):
        prompt += f'{p}\n'
        if i < len(P):
            prompt += '\n'

    prompt += f'--- END of API results ---\n'
    return prompt


#########################
####### version 4 #######
#########################


def tool_prompt1(Q):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer a math question in the input.

Indicate your final answer in boxed LaTeX. For example, if the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}}.

To assist you, you can invoke a math-aware search engine API (i.e., SEARCH) to find potentially relevant search results to the math question, and I will insert these results for you right after every API call.

An API call should be followed by its arguments which are a list of keywords in JSON format.

API arguments for the SEARCH API may contain useful keywords for finding relevant results, for example:

SEARCH["$x^2 = -1$", "imaginary numbers"]

DO NOT mix text and math in one JSON item, for example, this is NOT good:

SEARCH['$what kind of curve is defined by x^2 - y^2 = 4$']

Instead, separate different types of keywords by comma, and pick important keyword(s):

SEARCH["curve", "defined by", "$x^2 - y^2 = 4$"]

When an API returns any error or unexpected messages, exam your arguments carefully (e.g., look for format issues in JSON).
You may call the same API with corrected argument(s) again.

Only consider helpful API results for your goal, ignore irrelevant ones.
When search results are not helpful, do explore alternative ways. You do not have to rely on every API result.

When a search result is helpful, you can just rely on the result or extract the final answer from it directly,
in such case, there is no need to answer from the begining and redo any existing derivations in the result.

### Input:
'''
    prompt += Q
    return prompt


def find_good_keywords_1(Q):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Find good search keywords for a math question in the input.

You may use text keyword(s) or key formula(s) occurred in the question, or any relevant ones you think might be good keywords.

A key formula is an expression that is relevant for solving this problem, and can be used in a math-aware search engine to look up similar expressions and the solutions. As a result, a rare/identity formula (e.g., $x^2+y^2=z^2$) is better than a common/average formula (e.g., $a>0$).

Please indicate, using the SEARCH API call, with your search keywords in its arguments which a list of keywords in JSON format.

Here is an example:

SEARCH["$x^2 = -1$", "imaginary numbers"]

Note that the arugments MUST be a valid json, so this is NOT good:

SEARCH["$V = \pi R^2 H$"]

Instead, write:

SEARCH["$V = \\pi R^2 H$"]

DO NOT mix text and math in one JSON item, for example, this is NOT good:

SEARCH['$what kind of curve is defined by x^2 - y^2 = 4$']

Instead, separate different types of keywords by comma, and pick important keyword(s):

SEARCH["curve", "defined by", "$x^2 - y^2 = 4$"]

HINT: by only selecting and copying identity keywords from the original prolbem will get you a good baseline.

Now, take a deep breath and I now handle my math question to you!

### Input:
'''
    prompt += Q
    return prompt


def find_good_compute_call_1(Q):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Try to use a compute API to solve a math question in the input.

The compute API call, starts with the COMPUTE keyword, is then followed by its arguments in JSON format.
The first parameter `mode' can only be chosen from `calculate', `simplify' or `solve *', whereas the second parameter is the symbolic expression in LaTeX.

For example, to calculate sine of 270 degree, you can do:

COMPUTE["calculate", "\\sin(270 \\times \\frac{\\pi}{180})"]

To simplify $\sin^2 x + \cos^2 x$, you can do:

COMPUTE["simplify", "\\sin^2(x) + \\cos^2(x)"]

And to solve $y = 1 - 2 y^2$ for y, you can do:

COMPUTE["solve y", "y = 1 - 2 y^2"]

Lastly, please stop generating at your first API call, and you do not need to solve the problem.

Now, take a deep breath and I now handle my input to you!

### Input:
'''
    prompt += Q
    return prompt


def comp_aug_ans_prompt(Q, E, C):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve a given math question in the input with the help of a compute API call.

The compute API call, starts with the COMPUTE keyword, is then followed by its arguments in JSON format.
The first parameter `mode' can only be chosen from `calculate', `simplify' or `solve *', whereas the second parameter is the symbolic expression in LaTeX.

For example, to calculate sine of 270 degree, you can do:

COMPUTE["calculate", "\\sin(270 \\times \\frac{\\pi}{180})"]

To simplify $\sin^2 x + \cos^2 x$, you can do:

COMPUTE["simplify", "\\sin^2(x) + \\cos^2(x)"]

And to solve $y = 1 - 2 y^2$ for y, you can do:

COMPUTE["solve y", "y = 1 - 2 y^2"]

The API response will be inserted for you right after every API call.

If API response are helpful, utilize it to solve the given math problem.
Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} at the very end of your output.

Now, take a deep breath and I now handle my input to you!

### Input:
'''
    prompt += Q.strip('\n')
    prompt += '\n\n'

    prompt += r'''
### Response:
'''
    prompt += E.strip('\n')
    prompt += '\n'
    prompt += multihop_results1([C])
    return prompt


def ask_relevance_1(Q, P):
    prompt = r'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction
Given a math problem, and a provided search result potentially relevant to this problem.

Please indicate, on a scale of 0 to 2 and in the format of "rate[x]", the relevance of this search result.

The criteria of judging the relevance of a provided search result (in a passage) is:

rate[0]: The provided passage is not helpful in solving the given math problem;

rate[1]: The provided passage is relevant, one can potentially improve the likelihood to solve the given math problem by utilizing the provided passage.

rate[2]: The provided passage is fully relevant, one can follow the thought and easily derive the answer, or even extract the answer from the provided passage directly.

Now, take a deep breath and I now handle my input to you!

### Input:
'''
    prompt += Q
    prompt += '''

### Passage:
'''
    prompt += P
    return prompt


def DPO_default_prompt(instr, input):
    template = r'''Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{instr}
### Input:
{input}

### Response:
'''
    return template.format(instr=instr, input=input)


def prompt_tora(Q):
    prompt = f"<|user|>\n{Q}\n<|assistant|>\n"
    return prompt


def prompt_abel(Q):
    prompt = f"Question:\n{Q}\nAnswer:\nLet's think step by step.\n"
    return prompt


def prompt_metamath(Q):
    prompt = (f"Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{Q}\n\n### Response: Let's think step by step.")
    return prompt


def prompt_Llemma(Q):
    prompt = "Problem:\n" + Q + "\n\nSolution:"
    return prompt


if __name__ == '__main__':
    #prompt = ia_mytrain('Q', ['k1', 'k2 + k3'], 'foo', 'bar')
    prompt = adapt_wizard('Q', 'k1', 'k2')
    print(prompt)
    print('length:', len(prompt))
