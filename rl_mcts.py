from tools.prompt_factory import *
from functools import partial

from rl import get_cfg_json

import rl_data
from rl_tools import (
    sympy_solver,
    search_mux,
    has_any_captured,
    tool_invoke,
    ToolError
)


def direct_answering(step, K, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    batch_out = res_fn(config, models, batch_in)
    rewards = rwd_fn(config, batch_in, batch_out, models,
        **get_cfg_json(config, 'reward_args', {})
    )
    if log_fn:
        log_fn(locals(), **get_cfg_json(config, 'log_args', {}))


def rl_query_lm(step, K, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    list_batch, batch_raw = batch_in
    print(tokenizer.decode(list_batch[0]['input_ids'][0]))

    batch_out = res_fn(config, models, batch_in, trainer=trainer)
    print(tokenizer.decode(batch_out[0]))
    rewards = rwd_fn(config, batch_in, batch_out, models,
        **get_cfg_json(config, 'reward_args', {})
    )

    if config.getboolean('compare_refout', False):
        import numpy as np
        with model.pretrained_model.disable_adapter():
            ref_batch_out = res_fn(config, models, batch_in, trainer=trainer)
        ref_rewards = rwd_fn(config, batch_in, ref_batch_out, models,
            **get_cfg_json(config, 'reward_args', {})
        )
        cmp_rewards = np.array(rewards) - np.array(ref_rewards)
        rewards = cmp_rewards.tolist()

    if stp_fn and trainer:
        stats = stp_fn(config, trainer, batch_in, batch_out, rewards)

    if log_fn:
        batch_outstr = [
            out if isinstance(out, str) else tokenizer.decode(out)
            for out in batch_out
        ]
        log_fn(locals(), **get_cfg_json(config, 'log_args', {}))


def infer_query_lm(step, K, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    batch_out = res_fn(config, models, batch_in)
    query_key = config.get('collate__query_key', 'query')

    for inp, out_str in zip(batch_in[1], batch_out):
        out_str = out_str.replace('</s>', '').replace('<s>', '')
        inp['out_str'] = out_str
        uri = 'mabowdor'
        query = inp[query_key]
        tool_map = {
            'SEARCH': partial(
                search_mux, uri, query
            )
        }
        if not has_any_captured(out_str, tool_map):
            inp['tool_res'] = None
        else:
            pre_invoke, tool_res = tool_invoke(out_str, tool_map)
            if isinstance(tool_res, ToolError):
                inp['tool_res'] = None
            else:
                inp['tool_res'] = tool_res

    log_fn(locals(), **get_cfg_json(config, 'log_args', {}))


class Node():
    def __init__(self, node_type, state):
        self.node_type = node_type
        self.state = state
        self.children = []
        self.parent = None
        self.prompt = None

    def branch(self, node_type, child_state):
        node = Node(node_type, child_state)
        self.children.append(node)
        node.parent = self
        return node

    def __repr__(self):
        repr_state = self.state.replace('\n', r'\n')
        repr_state = repr_state[:128]
        return f'[{self.node_type}] {repr_state}...'

    def print_tree(self, level=0):
        print('  ' * level, self)
        for child in self.children:
            child.print_tree(level + 1)

    def json(self):
        return {
            'node_type': self.node_type,
            'state': self.state,
            'prompt': self.prompt,
            'children': [x.json() for x in self.children],
        }

    @staticmethod
    def from_json(obj):
        node = Node(obj['node_type'], obj['state'])
        node.prompt = obj['prompt']
        for child in obj['children']:
            new_node = Node.from_json(child)
            node.children.append(new_node)
            new_node.parent = node
        return node

    def get_path(self, include=[]):
        curr = self
        path = []
        while curr:
            if curr.node_type in include:
                path.append(curr)
            curr = curr.parent
        return path[::-1]

    def get_all_paths(self, include=[]):
        paths = []
        if len(self.children) > 0:
            for child in self.children:
                paths += child.get_all_paths(include)
        else:
            paths.append(self.get_path(include))
        return paths

    @staticmethod
    def gn(config, models, tok_fn, res_fn, inp):
        prompts = tok_fn([inp], eos=False)
        out = res_fn(config, models, (prompts, {'gn': inp}))
        return out[0]

    def calc_enter(self, config, models, tok_fn, res_fn, tm):
        assert self.node_type == 'Q'
        inp = find_good_compute_call_1(self.state)
        inp += '\n\n### Response:\n'
        out = Node.gn(config, models, tok_fn, res_fn, inp)
        if has_any_captured(out, tm):
            out, _ = tool_invoke(out, tm, dryrun=True)
            return inp, out
        else:
            return inp, None

    def calc_compute(self, config, models, tok_fn, res_fn, tm):
        assert self.node_type == 'E'
        assert self.parent.node_type == 'Q'
        _, res = tool_invoke(self.state, tm)
        if isinstance(res, ToolError):
            print('ToolError:', res)
            return None
        else:
            return res

    def query_calculator(self, config, models, tok_fn, res_fn, tm):
        inp, out = self.calc_enter(config, models,
            tok_fn, res_fn, tm)
        if out is None:
            return None, None
        e_node = self.branch('E', out)
        e_node.prompt = inp
        result = e_node.calc_compute(config, models,
            tok_fn, res_fn, tm)
        return e_node, result

    def keywords(self, config, models, tok_fn, res_fn, tm):
        assert self.node_type == 'Q'
        args = get_cfg_json(config, 'mcts_args', {})
        prompt_func_name = args.get(
            'keywords_prompt',
            'find_good_keywords_1'
        )
        prompt_func = globals()[prompt_func_name]
        inp = prompt_func(self.state)
        inp = inp.strip()
        inp += '\n\n### Response:\n'
        out = Node.gn(config, models, tok_fn, res_fn, inp)
        if has_any_captured(out, tm):
            out, _ = tool_invoke(out, tm,
                dryrun=True, args=[self.state])
            return inp, out
        else:
            return inp, None

    def search(self, config, models, tok_fn, res_fn, tm):
        assert self.node_type == 'K'
        assert self.parent.node_type == 'Q'
        _, res = tool_invoke(self.state, tm,
                args=self.parent.state)
        if isinstance(res, ToolError):
            print('ToolError:', res)
            return None
        elif len(res) == 0:
            print('Empty results!')
            return None
        else:
            return res

    def query_retriever(self, config, models, tok_fn, res_fn, tm):
        inp, out = self.keywords(config, models,
            tok_fn, res_fn, tm)
        if out is None:
            return None, None
        k_node = self.branch('K', out)
        k_node.prompt = inp
        results = k_node.search(config, models,
            tok_fn, res_fn, tm)
        if results is None:
            return k_node, []
        else:
            return k_node, results

    def answer(self, config, models, tok_fn, res_fn, tm):
        nodes = self.get_path(['Q', 'R', 'E', 'C'])
        states = [n.state for n in nodes]
        types = [n.node_type for n in nodes]
        if nodes[-1].node_type == 'C':
            inp = comp_aug_ans_prompt(*states[-3:])
        else:
            inp = adapt_wizard(*states)
        out = Node.gn(config, models, tok_fn, res_fn, inp)
        return inp, out


def mcts_explore(step, K, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    from rl import batch_tokenize
    tokenizer, model, ref_model = models
    tok_fn = partial(batch_tokenize, config, tokenizer)
    tool_map = {
        'SEARCH': partial(
            search_mux, 'mabowdor'
        ),
        'COMPUTE': sympy_solver
    }
    params = config, models, tok_fn, res_fn, tool_map

    root = Node('Q', batch_in['input'][0])
    curr = root
    query_only = config.getboolean('query_only', False)

    if not query_only:
        for _ in range(K):
            inp, answer = curr.answer(*params)
            a_node = curr.branch('A', answer)
            a_node.prompt = inp

    if not query_only:
        for _ in range(K):
            e_node, result = curr.query_calculator(*params)
            if result is None: continue
            c_node = e_node.branch('C', result)
            for _ in range(K):
                inp, answer = c_node.answer(*params)
                a_node = c_node.branch('A', answer)
                a_node.prompt = inp

    for _ in range(K):
        k_node, results = curr.query_retriever(*params)
        if not k_node: continue
        for res in results or []:
            r_node = k_node.branch('R', res)
            if query_only: continue
            for _ in range(K):
                inp, answer = r_node.answer(*params)
                a_node = r_node.branch('A', answer)
                a_node.prompt = inp

    root.print_tree()
    log_fn(step, batch_in['src_path'][0], root.json(),
        sol=batch_in['output'][0])


def mcts_explore_on_trees(step, K, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    from rl import batch_tokenize
    tokenizer, model, ref_model = models
    tok_fn = partial(batch_tokenize, config, tokenizer)
    params = config, models, tok_fn, res_fn, None

    inp = batch_in[1][0]
    path = inp['path']
    solution = inp['solution']
    json = inp['json']
    root = Node.from_json(json)

    def dfs(n):
        if n.node_type in ['Q', 'R']:
            for _ in range(K):
                inp, answer = n.answer(*params)
                ans_n = n.branch('A', answer)
                ans_n.prompt = inp
        for child in n.children:
            dfs(child)

    dfs(root)
    log_fn(step, path, root.json(), sol=solution)



#def concat(tok_fn, batch_in, batch_out):
#    dict_batch, batch_raw = batch_in
#    if 'input_ids' in dict_batch:
#        inp_texts = [
#            tokenizer.decode(b)
#            for b in dict_batch['input_ids']
#        ]
#    elif 'texts' in dict_batch:
#        inp_texts = [
#            t
#            for t in dict_batch['texts']
#        ]
#    else:
#        raise NotImplemented
#
#    concat_texts = [
#        a + b.replace('</s>', '')
#        for a, b in zip(inp_texts, batch_out)
#    ]
#    return tok_fn(concat_texts), batch_raw


def mcts_generalist_infer(step, K, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in

    breakpoint()

    #from rl import batch_tokenize
    #tok_fn = partial(batch_tokenize, config, tokenizer)

    #query_key = config.get('collate__query_key', 'query')
    #batch_out = res_fn(config, models, batch_in)
    #
    #root = Node('prompt', batch_in['input'][0])

    #while True:
    #    terminate = ['\\boxed' in o for o in batch_out]
    #    if any(terminate): break
    #    for i, out_str in enumerate(batch_out):
    #        search_uri = 'mabowdor'
    #        search_Q = batch_raw[i][query_key]
    #        tool_map = {
    #            'SEARCH': partial(
    #                search_mux, search_uri, search_Q
    #            ),
    #            'COMPUTE': sympy_solver
    #        }
    #        if not has_any_captured(out_str, tool_map):
    #            tool_res = ''
    #        else:
    #            _, tool_res = tool_invoke(out_str, tool_map)
    #            if isinstance(tool_res, ToolError):
    #                tool_res = multihop_err1()
    #            else:
    #                tool_res = multihop_results1(tool_res)
    #        batch_out[i] = out_str.replace('</s>', '') + tool_res
    #    breakpoint()
    #    batch_in = concat(tok_fn, batch_in, batch_out)
    #    batch_out = res_fn(config, models, batch_in)
    #    breakpoint()

    #rewards = rwd_fn(config, batch_in, batch_out, models,
    #    **get_cfg_json(config, 'reward_args', {})
    #)
    #if log_fn:
    #    log_fn(locals(), **get_cfg_json(config, 'log_args', {}))


if __name__ == '__main__':
    pass
