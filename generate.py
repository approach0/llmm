import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, temperature=1.0,
    top_k=0, top_p=0.0):
    # filter by top-k min score
    if top_k > 0:
        # Remove all tokens with a less than top-k probability
        top_k_scores = torch.topk(logits, top_k).values
        indices_to_remove = (logits < top_k_scores[..., -1])
        logits[indices_to_remove] = -float("inf")
    # sample by probs
    if top_p > 0.0:
        batch_size = logits.size()[0]
        for b in range(len(logits)):
            sorted_logits, sorted_idx = torch.sort(logits[b], descending=True)
            den_probs = F.softmax(sorted_logits / temperature, dim=-1)
            cum_probs = torch.cumsum(den_probs, dim=-1)
            # remove a dim X if it associates to a draw with P(X) <= top_p
            indices_to_remove = cum_probs > top_p
            # shift the indices to ensure at least we keep the first token
            indices_to_remove[1:] = indices_to_remove[:-1].clone()
            indices_to_remove[0] = False
            # remove the tail
            indices_to_remove = sorted_idx[indices_to_remove]
            logits[b][indices_to_remove] = -float("inf")
    return logits


class Generater:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, text_list, **kwargs):
        tokens = self.tokenizer(text_list, return_tensors='pt')
        with torch.no_grad():
            result = self._generate(tokens, **kwargs)
        return result

    def _generate(self, inputs, debug=False,
        max_length=128, top_k=40, top_p=0.9, temperature=0.9,
        **kwargs):
        # generate_length + 1 for EOS token
        max_length += 1

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        batch_size = input_ids.size(0)
        start_idx = input_ids.size(-1)

        past_key_values = None
        done = [False for _ in range(batch_size)]
        results = [None for _ in range(batch_size)]
        for i in range(max_length):
            if i == 0:
                logits, past_key_values = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True
                )
            else:
                logits, past_key_values = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            logits = logits[:, -1, :]
            if i == 0: # let us not stop at the first step
                logits[:, self.tokenizer.eos_token_id] = -float("inf")

            logits = top_k_top_p_filtering(logits,
                temperature=temperature, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits / temperature, dim=-1)
            # sample the next token from the filtered distribution
            next_token = torch.multinomial(probs, num_samples=1)

            if debug is True:
                print(f'Generating token#{i}: ',
                    list(map(self.tokenizer.decode, next_token)))

            # check termination, dealing each batch individually
            for b in range(batch_size):
                next_tok_of_b = next_token[b].item()
                if not done[b] and (
                    next_tok_of_b == self.tokenizer.eos_token_id or
                    i == max_length - 1
                ):
                    done[b] = True
                    results[b] = input_ids[b, start_idx:].tolist()
            if sum(done) == batch_size:
                break

            # append input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask, torch.ones((batch_size, 1),
                    dtype=torch.int, device=attention_mask.device
                )], dim=-1,
            )

        result_text = list(map(self.tokenizer.decode, results))
        return result_text
