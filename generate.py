import torch


class Generater:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, text_list, **kwargs):
        tokens = self.tokenizer(text_list, return_tensors='pt')
        with torch.no_grad():
            result = self._generate(tokens, **kwargs)
        return result

    def _generate(self, inputs,
        max_length=128,
        top_k=0,
        top_p=0.9,
        temperature=0.9,
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
                print('i=0')
                logits, past_key_values = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                print(logits.shape)
                print(len(past_key_values))
                print(len(past_key_values[0]))
                quit()
            else:
                logits, past_key_values = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            logits = logits[:, -1, :]

            if i == 0:
                logits[:, self.tokenizer.eos_token_id] = -float("inf")

            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            for idx in range(batch_size):
                if not done[idx] and (
                    next_token[idx].item() == self.tokenizer.eos_token_id or i == max_length - 1
                ):
                    done[idx] = True
                    results[idx] = input_ids[idx, start_idx:].clone().cpu().tolist()  # type: ignore # noqa: E501

            if sum(done) == batch_size:
                break

            # update input ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)],
                dim=-1,
            )

        result_text = list(map(self.tokenizer.decode, results))
        return result_text
