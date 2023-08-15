import torch
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
from datasets import load_dataset
import bitsandbytes as bnb


class LlamaRewardModel(LlamaForCausalLM):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.reward_head = torch.nn.Linear(config.hidden_size, 1, bias=False)
        
    def forward(self, decoder_input, only_last=True):
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
        output = self.model.forward(
            input_ids=decoder_input,
            attention_mask=attention_mask, 
            return_dict=True,
            use_cache=False
            )
        
        if only_last:
            logits = self.reward_head(output.last_hidden_state[:, -1, :]).squeeze(-1)
        else:
            logits = self.reward_head(output.last_hidden_state).squeeze(-1)
        
        return logits


if __name__ == '__main__':
    #raw_train_data = load_dataset('json', encoding='utf-8',
    #    data_files='output/MATH-pairs.json', split="train", cache_dir='./cache')
    tokenizer = LlamaTokenizer.from_pretrained( 'lmsys/vicuna-7b-v1.5-16k')
    if not hasattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token

    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from trl import create_reference_model
    from trl.core import respond_to_batch

    config = PPOConfig(
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        #'output/tiny_llama',
        device_map="auto",
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        load_in_8bit=True,
        device_map="auto",
    )
    #model_ref = create_reference_model(model, num_shared_layers=6)

    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    #optimizer = torch.optim.SGD(model.parameters(), lr=3e-5)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=3e-5)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        #dataset=dataset,
        #data_collator=collator,
        optimizer=optimizer,
    )

    tokens = tokenizer(
        'I need ',
        return_tensors="pt",
        padding="longest",
        max_length=12,
        truncation=True,
    )

    i = model.pretrained_model.device
    print(i)
    question_tensors = tokens['input_ids'].to(i)
    print(question_tensors.device)
    response_tensors = respond_to_batch(model, question_tensors)
    print(response_tensors.device)
    print(tokenizer.decode(response_tensors[0]))

    rewards = [torch.tensor(1.0, device=i)]
    stats = ppo_trainer.step([question_tensors[0]], [response_tensors[0]], rewards)
    print(stats)
    quit()
    import pdb; pdb.set_trace()
    quit()
    critic_model = LlamaRewardModel.from_pretrained('output/tiny_llama', 'lmsys/vicuna-7b-v1.5-16k')

