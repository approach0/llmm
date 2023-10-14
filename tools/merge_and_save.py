import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel


def merge_and_save(repo_or_path,
    cache_dir=None, adapter_name='default', output_dir='./output'):

    tokenizer = AutoTokenizer.from_pretrained(repo_or_path)
    print(tokenizer)

    peft_config = PeftConfig.from_pretrained(repo_or_path)
    base_model_path = peft_config.base_model_name_or_path
    print('Loading base model:', base_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path,
        torch_dtype=torch.float16, cache_dir=cache_dir)

    print('Loading peft model:', repo_or_path)
    peft_model = PeftModel.from_pretrained(base_model, repo_or_path,
        adapter_name=adapter_name, is_trainable=False)

    print('merging weights ...')
    model = peft_model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(merge_and_save)
