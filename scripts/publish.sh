copy_for_publish() {
    cp $1/config.json $2
    cp $1/generation_config.json $2
    cp $1/pytorch_model-*.bin $2
    cp $1/pytorch_model.bin.index.json $2

    cp $1/adapter_config.json $2
    cp $1/adapter_model.bin $2

    cp $1/special_tokens_map.json $2
    cp $1/tokenizer_config.json $2
    cp $1/tokenizer.model $2
    cp $1/tokenizer.json $2

    cp $1/*.ini $2
}

copy_for_publish output/finetune_phase2__mathy_lora.old ../azbert-lora/ckpt/
