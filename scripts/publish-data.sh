rm -rf ~/.cache/huggingface/datasets
rm -rf output/final-*
rm -rf output/retrieval-augment-finetune/
python tools/data-gen-final-dataset.py output/mcts_explore_fulltopics_using_chatgpt/ --minrel 1 -o output/final-dataset.json
#python tools/data-gen-final-dataset.py output/mcts_explore_trees_using_chatgpt/collection/ --minrel 2 -o output/final-dataset-extra.json
#python tools/data-gen-merged-pairs.py merge output/final-*.json -o output/final-merged.json
cp output/final-dataset.json output/final-merged.json
python tools/data-gen-merged-pairs.py push approach0/retrieval-augment-finetune --train_path output/final-merged.json
python tools/inspect_output.py final_dataset approach0/retrieval-augment-finetune -o output
sh scripts/sync.sh
