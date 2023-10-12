set -e
rm -rf ~/.cache/huggingface/datasets
rm -rf output/final-*
rm -rf output/retrieval-augment-finetune/
python tools/data-gen-final-dataset.py output/mcts_explore_fulltopics_using_chatgpt/ -o output/final-dataset.json
python tools/data-gen-final-dataset.py output/mcts_explore_trees_using_chatgpt/collection/ -o output/final-dataset-extra.json
#cp output/final-dataset.json output/final-merged.json
python tools/data-gen-merged-pairs.py merge output/final-*.json -o output/final-merged.json
python tools/data-gen-merged-pairs.py push approach0/retrieval-augment-finetune --train_path output/final-merged.json
python tools/inspect_output.py final_dataset approach0/retrieval-augment-finetune -o output
sh scripts/sync.sh
