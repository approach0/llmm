# export RSYNC_PASSWORD=takemymoney
while true; do
	rsync -zav --delete ./output/final-dataset rsync://rsyncclient@approach0.xyz/data/GCR
	sleep 3
done
