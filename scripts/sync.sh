# export RSYNC_PASSWORD=takemymoney
while true; do
	rsync -zav --delete ./output/final_inference/* rsync://rsyncclient@approach0.xyz/data/basilisk
	sleep 3
done
