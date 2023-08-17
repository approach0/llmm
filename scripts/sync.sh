while true; do
	rsync -zav --delete ./output/MATH rsync://rsyncclient@approach0.xyz/data
	sleep 3
done
