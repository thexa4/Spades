#!/bin/bash

set -euo pipefail

generation="$1"

padded="$(sed -e :a -e 's/^.\{1,2\}$/0&/;ta' <<<"$generation")"
samples="${SAMPLES:-1024}"

function sync_data {
	set -euo pipefail

	while read -r host; do
		sshhost="$(cut -d'/' -f2 <<<"$host")"
		echo "$sshhost"
		ssh -n $sshhost "mkdir -p spades/max2/data"
		rsync -vr $sshhost:spades/max2/data/ max2/data/
		ssh -n $sshhost "cd spades && git pull --ff-only"
		ssh -n $sshhost "rm -r spades/max2/data"
	done < max2/parallelhosts
}

#python="python3"
#if ! command -v python3 &> /dev/null; then
#	python="python"
#fi

sync_data

for q in 1 2; do
	for i in $(seq 1 $samples); do
		ipadded="$(sed -e :a -e 's/^.\{1,3\}$/0&/;ta' <<<"$i")"
		if [ ! -f "max2/data/q$q/gen$padded/samples/$ipadded.flat.gz" ]; then
			echo cd spades "&&" python3 "generate-fixed.py" "$q" "$generation" "$i"
		fi
	done
done | parallel --nice 17 --controlmaster --sshloginfile max2/parallelhosts --sshdelay 0.2 --line-buffer --progress --joblog "max2/data/gen$padded.joblog"

sync_data

#$python learn.py 1 $generation
#$python learn.py 2 $generation

#git add max2/models
#git commit -m "Generation $generation"
#git push