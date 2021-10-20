#!/bin/bash

q="$1"
generation="$2"

padded="$(sed -e :a -e 's/^.\{1,2\}$/0&/;ta' <<<"$generation")"
cores="16"
blocks="$((256 / cores))"

for r in $(seq 1 $blocks); do
	for c in $(seq 1 $cores); do
		i=$(( r * $cores + c ))

		ipadded="$(sed -e :a -e 's/^.\{1,3\}$/0&/;ta' <<<"$i")"
		if [ ! -f "max2/data/q$q/gen$padded/samples/$ipadded.flat" ]; then
			python generate-fixed.py "$q" "$generation" "$i" &
		fi
	done
	wait
done
