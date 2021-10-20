#!/bin/bash

q="$1"
generation="$2"

padded="$(sed -e :a -e 's/^.\{1,2\}$/0&/;ta' <<<"$generation")"
cores="64"

for c in $(seq 1 512); do
	i=$(( r * $cores + c ))

	ipadded="$(sed -e :a -e 's/^.\{1,3\}$/0&/;ta' <<<"$i")"
	if [ ! -f "max2/data/q$q/gen$padded/samples/$ipadded.flat" ]; then
		echo python3 generate-fixed.py "$q" "$generation" "$i"
	fi
done | xargs -d "\n" -n 1 -P "$cores" bash -c
