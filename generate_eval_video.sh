#!/bin/bash
let "max=$3-1"
for ((itr = 0; itr <= $max; itr++)); do
	ffmpeg -r 15 -start_number 0 -i images/$1_eval_on_$2_itr${itr}_%d.png -vcodec mpeg4 videos/$1_eval_on_$2_itr$itr.mp4
done
