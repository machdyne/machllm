#!/bin/sh
# machllm: put .mlm models on an SD card's FAT partition, or build a card image.
#
#   tools/mkcard.sh /media/you/CARD  model1.mlm [NAME1.MLM] ...   copy to a mounted card
#   tools/mkcard.sh -i card.img SIZE_MB model.mlm ...              make an image (mtools)
#
# The firmware lists *.MLM in the root directory (8.3 names, upper case)
# and loads the first one at power-up; :load N picks another.
set -e
name83() { b=$(basename "$1" .mlm | tr 'a-z' 'A-Z' | tr -cd 'A-Z0-9_' | cut -c1-8); echo "$b.MLM"; }
if [ "$1" = "-i" ]; then
	img=$2; size=$3; shift 3
	rm -f "$img"; truncate -s "${size}M" "$img"
	mkfs.fat -F 32 -n MACHLLM "$img" > /dev/null
	for m in "$@"; do mcopy -i "$img" "$m" "::$(name83 "$m")"; echo "$(name83 "$m")"; done
	echo "write it with: dd if=$img of=/dev/sdX bs=4M conv=fsync"
else
	dst=$1; shift
	for m in "$@"; do cp "$m" "$dst/$(name83 "$m")"; echo "$dst/$(name83 "$m")"; done
	sync
fi
