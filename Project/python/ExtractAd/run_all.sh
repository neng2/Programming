#!/bin/bash

input_dir="/mnt/Project/python/ExtractAd/_DB/"
output_dir="/mnt/Project/python/ExtractAd/_DB_processed/"

# for i in {1..16}
for i in {1..2}
do
  input_file="${input_dir}test${i}.mp4"
  python temp.py --input_file "$input_file"
done

echo "All files processed."