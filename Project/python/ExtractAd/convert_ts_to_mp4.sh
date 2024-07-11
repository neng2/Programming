#!/bin/bash

# 디렉토리 경로 설정
input_dir="/mnt/Project/python/ExtractAd/_DB/"
output_dir="/mnt/Project/python/ExtractAd/_DB/"

# 변환할 파일 리스트
for i in {1..16}
do
  input_file="${input_dir}test${i}.ts"
  output_file="${output_dir}test${i}.mp4"
  
  # FFmpeg 명령어 실행
  ffmpeg -i "$input_file" -c:v libx264 -c:a aac -strict experimental "$output_file"
done

echo "All TS files have been converted to MP4."
