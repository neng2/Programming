import os
import pytube # pip install pytube
from moviepy.editor import * # pip install moviepy
import pandas as pd
import requests # pip install requests없음
from bs4 import BeautifulSoup # pip install bs4
#해당 경로에서 csv 파일 가져오기
load_wb = load_workbook("C:/Users/kwons/Desktop/데이터셋_soft.xlsx", data_only=True)
#시트 이름으로 불러오기
load_ws = load_wb['Sheet1']

print('-----헤더를 제외한 엑셀의 모든 행과 열 저장-----')
all_values = []
for row in load_ws['A2':'B151']:
    row_value = []
    for cell in row:
        row_value.append(cell.value)
    all_values.append(row_value)
#print(all_values)
#print(all_values[0][0], all_values[0][1]) # 각각 id->new_filename에 이용, title-> query에 이용
#print(len(all_values)) # 150


URL = 'https://www.youtube.com/results'
for row in range(len(all_values)): # 0~149 까지 엑셀파일의 모든 title loop
    # HTTP request
    params = {'search_query': all_values[row][1] } # 엑셀파일의 'title'에 해당하는 문자열로 query
    response = requests.get(URL, params=params)

    # parsing
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    # 재생 시간이 7분 미만인 동영상을 찾아서 가져온다.
    video_index=0
    while True:
        running_time = soup.find_all(class_='video-time')[video_index].text # 동영상 재생시간
        splitted_time = running_time.split(":")

        if len(splitted_time) == 2: # 재생시간이 분, 초
            if int(splitted_time[0]) >= 7: # 재생시간이 7분 이상이면 다음 순서의 동영상을 가져온다.
                video_index += 1
                continue
            else :
                watch_url = soup.find_all(class_='yt-uix-sessionlink spf-link')[video_index]['href']
                break
        elif len(splitted_time) == 1 or len(splitted_time) == 3: # 재생시간이 초 또는 시,분,초
            video_index += 1 # 다음 순서의 동영상을 가져온다
            continue

    # 유튜브 가져오기
    youtube = pytube.YouTube("https://www.youtube.com" + watch_url) # 동영상 url
    videos = youtube.streams.all()

    '''
    # 다운받을 수 있는 스트리밍 종류
    for i in range(len(videos)) :
        print(i,":", videos[i])
    '''

    parent_dir = "C:/Users/kwons/dataset_soft" # 저장할 파일 경로
    videos[0].download(parent_dir) # mp4로 다운로드

    default_filename = videos[0].default_filename # 기존 mp4 파일이름

    # id format 값 변환 후 'id_format'.mp3 파일로 저장
    id = int(all_values[row][0]) # 엑셀파일의 id column값

    if int(id / 10) == 0 :
        id_format = "00" + str(id) # 한자리 수 id 일 경우 00x로 변환
    elif int(id / 100) == 0 :
        id_format = "0" + str(id) # 두자리 수 id 일 경우 0xx로 변환
    else:
        id_format = str(id) # 세자리 수 id 일 경우 id 값 그대로

    new_filename = id_format +".mp3" # mp3로 변환할 파일 이름

    video = VideoFileClip(os.path.join(parent_dir,default_filename)) # mp3로 변환할 기존 mp4파일 경로
    video.audio.write_audiofile(os.path.join(parent_dir,new_filename)) # mp3로 변환 후 저장할 경로

    video.close() # process를 끝내야 mp4파일을 지울 수 있다.

    if os.path.isfile(parent_dir + "/" + default_filename):
        os.remove(parent_dir + "/" + default_filename) # 기존 mp4 파일 지우기