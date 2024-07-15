import os
import pytube # pip install pytube
from moviepy.editor import * # pip install moviepy
import pandas as pd
import requests # pip install requests
from bs4 import BeautifulSoup # pip install bs4
from pydub import AudioSegment #pip install pydub



os.chdir("C:/Users/BONITO/Programming/Project/python/CAE/AudioDataset/") # 작업 절대 경로 세팅
system_path = "C:/Users/BONITO/Programming/Project/python/CAE/"
file_name = ["unbalanced_train_segments.csv","eval_segments.csv","balanced_train_segments.csv"] #system_path 및 파일 경로 세팅

# for i in file_name:
#     data = pd.read_csv(file_path+file_name[i])
index=0
dfs = []
for name in file_name: #다른 csv파일 있을거 대비
    df = pd.read_csv(system_path+name, sep=r',(\s)',usecols=['YTID','start_seconds','end_seconds','positive_labels'])  #구분자를 공백 포함하도록 -> '\s 가 구분자 ( \s == ' ' )  #dup data로 발생하는 컬럼 추가 방지
    # Video labeling이 필요할 경우 'positive_labels' 컬럼 추가
    dfs.append(df)


""" #########################################################################################################
# data = pd.DataFrame()
# file_list = os.listdir(system_path)
# file_list_py = [file for file in file_list if file.endswith('csv')]
# for name in file_list_py:
#     file_path=system_path+name
#     data = pd.read_csv(file_path, sep=r',(\s)')  #구분자를 공백 포함하도록 -> '\s 가 구분자 ( \s == ' ' ) 
######################################################################################################### Scalability """ 

URL = 'https://www.youtube.com/watch?v='

speech_tag = "/m/09x0r"
for i,ID in enumerate(dfs[1].loc[:,'YTID']):
    if i<=645 : continue
    #print(i)
    if speech_tag in dfs[1]['positive_labels'][i] : # speech labeling 된 파일만
        try:
            # ID_list.append(URL+ID)  
            youtube = pytube.YouTube(URL+ID) #watch_url 세팅
            videos = youtube.streams.filter(progressive="True",file_extension="mp4").order_by('resolution') # 제일 낮은 화질로부터 음성 추출하게
            parent_dir = "C:/Users/BONITO/Programming/Project/python/CAE/AudioDataset/" # 데이터셋 저장 경로
            # for i in range(len(videos)) :
            #     print(i,":", videos[i])
            videos[0].download(parent_dir) # 경로로부터 다운로드
            video_name = videos[0].default_filename
            # audio_name = str(i+1)+"_"+ID+".mp3"
            audio_name = "temp.mp3"
            video = VideoFileClip(os.path.join(parent_dir,video_name))
            video.audio.write_audiofile(os.path.join(parent_dir,audio_name)) # mp3 변환후 저장
            audio = AudioSegment.from_mp3(audio_name) # mp3파일 리딩 (ffmpeg 필수)
            start_time = int(dfs[1]['start_seconds'][i])*1000
            end_time = int(dfs[1]['end_seconds'][i])*1000 # trim time interval 세팅
            trimmed_audio = audio[start_time:end_time] # mp3파일 trimming
            trimmed_audio.export(str(i+1)+"_"+ID+".mp3",format="mp3") # mp3로 추출후 저장
            video.close()
            if os.path.isfile(parent_dir+"/"+video_name):
                os.remove(parent_dir+"/"+video_name)
            if os.path.isfile(parent_dir+audio_name):
                os.remove(parent_dir+audio_name) # 유튜브 링크로부터 추출된 mp4, mp3파일 삭제
        except:
            print("Video "+str(i)+" is unavailable") # Expired watch_url 핸들링
    else : continue # speech 관련 영상 아닐 경우 skip

    