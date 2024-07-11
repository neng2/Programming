import torch
import subprocess
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import faiss
from transformers import AutoImageProcessor, AutoModel
# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
import random
import cv2
import sys
import re
import utils_video 
import gc
import argparse
import time
# feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
# model = ViTModel.from_pretrained('facebook/dino-vits16').to("cuda")
import warnings

# FutureWarning 경고 메시지를 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

input_files = [f"/mnt/Project/python/ExtractAd/_DB/test{i}.mp4" for i in range(1, 3)]

feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')
# def main(input_file):
for input_file in input_files:
    # model.eval()
    # model.to("cuda")


    input_path_CM = "/mnt/Project/python/ExtractAd/_DB/CMDB"
    output_path_CM = "/mnt/Project/python/ExtractAd/_DB_processed/CM"
    # output_path_CM = "/mnt/Project/python/ExtractAd/_DB/CMDB"
    output_path_Testset = "/mnt/Project/python/ExtractAd/_DB_processed/Testset"
    input_path_Source = "/mnt/Project/python/ExtractAd/_DB/Source"
    output_path_Source = "/mnt/Project/python/ExtractAd/_DB_processed/Source"
    DB_path = "/mnt/Project/python/ExtractAd/_DB_processed/Faiss"

    scale = 0.3

    def extract_lst(input_path,output_path):
        target_ext = ('mov','mp4','avi','asf','mkv','mxf','mpg','webm','ts','wmv')
        target_ext += tuple(ext.upper() for ext in target_ext)
        post_fix = ''
        file_list = []
        input_files = []
        output_files = []
        for ext in target_ext:
            file_list += glob.glob(os.path.join(input_path, '*.' + ext))
        file_list = sorted(file_list)

        for i in range(len(file_list)):
            file_name, ext = os.path.splitext(os.path.basename(file_list[i]))

            input_files.append(os.path.join(input_path, file_name + ext))
            output_files.append(os.path.join(output_path, file_name + post_fix + '.mp4'))
        return input_files, output_files
    # ffmpeg 명령 실행
    def run_ffmpeg(input_files, output_files, scale):
        for input_file, output_file in zip(input_files, output_files):
            command = [
                'ffmpeg', '-y', '-ss', '0.8', '-i', input_file,
                '-vf', f'scale=iw*{scale}:ih*{scale}', 
                '-vframes', '1', output_file.replace('.mp4', '_000001.png')
            ]
            subprocess.run(command, check=True)
    def run_ffmpeg_random(input_files, output_files, scale):
        fps = 30  # 비디오의 FPS를 알고 있다고 가정

        for input_file, output_file in zip(input_files, output_files):
            # 각 파일마다 current_frame을 0으로 초기화
            current_frame = 0

            # 랜덤 간격 결정 (예: 1에서 10 프레임 사이)
            frame_jump = random.randint(1, 10)
            current_frame += frame_jump

            # 해당 프레임의 시간 계산
            frame_time = current_frame / fps

            # 프레임 추출 명령 실행
            command = [
                'ffmpeg', '-y', '-ss', str(frame_time), '-i', input_file,
                '-vf', f'fps=1,scale=iw*{scale}:ih*{scale}', '-vframes', '1',
                output_file.replace('.mp4', f'_{current_frame}.png')
            ]
            subprocess.run(command, check=True)
    def run_ffmpeg_succesive(input_files, output_files, scale):
        for input_file, output_file in zip(input_files, output_files):
            command = [
                'ffmpeg', '-y', '-i', input_file, 
                '-vf', f'fps=1,scale=iw*{scale}:ih*{scale}',
                '-ss', '0', '-to', '0.9',
                output_file.replace('.mp4', '_%06d.png')
            ]

            subprocess.run(command, check=True)
    # 정렬함수
    def sort_key_by_num(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    # 이미지 전처리
    def preprocess_image(image_path):
        tensors = []

        for file in sorted(os.listdir(image_path), key=sort_key_by_num):
            file_path = os.path.join(image_path, file)
            print(file_path)
            image = Image.open(file_path).convert("RGB")
            input = feature_extractor(images=image, return_tensors="pt").to("cuda")
            tensors.append(input)
        return tensors
    # 이미지 특징 추출
    def extract_features(image_tensors, name):
        embeddings = []
        # with torch.no_grad():
        with torch.inference_mode():
            for tensor in image_tensors:
                outputs = model(**tensor)
                last_hidden_states = outputs.pooler_output.detach().cpu().numpy().squeeze()
                embeddings.append(last_hidden_states.tolist())
            
            embedding_arr = np.array(embeddings, dtype=np.float32)
            # print(embedding_arr.shape)
            d = embedding_arr.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embedding_arr)
            DB_name = "faissDB_"+ name
            faiss.write_index(index, f"{os.path.join(DB_path,DB_name)}.index")     
        return embeddings

# print(cv2.getBuildInformation())
# input_file = "/mnt/Project/python/ExtractAd/_DB/test1.mp4"

    # model.eval()
    # model.to("cuda")
    # input_files_CM, output_files_CM = extract_lst(input_path_CM, output_path_CM)
    # # run_ffmpeg(input_files_CM, output_files_CM, scale)
    # run_ffmpeg_succesive(input_files_CM, output_files_CM, scale)
    # cm_tensors = preprocess_image(output_path_CM)
    # features1 = extract_features(cm_tensors,"cm")
    # DB_name = "faissDB_cm"
    # faiss_index = faiss.read_index(f"{os.path.join(DB_path,DB_name)}.index")
    cap = cv2.VideoCapture(input_file)

    # 동영상 정보 추출
    frame_total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Shot split
    # nt = time.time()
    shotInfo = utils_video.ShotInfo(input_file, frame_total)
    # print("shot boundary detection elapsed time : ",time.time()-nt)
    sceneIndex = []
    # shotIndex2 = []
    for slice_idx, slice in enumerate(shotInfo.scene_list_adjusted):
        slice_s = slice[0]
        slice_e = slice[1]
        sceneIndex.append([slice_s,slice_e])
        # sceneIndex.append(slice_s)
    # print(shotIndex)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    scene_index = 0
    frame_interval = 1
    flag = False
    firstframe = True
    firstframeIndex = 0
    firstframelst = []
    firstframetimeline_lst = []
    CMinfo = []
    min_distance = 99999
    cnt = 0
    min_indices = 0
    ignore_scene_change_until = 0
    ignore_duration = 15
    temptimeline = 0
    model.eval()
    model.to("cuda")
    # input_files_CM, output_files_CM = extract_lst(input_path_CM, output_path_CM)
    # # run_ffmpeg(input_files_CM, output_files_CM, scale)
    # run_ffmpeg_succesive(input_files_CM, output_files_CM, scale)
    # cm_tensors = preprocess_image(output_path_CM)
    # features1 = extract_features(cm_tensors,"cm")
    # DB_name = "faissDB_cm"
    # faiss_index = faiss.read_index(f"{os.path.join(DB_path,DB_name)}.index")

    print(f"Processing file: {os.path.basename(input_file)}")
    with torch.inference_mode():
        nt = time.time()
        while True:
            # ret, frame = cap.read()
            ret = cap.grab()
            if not ret:
                break
            if scene_index >= len(sceneIndex):
                break
            if sceneIndex[scene_index][0] == 0 :
                sceneIndex[scene_index][0] = 1
            if frame_index == sceneIndex[scene_index][0]-1 :
                firstframeIndex = frame_index
                # print(frame_index)
                # print(scene_index)
                # print(sceneIndex[scene_index][0], sceneIndex[scene_index][1])
                firstframetimeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                scene_index += 1
                if firstframetimeline >= ignore_scene_change_until :
                    flag = True
            if flag:
                ret, frame = cap.retrieve()
                if firstframe:
                    temptimeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                    firstframe = False
                if not ret:
                    frame_index += 1
                    continue
                if frame_index % frame_interval ==0:
                    cnt += 1
                    # frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input = feature_extractor(images=frame, return_tensors="pt").to("cuda")
                    temp = model(**input)
                    output = temp.pooler_output.detach().cpu().numpy().squeeze()
                    distances , indices = faiss_index.search(np.array([output],dtype=np.float32), 1)
                    timeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                    if distances[0][0] < min_distance:
                        # if frame_index >= 1031 and frame_index <= 1104:
                        #     cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"frame{frame_index}_CM{indices[0][0]+1}  {timeline:0.1f}sec_sim{distances[0][0]}.png", frame)      

                        min_distance = distances[0][0]
                        min_indices = int(indices[0][0]/15)+1
                        originalminindices = indices[0][0]+1
                        progress = (frame_index / frame_total) * 100
                        # print(f"Processing frame ({progress:.2f}%)")
                        # print(f"maybe CM_{int(indices[0][0]/15)+1}, Similarity : {distances[0][0]} in main stream {timeline:0.2f} sec")
                if timeline - temptimeline >= 1.5 or frame_index == sceneIndex[scene_index-1][1]-1:
                    cnt = 0
                    temptimeline = 0
                    firstframe = True
                    flag = False
                    if min_distance < 500:
                        cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"CM{min_indices}_sim{min_distance}.png", frame)
                        firstframetimeline_lst.append(str(firstframetimeline))
                        CMinfo.append(min_indices)
                        ignore_scene_change_until = firstframetimeline + 14
                        firstframelst.append(firstframeIndex)
                        # cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/temp/"+f"frame{frame_index}_CM{indices[0][0]+1}_{originalminindices}_  {timeline:0.1f}sec_sim{distances[0][0]}.png", frame)      

                    min_distance = 99999

            frame_index += 1
        print(CMinfo)
        print(firstframelst)
        print(firstframetimeline_lst) # 수정
        print("\nExtract ad elapsed time : ",time.time()-nt)
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process video files")
#     parser.add_argument('--input_file', type=str, required=True, help="Path to the input video file")
#     args = parser.parse_args()

#     main(args.input_file)
 

        # if frame_index % frame_interval ==0:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     input = feature_extractor(images=frame, return_tensors="pt").to("cuda")
        #     temp = model(**input)
        #     output = temp.pooler_output.detach().cpu().numpy().squeeze()
        #     distances , indices = faiss_index.search(np.array([output],dtype=np.float32), 1)
        #     timeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        #     if distances[0][0] < 500:
        #         progress = (frame_index / frame_total) * 100
        #         print(f"Processing frame ({progress:.2f}%)")
        #         print(f"maybe CM_{indices[0][0]+1}, Similarity : {distances[0][0]} in main stream {timeline:0.2f} sec")
            

        # if frame_index % frame_interval ==0:
       

    
# input_files_CM, output_files_CM = extract_lst(input_path_CM, output_path_CM)
# run_ffmpeg(input_files_CM, output_files_CM, scale)
# cm_tensors = preprocess_image(output_path_CM)
# features1 = extract_features(cm_tensors,"cm")

# DB_name = "faissDB_cm"
# faiss_index = faiss.read_index(f"{os.path.join(DB_path,DB_name)}.index")
# # idx = 0
# # for testset in features3:
# #     idx+=1
# #     print(f"@@@@@@{idx}@@@@@@@@")
# #     distances , indices = faiss_index.search(np.array([testset],dtype=np.float32), 4)
# #     print(indices[0])
# #     print(distances[0])

# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_index = 0
# frame_interval = 1

# # 결과를 저장할 파일 열기
# with torch.inference_mode():
#     while True:
#         # ret, frame = cap.read()
#         ret = cap.grab()
#         if not ret:
#             break

#         if frame_index % frame_interval ==0:
            
#             ret, frame = cap.retrieve()
#             if not ret:
#                 frame_index += 1
#                 continue
#             # cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"Interval{frame_interval}_frame{frame_index}.png", frame)      
            
#             # frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             input = feature_extractor(images=frame, return_tensors="pt").to("cuda")
#             temp = model(**input)
#             output = temp.pooler_output.detach().cpu().numpy().squeeze()
#             distances , indices = faiss_index.search(np.array([output],dtype=np.float32), 1)
#             timeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
#             if distances[0][0] < 500:
#                 progress = (frame_index / total_frames) * 100
#                 print(f"Processing frame ({progress:.2f}%)")
#                 print(f"maybe CM_{indices[0][0]+1}, Similarity : {distances[0][0]} in main stream {timeline:0.2f} sec")
#                 cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"Interval{frame_interval}_frame{frame_index}_CM{indices[0][0]+1}  {timeline:0.1f}sec_sim{distances[0][0]}.png", frame)      
      
#         frame_index += 1

# cap.release()





##############################
##############################
##############################


# input, output파일 준비, ffmpeg 이용하여 프레임 추출, 이미지 변환 후 embedding 추출
# extract_lst -> run_ffmpeg_* -> preprocess_image -> extract_features
# 한번 추출해서 이미지 및 db화 한 이후에는 DB_name과 경로만 이용하여 접근

# input_files_CM, output_files_CM = extract_lst(input_path_CM, output_path_CM)
# input_files_CM, output_files_Testset = extract_lst(input_path_CM, output_path_Testset)
# input_files_Source, output_files_Source = extract_lst(input_path_Source, output_path_Source)

# # run_ffmpeg(input_files_CM, output_files_CM, scale)
# # run_ffmpeg(input_files_Source, output_files_Source, scale)

# # run_ffmpeg_succesive(input_files_CM, output_files_Testset, scale)
# # run_ffmpeg_random(input_files_Source, output_files_Source, scale)

# cm_tensors = preprocess_image(output_path_CM)
# test_tensors = preprocess_image(output_path_Testset)
# source_tensors = preprocess_image(output_path_Source)

# features1 = extract_features(cm_tensors,"cm")
# features2 = extract_features(source_tensors,"source")
# features3 = extract_features(test_tensors,"test")

# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# DB_name = "faissDB_cm"
# faiss_index = faiss.read_index(f"{os.path.join(DB_path,DB_name)}.index")
# idx = 0
# for testset in features3:
#     idx+=1
#     print(f"@@@@@@{idx}@@@@@@@@")
#     distances , indices = faiss_index.search(np.array([testset],dtype=np.float32), 4)
#     print(indices[0])
#     print(distances[0])


# n 프레임 간격으로 원본 영상에서 비교한다 쳤을때 CM DB의 첫 1~n프레임까지는 다 faiss db에 넣으면 첫프레임 찾기 가능
# 광고라고 판단됐을 때 광고라고 판단한 프레임이 faiss db의 1~n프레임중 어떤 프레임이랑 유사도가 제일 높은지 확인 (x 프레임이라 가정)
# total frame 세고 있고 현재까지 total 프레임이 a면 a-x 번째 프레임이 광고의 첫 프레임
# 그 후 total 프레임 a'를 계속 세는데 fps*15 + a-x 프레임 이하까지는 유사도 비교 x 
# ...