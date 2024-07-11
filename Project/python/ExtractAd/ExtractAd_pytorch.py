import torch
import subprocess
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import faiss
from transformers import AutoImageProcessor, AutoModel
import random
import cv2
import sys
import re
import utils_video 
import gc
import argparse
import time
import warnings
from transnetv2.transnetv2_pytorch import TransNetV2

# FutureWarning 경고 메시지를 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
##################### path ########################

input_files = [f"/mnt/Project/python/ExtractAd/_DB/test{i}.mp4" for i in range(5, 6)]
input_path_CM = "/mnt/Project/python/ExtractAd/_DB/CMDB"
output_path_CM = "/mnt/Project/python/ExtractAd/_DB_processed/CM"
# output_path_CM = "/mnt/Project/python/ExtractAd/_DB/CMDB"
output_path_Testset = "/mnt/Project/python/ExtractAd/_DB_processed/Testset"
input_path_Source = "/mnt/Project/python/ExtractAd/_DB/Source"
output_path_Source = "/mnt/Project/python/ExtractAd/_DB_processed/Source"
DB_path = "/mnt/Project/python/ExtractAd/_DB_processed/Faiss"

##################### global variable ########################

scale = 1 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tno_DB = 1
frame_check_interval = 45

#################### model definition ###################

feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
dinov2 = AutoModel.from_pretrained('facebook/dinov2-small')
transnetv2 = TransNetV2()
state_dict = torch.load("/mnt/Project/python/ExtractAd/transnetv2/transnetv2-pytorch-weights.pth")
transnetv2.load_state_dict(state_dict)

##################### function #######################

def video_to_tensor(video_path, frame_count=100, height=27, width=48):
    cap = cv2.VideoCapture(video_path)
    fnt = 0
    frames = []
    try:
        while cap.isOpened() :
            ret, frame = cap.read()
            fnt += 1
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV는 기본적으로 BGR 형식으로 이미지를 처리합니다.
            frames.append(frame)
    finally:
        cap.release()
        # return torch.from_numpy(np.array(frames))

    # 리스트를 NumPy 배열로 변환 후 토치 텐서로 변환
    frames = np.array(frames, dtype=np.uint8)
    frames = torch.from_numpy(frames).permute(0,1,2,3)  # CHW 형식으로 변경
    # frames = frames.float() / 255.0  # [0, 1] 범위로 정규화
    return frames.unsqueeze(0), fnt  # 배치 차원 추가 (NCHW)
    # return frames  # 배치 차원 추가 (NCHW)
def predictions_to_scenes(predictions, threshold: float = 0.6):
        
        # print(predictions.shape)
        predictions = (predictions > threshold).astype(np.uint8)
        np.set_printoptions(threshold=sys.maxsize)

        # with open("/mnt/Project/python/ExtractAd/_DB/compare.txt","w") as f:
        #     f.write(str(predictions))
        # print(predictions)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)
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
            '-vf', f'fps=10,scale=iw*{scale}:ih*{scale}',
            '-ss', '0', '-to', '1.5',
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
    dinov2.eval()
    dinov2.to("cuda")
    # with torch.no_grad():
    with torch.inference_mode():
        for tensor in image_tensors:
            outputs = dinov2(**tensor)
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

def extract_ad(frames):
    min_distance = 99999
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input = feature_extractor(images=frame, return_tensors="pt").to("cuda")
        temp = dinov2(**input)
        output = temp.pooler_output.detach().cpu().numpy().squeeze()
        distances , indices = faiss_index.search(np.array([output],dtype=np.float32), 1)
        if distances[0][0] < min_distance:
            min_distance = distances[0][0]
            min_indices = int(indices[0][0]/tno_DB)+1
            # originalminindices = indices[0][0]+1
            # progress = (frame_index / frame_total) * 100
    transnetv2.eval().cuda() # dino 모델 사용 끝났으니까 transnet 모드로 변경
    return min_distance, min_indices



# cm_tensors = preprocess_image(output_path_CM)
# features1 = extract_features(cm_tensors,"cm")

DB_name = "faissDB_cm"
faiss_index = faiss.read_index(f"{os.path.join(DB_path,DB_name)}.index")


# def main(input_file):
for input_file in input_files:
    
    cap = cv2.VideoCapture(input_file)

    # 동영상 정보 추출
    frame_total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # local variable for shot boundary detection
    fnt = 0
    frames_for_SBD = []
    frame_chunksz = 30
    padd_sz = 29
    predictions = []
    scenes = []
    ft = True
    slice_s = 0
    slice_e = 0
    prev_i = 0
    total_predictions = []

    # local variable for extract ad

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_index = 0
    scene_index = 0
    frame_interval = 1
    flag = False
    firstframe = True
    firstframeIndex = 0
    firstframelst = []
    firstframetimeline_lst = []
    CMinfo = []
    # min_distance = 99999
    cnt = 0
    ignore_scene_change_until = 0
    ignore_duration = 14
    extract_flag = True
    temptimeline = 0
    frames_for_check = []

    test_check=0
    # Extract ad with shot boundary extraction 
    transnetv2.eval().cuda()
    with torch.inference_mode():
        print(f"Processing file: {os.path.basename(input_file)}")
        nt = time.time()
        while cap.isOpened() :
            ret, frame = cap.read()
            fnt += 1
            if not ret: # 동영상 프레임 더 이상 안들어오면
                # tensor_input = torch.from_numpy(np.array(frames, dtype=np.uint8)).permute(0,1,2,3).unsqueeze(0).cuda()
                # # tensor_input = tensor_input.unsqueeze(0).cuda()
                # single_frame_pred, a = transnetv2(tensor_input)
                # single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
                # predictions = (single_frame_pred[0] > 0.6).astype(np.uint8)
                # indices = np.where(predictions ==1)[0]+fnt-frame_chunksz
                # if len(indices)==0:
                #     scenes.append([slice_e+1,fnt-1])
                # elif len(indices) > 0 :
                #     for i in indices:
                #         if prev_i == i :
                #             continue
                #         slice_s = slice_e+1
                #         slcie_e = i
                #         scenes.append([slice_s,slice_e])
                #     scenes.append([slice_e+1,fnt-1])
                break
            
            # frames.append(frame)
            frames.append(frame)
            frame = cv2.resize(frame, (48, 27))
            frames_for_SBD.append(frame)
            # print(len(frames))
            # print(len(frames_for_SBD))
            if len(frames) == frame_chunksz:
                if fnt - slice_e > ignore_duration * 30:
                    extract_flag = True
        
                if extract_flag == False : 
                    frames = frames[frame_chunksz-padd_sz:]  # 마지막 padd_sz 프레임을 유지
                    frames_for_SBD = frames_for_SBD[frame_chunksz-padd_sz:]  # 마지막 padd_sz 프레임을 유지
                    continue
                # print(test_check)
                # 첫 번째 청크를 모델에 전달
                # if len(frames_for_check) != 0 and frames_for_check < 45:
                #     frames_for_check.append(frames[0:45-frames_for_check])  # 이전에 개수 모잘라서 검사 안된 구간 존재하면 45개 채워넣기
                
                tensor_input = torch.from_numpy(np.array(frames_for_SBD[:frame_chunksz], dtype=np.uint8)).permute(0,1,2,3).unsqueeze(0).cuda()

                # 모델 실행
                single_frame_pred, a = transnetv2(tensor_input)
                single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
                predictions = (single_frame_pred[0] > 0.6).astype(np.uint8)
                total_predictions.append(predictions)
                indices = np.where(predictions ==1)[0]
                real_indices = indices + fnt - frame_chunksz
                print(real_indices)
                if len(indices)!=0  : 
                    print("Detected shot index:", indices , "Frame index:",real_indices,"Current frame num:",fnt) 
                dinov2.eval()
                dinov2.to("cuda") 
                if len(indices) == 0 and len(frames_for_check) != 0: # shot change 감지 X, 여기서 이게 걸리면 무조건 이전 구간에서 완료 안된 구간이 있고, 다음 shot change까지 45프레임 이상 남은거니까 여기서 광고 검출
                    frames_for_check.extend(frames[padd_sz:padd_sz+frame_check_interval-len(frames_for_check)])
                    min_distance, min_indice = extract_ad(frames_for_check)
                    
                    if min_distance < 500 and extract_flag:
                        CMinfo.append(min_indice)
                        firstframelst.append(firstframeIndex)
                        print(f"프레임 {firstframeIndex}에서 광고 {min_indice} 검출1")
                        cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"CM{min_indice}_sim{min_distance}.png", frames_for_check[0])
                        extract_flag = False
                    frames_for_check.clear() # 처리한 리스트 비워서 다시 len(frames_for_check) 0으로 세팅
                else : # 1개 이상의 shot change 감지
                    ft_flag = True
                    for index , real_indice in enumerate(real_indices):
                        indice_sz = len(real_indices)
                        if prev_i >= real_indice :
                            # if prev_i == real_indice:
                            #     frames_for_check.extend(frames[indices[index]:indices[index]+frame_check_interval-len(frames_for_check)])
                            #     min_distance, min_indice = extract_ad(frames_for_check)
                            #     frames_for_check.clear()
                            continue
                        # shot change가 감지는 됐는데 frames_for_check가 남아있을 때 (이전 이터레이션에서 넘어온거니까 검사 처음에 한번만 걸려야함 (index == 0 을 조건으로 하면 위에 prev_i 조건때문에 안됨))
                        if len(frames_for_check) != 0 and ft_flag:# index==0인 조건이랑 좀 나눠야할듯
                            if index!=0 and len(frames_for_check)+indices[index-1]-padd_sz < frame_check_interval: # 이전 단계에서 넘어온 후, 넘어온 프레임 + 다음 구간 시작까지 프레임 개수가 45개 이하면
                                frames_for_check.extend(frames[padd_sz:indices[index]])
                                min_distance, min_indice = extract_ad(frames_for_check)
                            elif index ==0 and len(frames_for_check)+indices[index]-padd_sz < frame_check_interval:
                                if indices[index]>padd_sz:
                                    frames_for_check.extend(frames[padd_sz:indices[index]])
                                min_distance, min_indice = extract_ad(frames_for_check)
                            else :
                                frames_for_check.extend(frames[padd_sz:padd_sz+frame_check_interval-len(frames_for_check)])
                                print(len(frames_for_check))
                                min_distance, min_indice = extract_ad(frames_for_check)

                            if len(CMinfo) != 0 and CMinfo[-1] == min_indice and firstframeIndex - firstframelst[-1] < ignore_duration * 30: # 중복 필터링
                                pass
                            elif min_distance < 500 and extract_flag:
                                CMinfo.append(min_indice)
                                firstframelst.append(firstframeIndex)
                                print(f"프레임 {firstframeIndex}에서 광고 {min_indice} 검출2")
                                cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"CM{min_indice}_sim{min_distance}.png", frames_for_check[0])
                                # print(prev_i, real_indice, index)
                                extract_flag = False
                            frames_for_check.clear()
                            ft_flag = False

                        # print(real_indice,fnt)
                        # print(fnt)
                        if ft: # 동영상 첫 구간
                            slice_e = real_indice
                            scenes.append([1,slice_e])
                            if slice_e > frame_check_interval : # 첫 구간 길이 45 이상일 때
                                min_distance, min_indice = extract_ad(frames[0:frame_check_interval])
                                firstframeIndex = 1
                            else : # 첫 구간 길이가 45가 안될 때
                                min_distance, min_indice = extract_ad(frames[0:slice_e])
                                firstframeIndex = 1
                            # print(len(frames[indices[index]:]))
                            if min_distance < 500 and extract_flag:
                                CMinfo.append(min_indice)
                                firstframelst.append(firstframeIndex)
                                print(f"프레임 {firstframeIndex}에서 광고 {min_indice} 검출3")
                                extract_flag = False
                            ft = False
                        else :
                            slice_s = slice_e+1
                            slice_e = real_indice
                            scenes.append([slice_s,slice_e])    
                            # print(len(frames[indices[index]:]))

                            if index + 1 < indice_sz: # 프레임 인터벌 내에서 검출된 boundary index가 두개 이상이고, 현재 index가 boundary list 의 끝이 아닐때

                                if len(frames[indices[index]:indices[index+1]]) <= frame_check_interval : # 검출 구간의 크기가 45 이하일때
                                    min_distance, min_indice = extract_ad(frames[indices[index]:indices[index+1]])
                                    firstframeIndex = real_indice
                                else:
                                    min_distance, min_indice = extract_ad(frames[indices[index]:indices[index]+frame_check_interval])
                                    firstframeIndex = real_indice

                            else : # 프레임 인터벌 내에서 검출된 boundary index가 한개거나, 현재 index가 boundary list의 끝일때

                                if len(frames[indices[index]:]) < frame_check_interval : # 현재 저장된 프레임의 마지막 shot change index~마지막 index까지 개수가 지정한 프레임 검사 개수에 못 미칠 경우
                                    frames_for_check.extend(frames[indices[index]:]) # 해당 프레임들 저장해놓고 나중에 활용
                                    firstframeIndex = real_indice

                                elif len(frames[indices[index]:]) >= frame_check_interval: # 현재 저장된 프레임의 마지막 shot change index~마지막 index까지 개수가 지정한 프레임 검사 개수 이상일 때
                                    min_distance, min_indice = extract_ad(frames[indices[index]:indices[index]+frame_check_interval])
                                    firstframeIndex = real_indice
                            if len(CMinfo) != 0 and CMinfo[-1] == min_indice and firstframeIndex - firstframelst[-1] < ignore_duration * 30: # 중복 필터링
                                pass
                            elif min_distance < 500 and extract_flag:
                                CMinfo.append(min_indice)
                                firstframelst.append(firstframeIndex)
                                print(f"프레임 {firstframeIndex}에서 광고 {min_indice} 검출4")
                                cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"CM{min_indice}_sim{min_distance}.png", frames[indices[index]])
                                print(prev_i, real_indice, index)
                                extract_flag = False
                        prev_i = real_indice
                        
                # 다음 청크를 위해 오버랩 고려하여 프레임 저장
                frames = frames[frame_chunksz-padd_sz:]  # 마지막 padd_sz 프레임을 유지
                frames_for_SBD = frames_for_SBD[frame_chunksz-padd_sz:]  # 마지막 padd_sz 프레임을 유지

    print("Detected shot:", scenes)
    # print("\nshot boundary detection elapsed time : ",time.time()-nt)

    print("Detected CM:", CMinfo)
    print("Each CM's start frame index:",firstframelst)
    # print("\nExtract ad elapsed time : ",time.time()-nt)

    # shotInfo = utils_video.ShotInfo(input_file, frame_total)
    # sceneIndex = []
    # # shotIndex2 = []
    # for slice_idx, slice in enumerate(shotInfo.scene_list_adjusted):
    #     slice_s = slice[0]
    #     slice_e = slice[1]
    #     sceneIndex.append([slice_s,slice_e])
        # sceneIndex.append(slice_s)
    # print(shotIndex)

    # dinov2.eval()
    # dinov2.to("cuda")
    # input_files_CM, output_files_CM = extract_lst(input_path_CM, output_path_CM)
    # # run_ffmpeg(input_files_CM, output_files_CM, scale)
    # run_ffmpeg_succesive(input_files_CM, output_files_CM, scale)
    # cm_tensors = preprocess_image(output_path_CM)
    # features1 = extract_features(cm_tensors,"cm")
    # DB_name = "faissDB_cm"
    # faiss_index = faiss.read_index(f"{os.path.join(DB_path,DB_name)}.index")

    # cap = cv2.VideoCapture(input_file)
    # print(f"Processing file: {os.path.basename(input_file)}")
    # with torch.inference_mode():
    #     nt = time.time()
    #     while True:
    #         # ret, frame = cap.read()
    #         ret = cap.grab()
    #         if not ret:
    #             break
    #         if scene_index >= len(scenes):
    #             break
    #         if scenes[scene_index][0] == 0 :
    #             scenes[scene_index][0] = 1
    #         if frame_index == scenes[scene_index][0]-1 :
    #             firstframeIndex = frame_index
    #             firstframetimeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    #             scene_index += 1
    #             if firstframetimeline >= ignore_scene_change_until :
    #                 flag = True
    #         if flag:
    #             ret, frame = cap.retrieve()
    #             if firstframe:
    #                 temptimeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    #                 firstframe = False
    #             if not ret:
    #                 frame_index += 1
    #                 continue
    #             if frame_index % frame_interval ==0:
    #                 cnt += 1
    #                 # frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 input = feature_extractor(images=frame, return_tensors="pt").to("cuda")
    #                 temp = dinov2(**input)
    #                 output = temp.pooler_output.detach().cpu().numpy().squeeze()
    #                 distances , indices = faiss_index.search(np.array([output],dtype=np.float32), 1)
    #                 timeline = float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    #                 if distances[0][0] < min_distance:
    #                     # if frame_index >= 1031 and frame_index <= 1104:
    #                     #     cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"frame{frame_index}_CM{indices[0][0]+1}  {timeline:0.1f}sec_sim{distances[0][0]}.png", frame)      

    #                     min_distance = distances[0][0]
    #                     min_indices = int(indices[0][0]/15)+1
    #                     originalminindices = indices[0][0]+1
    #                     progress = (frame_index / frame_total) * 100
    #                     # print(f"Processing frame ({progress:.2f}%)")
    #                     # print(f"maybe CM_{int(indices[0][0]/15)+1}, Similarity : {distances[0][0]} in main stream {timeline:0.2f} sec")
    #             if timeline - temptimeline >= 1.5 or frame_index == scenes[scene_index-1][1]-1:
    #                 cnt = 0
    #                 temptimeline = 0
    #                 firstframe = True
    #                 flag = False
    #                 if min_distance < 500:
    #                     firstframetimeline_lst.append(str(firstframetimeline))
    #                     CMinfo.append(min_indices)
    #                     ignore_scene_change_until = firstframetimeline + 14
    #                     firstframelst.append(firstframeIndex)
    #                     # cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/temp/"+f"frame{frame_index}_CM{indices[0][0]+1}_{originalminindices}_  {timeline:0.1f}sec_sim{distances[0][0]}.png", frame)      

    #                 min_distance = 99999

    #         frame_index += 1
    #     print(CMinfo)
    #     print(firstframelst)
    #     print(firstframetimeline_lst) # 수정
    #     print("\nExtract ad elapsed time : ",time.time()-nt)