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
import multiprocessing as mp
from multiprocessing import set_start_method, shared_memory, Lock


try:
    set_start_method('spawn')
except RuntimeError:
    pass

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

shm = []
scale = 0.3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tno_DB = 1
frame_check_interval = 8
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
        # DB_name = "test_faissDB_"+ name
        faiss.write_index(index, f"{os.path.join(DB_path,DB_name)}.index")     
    return embeddings

# process = mp.Process(target=extract_ad, args=(result_queue,frames_queue, faiss_index,real_indices,indices,tno_DB))
def extract_ad(result_queue, frames_queue, faiss_index,real_indices,indices,tno_DB):
    print("Process start")
    nt = time.time()
    min_distance = 99999
    min_indice = -1
    min_idx = -1
    dinov2.eval()
    dinov2.to("cuda")
    # shm = shared_memory.SharedMemory(name=shm_name)
    # frames_buffer = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    nt = time.time()
    while True:
        if not frames_queue.empty():
            min_distance, min_indice = 1000, -1
            frames = frames_queue.get()
            # print(frames[1][1])
            if frames == "END":
                break
            local_real_indices = real_indices.get()
            local_indices = indices.get()
            cutindex = indices.get()
            for frame, idx in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input = feature_extractor(images=frame, return_tensors="pt").to("cuda")
                temp = dinov2(**input)
                output = temp.pooler_output.detach().cpu().numpy().squeeze()
                distances , local_indices = faiss_index.search(np.array([output],dtype=np.float32), 1)
                if distances[0][0] < min_distance:
                    min_distance = distances[0][0]
                    min_indice = int(local_indices[0][0]/tno_DB)+1
                    min_idx = idx
            firstframeIndex = local_real_indices[0]
            if min_distance < 500:
                cnt = 0
                for i in local_real_indices:
                    if min_idx > i:
                        firstframeIndex = i
                        # frameidx = local_indices[cnt]
                    cnt+=1
                result_queue.put([firstframeIndex, min_indice,frames[-1][1],cutindex])
    os._exit(0)

DB_name = "faissDB_cm"
faiss_index = faiss.read_index(f"{os.path.join(DB_path,DB_name)}.index")

def main():

    for input_file in input_files:
        global shm
        cap = cv2.VideoCapture(input_file)
        # local variable for shot boundary detection
        fnt = 0
        frames_for_SBD = []
        frame_chunksz = 30
        padd_sz = 29
        predictions = []
        scenes = []   
        slice_s = 0
        slice_e = 0
        prev_i = 0

        # local variable for extract ad
        frames = []
        flag = False
        firstframeIndex = 0
        ignore_scene_change_until = -10
        ignore_duration = 14
        manager = mp.Manager()
        result_queue = mp.Queue()
        frames_queue = mp.Queue()
        real_indices = []
        main_real_indices = mp.Queue()
        indices = []
        main_indices = mp.Queue()
        firstframelst = []
        CMinfo = []
        process = mp.Process(target=extract_ad, args=(result_queue,frames_queue, faiss_index, main_real_indices, main_indices,tno_DB))
        process.start()
        # Extract ad with shot boundary extraction 
        transnetv2.eval().cuda()
        with torch.inference_mode():
            print(f"Processing file: {os.path.basename(input_file)}")
            while cap.isOpened() :
                ret, frame = cap.read()
                if not ret and flag: # 동영상 프레임 더 이상 안들어오
                    # tensor_input = torch.from_numpy(np.array(frames_for_SBD, dtype=np.uint8)).permute(0,1,2,3).unsqueeze(0).cuda()
                    # single_frame_pred, a = transnetv2(tensor_input)
                    # single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
                    # predictions = (single_frame_pred[0] > 0.6).astype(np.uint8)
                    # indices = np.where(predictions ==1)[0] #+fnt-frame_chunksz
                    # real_indices = indices + fnt - frame_chunksz 

                    # for index, real_indice in enumerate(real_indices):
                    #     min_distance, min_indice = 1000, -1
                    #     if fnt-real_indice < frame_check_interval :
                    #         continue
                    #     if prev_i >= real_indice: 
                    #         continue
                    #     slice_s = slice_e+1
                    #     slice_e = real_indice
                    #     scenes.append([slice_s, slice_e])
                    #     min_distance, min_indice, min_idx = extract_ad(frames) # frame_index를 불러와라
                    #     firstframeIndex = real_indice
                    #     if len(CMinfo) != 0 and CMinfo[-1] == min_indice and firstframeIndex - firstframelst[-1] < ignore_duration * 30: # 중복 필터링
                    #         pass
                    #     elif min_distance < 500:
                    #         for i in real_indices:
                    #             if min_idx > i:
                    #                 firstframeIndex = i
                    #         ignore_scene_change_until = ignore_duration * 30
                    #         flag = False
                    #         CMinfo.append(min_indice)
                    #         firstframelst.append(firstframeIndex)       
                    #         print(f"프레임 {firstframeIndex}에서 광고 {min_indice} 검출   추출 시점: frmaeNo {fnt}")
                    #         prev_i = real_indice
                    #         break

                    #     prev_i = real_indice                
                    # scenes.append([slice_e+1,fnt-1])
                    break

                # while not result_queue.empty():
                while not result_queue.empty():
                    result = result_queue.get()
                    firstframeIndex = result[0]
                    min_indice = result[1]
                    fnt_before = result[2]
                    cutindex = result[3]
                    if result is not None:
                        if len(CMinfo) != 0 and CMinfo[-1] == min_indice and firstframeIndex - firstframelst[-1] < ignore_duration * 30:
                            pass
                        else:
                            ignore_scene_change_until = firstframeIndex + ignore_duration * 30
                            CMinfo.append(min_indice)
                            firstframelst.append(firstframeIndex)    
                            print(f"프레임 {firstframeIndex}에서 광고 {min_indice} 검출   추출 시점: frmaeNo {fnt} 입력 시점: frmaeNo {fnt_before}") 
                            frames = []
                            frames_for_SBD = []
                            # frames[cutindex:] = [[dummy_frame, -1] for _ in range(len(frames) - cutindex)]  
                            prev_i = firstframeIndex
                            flag = False
                
                if fnt > ignore_scene_change_until-frame_chunksz: #마지막 광고 검출 후 마지막 광고 첫 프레임 이후로 광고 길이만큼 안지났으면 검출 미시행
                    flag = True
                
                if flag:
                    fnt += 1
                    adframe = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    frames.append([adframe,fnt])
                    frame = cv2.resize(frame, (48, 27))
                    frames_for_SBD.append(frame) #2024.06.17 다른 코드로 백업 후 chatgpt 내의 shared_mem 코드 테스트
                    
                    #########################################shot boundary detection###########################################
                    if len(frames) == frame_chunksz: # 광고 검출을 위한 프레임 리스트의 사이즈가 설정한 크기가 되면
                        # copy to shared mem
                        # nt = time.time()
                        # if shm_flag:
                        #     frame_arr = np.array(frames)
                        #     shm = shared_memory.SharedMemory(create=True, size=frame_arr.nbytes)
                        #     tempbuf = np.ndarray(frame_arr.shape, dtype=frame_arr.dtype, buffer=shm.buf)
                        #     tempbuf.fill(0)
                        #     is_empty = np.all(tempbuf==0)
                        #     print(is_empty)
                        #     process = mp.Process(target=extract_ad, args=(result_queue,faiss_index,real_indices,indices,tno_DB, shm.name, frame_arr.shape, frame_arr.dtype))
                        #     process.start()
                        #     shm_flag = False
                        
                        # shot boundary detection
                        tensor_input = torch.from_numpy(np.array(frames_for_SBD[:frame_chunksz], dtype=np.uint8)).permute(0,1,2,3).unsqueeze(0).cuda()
                        single_frame_pred, a = transnetv2(tensor_input)
                        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
                        predictions = (single_frame_pred[0] > 0.8).astype(np.uint8)
                        indices = np.where(predictions ==1)[0] + 1
                        real_indices = indices + fnt - frame_chunksz
                        

                    #############################################Extract ad#####################################################
                        for index, real_indice in enumerate(real_indices):
                            min_distance, min_indice = 1000, -1
                            if fnt-real_indice < frame_check_interval :
                                continue
                            if prev_i >= real_indice: 
                                continue                        
                            # if slice_s == 0: # 첫 구간
                            #     dummy_frame = np.zeros_like(frames[0][0])
                            #     slice_s, slice_e = 1, real_indice
                            #     scenes.append([slice_s, slice_e])
                            #     min_distance, min_indice, min_idx = extract_ad(frames[0:(slice_e if slice_e <frame_check_interval else frame_check_interval)])
                            #     firstframeIndex = 1
                            #     if len(CMinfo) != 0 and CMinfo[-1] == min_indice and firstframeIndex - firstframelst[-1] < ignore_duration * 30: # 중복 필터링
                            #         pass
                            #     elif min_distance < 500 :
                            #         ignore_scene_change_until = ignore_duration * 30
                            #         flag = False
                            #         CMinfo.append(min_indice)
                            #         firstframelst.append(firstframeIndex)
                            #         print(f"프레임 {firstframeIndex}에서 광고 {min_indice} 검출3")
                            #         # cv2.imwrite("/mnt/Project/python/ExtractAd/_DB/test/"+f"CM{min_indice}_sim{min_distance}.png", min_frame)
                            #         print(firstframelst[-1])
                            #         # frames[indices[index]:] = [dummy_frame] * (len(frames)-indices[index])
                            #         frames[indices[index]:] = [[dummy_frame, -1] for _ in range(len(frames) - indices[index])]
                            #         prev_i = real_indice
                            #         break

                            else:
                                print("Current shot index:", indices[index],"Current shot boundary:",real_indice,"indice index:",index,"fnt:",fnt)
                                slice_s = slice_e+1
                                slice_e = real_indice
                                # print(real_indice)
                                scenes.append([slice_s, slice_e])
                                frames_queue.put(frames)
                                main_indices.put(indices)
                                main_indices.put(indices[index])
                                main_real_indices.put(real_indices)
                                # frame_arr = np.array(frames)
                                # shared_frames = np.ndarray(frame_arr.shape, dtype = frame_arr.dtype, buffer=shm.buf)
                                # np.copyto(shared_frames,frame_arr)
                                # print(nt)
                                # min_distance, min_indice, min_idx = extract_ad(frames) # frame_index를 불러와라
                                # extract_ad(result_queue,frames,faiss_index,real_indices,indices,tno_DB,fnt):
                                # nt = time.time()
                                # process = mp.Process(target=testfunction,args=(nt,))
                                # process = mp.Process(target=extract_ad, args=(result_queue,frames,faiss_index,real_indices,indices,tno_DB,fnt))
                                # process.start()
                            prev_i = real_indice              
                        
                        # fnt_lst = fnt_lst[frame_chunksz-padd_sz:]
                        # frames = []  # 마지막 padd_sz 프레임을 유지
                        # frames_for_SBD = []  # 마지막 padd_sz 프레임을 유지
                        frames = frames[frame_chunksz-padd_sz:]  # 마지막 padd_sz 프레임을 유지
                        frames_for_SBD = frames_for_SBD[frame_chunksz-padd_sz:]  # 마지막 padd_sz 프레임을 유지
                else: 
                    fnt+=1

        print("Detected shot:", scenes)
        print("Detected CM:", CMinfo)
        print("Each CM's start frame index:",firstframelst)
        frames_queue.put("END")
        process.join()
        process.close()


if __name__ == '__main__':
    # try:
    main()
    # except KeyboardInterrupt:
    #     if shm is not None:
    #         shm.close()
    #         shm.unlink()  # 공유 메모리 객체를 시스템에서 제거
    # finally:
    #     if shm is not None:
    #         shm.close()
    #         shm.unlink()  # 공유 메모리 객체를 시스템에서 제거