import PIL.ImageDraw
import PIL.ImageFile
import PIL.ImageFont
import easyocr
import os
import cv2
import time
import numpy as np
import PIL
import re
import pandas as pd
from symspellpy import SymSpell, Verbosity
from hangul_utils import split_syllable_char, split_syllables, join_jamos

from difflib import SequenceMatcher as SQ
from utils.preprocessing import imgCrop

from g2pk import G2p
from konlpy.tag import Mecab
from konlpy.tag import Okt
# Mecab 인스턴스 생성
mecab = Mecab('/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic')
okt = Okt()
# 경로 지정
img_path = "../app/easyexample/"
video_out = "../app/VideoTest/"
video_path = "../app/RECentral/"
logo_path = "logo/"
epg_path = "epg/"
entire_path = "entire/"

easyocr_path = "../app/easyexample/easyocr/"


wide = (56,42,1900,232)
left = (56,42,930,232)
right = (1500,42,1900,232)
# easyocr 세팅
reader = easyocr.Reader(['ko','en'],gpu=True)


# # 이미지 크롭 및 경로 셋팅
crop = imgCrop(img_path=img_path+"True/",logo_path=img_path+logo_path,epg_path=img_path+epg_path)
crop_true = imgCrop(img_path=img_path+"True/",logo_path=img_path+logo_path,epg_path=img_path+epg_path)
crop_false = imgCrop(img_path=img_path+"False/",logo_path=img_path+logo_path,epg_path=img_path+epg_path)
# crop.crop_logo_to_pic_all(-1)
# crop.crop_epg_to_pic_all(-1)
# # 한글 출력 위한 폰트 설정
fontpath = "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"
font = PIL.ImageFont.truetype(fontpath,40)
font_forresult = PIL.ImageFont.truetype(fontpath,60)

eng_pattern = re.compile(r'[A-Za-z]+')
eng_kor_pattern = re.compile(r"[^가-힣A-Za-z\s]")
title_pattern = re.compile(r"\d+\s*회|\d+\s*화|시즌\s*\d+")
phrases_to_remove = [
        r"하이라이트", r"스[페폐패퍠][셜설]?", r"신작(영화|드라마)?", r"연속방송", r"(재)|(재|재)", r"바로가기", r"다시보기"
    ]
g2p = G2p()


def process_korean_string(input_str):
    #공백 제거
    #EPG에서 자주 쓰이는 표현 제거 (제목 외)
    # 회차 정보 제거 ('n회', 'n화', '시즌n' 등)
    input_str = re.sub(title_pattern, "", input_str)
    input_str = input_str.replace(" ","")
    pattern = '|'.join(phrases_to_remove)
    input_str = re.sub(pattern, "", input_str)

    # 괄호 처리
    while True:
        open_paren = input_str.find('(')
        if open_paren != -1:
            close_paren = input_str.find(')', open_paren)
            if close_paren != -1:
                # 괄호가 제대로 닫힌 경우, 괄호와 괄호 안의 내용을 삭제
                input_str = input_str[:open_paren] + input_str[close_paren+1:]
            else:
                # # 괄호가 닫히지 않은 경우, 괄호 이후 모든 문자를 삭제
                # input_str = input_str[:open_paren]
                break
        else:
            break
    while True:
        open_paren = input_str.find('[')
        if open_paren != -1:
            close_paren = input_str.find(']', open_paren)
            if close_paren != -1:
                input_str = input_str[:open_paren] + input_str[close_paren+1:]
            else:
                # input_str = input_str[:open_paren]
                break
        else:
            break

    # 한글과 영어만 추출 (한글, 영어, 공백 포함)
    result = re.sub(eng_kor_pattern, "", input_str)
    
    # result = re.sub(r"\s+", "", extracted)
    return result 


def replace_english(text):
    # 영어 단어만 추출하는 정규 표현식
    pattern = eng_pattern

    # 문자열 내 모든 부분을 검사하고, 영어 부분만 한글 발음으로 변환
    def replace_match(match):
        word = match.group(0)
        return g2p(word)

    # 정규 표현식을 사용해 문자열 내의 모든 영어 단어를 한글 발음으로 변환
    result = pattern.sub(replace_match, text)
    return result

def process_korean_string_v2(input_str):
    #공백 제거
    #EPG에서 자주 쓰이는 표현 제거 (제목 외)
    
    input_str = re.sub(title_pattern, "", input_str)
    # input_str = input_str.replace(" ","")
    pattern = '|'.join(phrases_to_remove)
    input_str = re.sub(pattern, "", input_str)

    # 회차 정보 제거 ('n회', 'n화', '시즌n' 등)
    # 괄호 처리
    while True:
        open_paren = input_str.find('(')
        if open_paren != -1:
            close_paren = input_str.find(')', open_paren)
            if close_paren != -1:
                # 괄호가 제대로 닫힌 경우, 괄호와 괄호 안의 내용을 삭제
                input_str = input_str[:open_paren] + input_str[close_paren+1:]
            else:
                # 괄호가 닫히지 않은 경우, 괄호 이후 모든 문자를 삭제
                # input_str = input_str[:open_paren]
                break
        else:
            break
    while True:
        open_paren = input_str.find('[')
        if open_paren != -1:
            close_paren = input_str.find(']', open_paren)
            if close_paren != -1:
                input_str = input_str[:open_paren] + input_str[close_paren+1:]
            else:
                # input_str = input_str[:open_paren]
                break
        else:
            break

    # 한글과 영어만 추출 (한글, 영어, 공백 포함)
    result = re.sub(eng_kor_pattern, "", input_str)
    
    # result = re.sub(r"\s+", "", extracted)
    return result 



def top_values(data,index):
    # 리스트를 내림차순으로 정렬
    sorted_data = sorted(data, reverse=True)
    # 중복 제거
    unique_sorted_data = list(dict.fromkeys(sorted_data))
    # 가장 큰 값부터 index 번째 큰 값까지 추출
    return unique_sorted_data[:index]


# 각 부분 문자열을 문자열 2와 비교하는 함수
def compare_substrings(str1, str2, flag):

    results = []
    substrings = []
    if len(str1)>len(str2):

        len2 = len(str2)
        # 문자열 1을 문자열 2의 길이에 맞게 부분 문자열로 분할
        substrings = [str1[i:i+len2] for i in range(len(str1) - len2 + 1)]
        if flag == 1 :
            return substrings
        
        # 결과를 저장할 리스트
        for sub in substrings:
            # result = function(sub, str2)
            result = round(SQ(None, sub, str2).ratio()*100,2)
            results.append(result)
        if len(results) == 0 :
            results.append(0)
    
        max_value = max(results)  # 최대 결과 값 찾기
        max_index = results.index(max_value)  # 최대 값의 인덱스
        max_substring = substrings[max_index]  # 최대 값에 해당하는 부분 문자열
    else :
        result = round(SQ(None, str1, str2).ratio()*100,2) 
        max_value = result  # 최대 결과 값 찾기
        max_index = 0  # 최대 값의 인덱스
        max_substring = str1  # 최대 값에 해당하는 부분 문자열  
    # return results
    return results, substrings, max_value, max_substring, max_index

def compare_substrings_v2(str1, str2):
    str1_substrings = str1.split()
    str2_substrings = str2.split()
     # 리스트의 크기가 더 짧은 리스트가 str1_substrings가 되도록 스왑
    if len(str1_substrings) > len(str2_substrings):
        str1_substrings, str2_substrings = str2_substrings, str1_substrings

    # 결과 저장
    similarity_results = []
    # 최대값 저장
    max_similarity_results = []
    max_sub_pair = []

    # str1_substrings의 각 부분 문자열에 대해 str2_substrings의 모든 부분 문자열과 비교
    for sub1 in str1_substrings:
        max_sim = 0
        for sub2 in str2_substrings:
            # SequenceMatcher를 사용하여 두 부분 문자열 사이의 유사도 계산
            similarity = round(SQ(None, sub1, sub2).ratio()*100,2)
            # 결과를 직관적으로 표현하기 위해 문자열 형식으로 저장
            if similarity>max_sim:
                max_sim = similarity
                max_sub_pair = (sub1,sub2)
            similarity_results.append(((sub1, sub2), similarity))
        
        if max_sim!=0:
            max_similarity_results.append((max_sub_pair, max_sim))
        else :
            max_similarity_results.append(("유사한 단어가 존재하지 않습니다.", 0))
    return similarity_results, max_similarity_results

def compare_substrings_v3(str1_lst, str2_lst): #리스트로 넘겨받았을때
    josa_filter = ["JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC"]

     # 리스트의 크기가 더 짧은 리스트가 str1_substrings가 되도록 스왑
    if len(str1_lst) > len(str2_lst):
        str1_lst, str2_lst = str2_lst, str1_lst

    similarity_results = []
    max_similarity_results = []
    max_sub_pair = []

    for sub1 in str1_lst:
        if sub1[1] in josa_filter:
            str1_lst.remove(sub1)
            continue
        max_sim = 0
        for sub2 in str2_lst:
            if sub2[1] in josa_filter:
                str2_lst.remove(sub2)
                continue
            similarity = round(SQ(None, sub1[0], sub2[0]).ratio()*100,2)
            if similarity>max_sim:
                max_sim = similarity
                max_sub_pair = (sub1[0], sub2[0])
            similarity_results.append(((sub1[0], sub2[0]), similarity))
        
        if max_sim!=0:
            max_similarity_results.append((max_sub_pair, max_sim))
        else :
            max_similarity_results.append(("유사한 단어가 존재하지 않습니다.", 0))
    return similarity_results, max_similarity_results

def compare_substrings_v4(str1_lst, str2_lst): #리스트로 넘겨받았을때

     # 리스트의 크기가 더 짧은 리스트가 str1_substrings가 되도록 스왑
    if len(str1_lst) > len(str2_lst):
        str1_lst, str2_lst = str2_lst, str1_lst

    similarity_results = []
    max_similarity_results = []
    max_sub_pair = []
            
    for sub1 in str1_lst:
        max_sim = 0
        for sub2 in str2_lst:
            similarity = round(SQ(None, sub1, sub2).ratio()*100,2)
            if similarity>max_sim:
                max_sim = similarity
                max_sub_pair = (sub1,sub2)
            similarity_results.append(((sub1, sub2), similarity))
        
        if max_sim!=0:
            max_similarity_results.append((max_sub_pair, max_sim))
        else :
            max_similarity_results.append(("유사한 단어가 존재하지 않습니다.", 0))
    return similarity_results, max_similarity_results #<- 리턴 개수 수정
    

    # results = []
    # substrings = []
    # if len(str1)>len(str2):

    #     len2 = len(str2)
    #     # 문자열 1을 문자열 2의 길이에 맞게 부분 문자열로 분할
    #     substrings = [str1[i:i+len2] for i in range(len(str1) - len2 + 1)]
    #     if flag == 1 :
    #         return substrings
        
    #     # 결과를 저장할 리스트
    #     for sub in substrings:
    #         # result = function(sub, str2)
    #         result = round(SQ(None, sub, str2).ratio()*100,2)
    #         results.append(result)
    #     if len(results) == 0 :
    #         results.append(0)
    
    #     max_value = max(results)  # 최대 결과 값 찾기
    #     max_index = results.index(max_value)  # 최대 값의 인덱스
    #     max_substring = substrings[max_index]  # 최대 값에 해당하는 부분 문자열
    # else :
    #     result = round(SQ(None, str1, str2).ratio()*100,2) 
    #     max_value = result  # 최대 결과 값 찾기
    #     max_index = 0  # 최대 값의 인덱스
    #     max_substring = str1  # 최대 값에 해당하는 부분 문자열  
    # # return results
    # return results, substrings, max_value, max_substring, max_index

def videoocr(frame):
   
    # image = cv2.imread(img_path+path+img)
    compare_result = True
    img_logo = crop.crop_logo(frame)
    img_epg = crop.crop_epg(frame)

    bgrLower = np.array([180, 180, 180])    # 추출할 색의 하한(BGR)
    bgrUpper = np.array([255, 255, 255])    # 추출할 색의 상한(BGR)
    img_mask = cv2.inRange(img_epg, bgrLower, bgrUpper)
    img_epg = cv2.bitwise_and(img_epg, img_epg, mask=img_mask)
    
    epg_easyocr = reader.readtext(img_epg)
    logo_easyocr = reader.readtext(img_logo)
    
    # ocr 결과물 tuple 저장할 리스트 선언
    epg_ocr_result_seg = []
    logo_ocr_result_seg = []
    # tuple중 ocr 텍스트 관련 데이터만 append
    for i in epg_easyocr :  
        epg_ocr_result_seg.append(str(i[1]))
    for i in logo_easyocr :  
        logo_ocr_result_seg.append(str(i[1]))
    
    # 리스트 하나로
    ocr_result = ''.join(epg_ocr_result_seg)
    # 생성된 문자열 pre processing
    epg_ocr_result = process_korean_string(ocr_result)
    # logo도 동일 작업 수행
    ocr_result = ''.join(logo_ocr_result_seg)
    logo_ocr_result = process_korean_string(ocr_result)
    # comp_results = []
    # print("EPG : " + epg_ocr_result)
    # print("Logo : " + logo_ocr_result)
    # comp_results = compare_substrings(logo_ocr_result,epg_ocr_result,0)
    comp_results, sub_strings, max_value, max_substring, max_index = compare_substrings(logo_ocr_result,epg_ocr_result,0)
    # for i, string in enumerate(sub_strings):
    #     print(string+" "+str(comp_results[i])+"\n")
    # print("Similarity estimation with SQ: " + str(top_values(comp_results,3)))
    # print(top_values(comp_results,1)[0])
    if max_value>=30 :
        compare_result = True
    else : 
        compare_result = False
    return logo_ocr_result, epg_ocr_result, compare_result, max_substring, max_index, max_value


def imgocr(imglst, img_path, logo_result_path, epg_result_path):
    for index,img in enumerate(imglst):
        # image = cv2.imread(img_path+path+img)
        image = cv2.imread(img_path+img)

        img_logo = crop.crop_logo(image)
        img_epg = crop.crop_epg(image)

        bgrLower = np.array([180, 180, 180])    # 추출할 색의 하한(BGR)
        bgrUpper = np.array([255, 255, 255])    # 추출할 색의 상한(BGR)
        img_mask = cv2.inRange(img_epg, bgrLower, bgrUpper)
        img_epg = cv2.bitwise_and(img_epg, img_epg, mask=img_mask)
        
        start_time = time.time()
        epg_easyocr = reader.readtext(img_epg)
        logo_easyocr = reader.readtext(img_logo)
        easyocr_elapsed_time = time.time()-start_time
        
        # ocr 결과물 tuple 저장할 리스트 선언
        epg_ocr_result_seg = []
        logo_ocr_result_seg = []
        # tuple중 ocr 텍스트 관련 데이터만 append
        for i in epg_easyocr :  
            epg_ocr_result_seg.append(str(i[1]))
        for i in logo_easyocr :  
            logo_ocr_result_seg.append(str(i[1]))
        
        # 리스트 하나로
        ocr_result = ''.join(epg_ocr_result_seg)
        # 생성된 문자열 pre processing
        epg_ocr_result = process_korean_string(ocr_result)
        # logo도 동일 작업 수행
        ocr_result = ''.join(logo_ocr_result_seg)
        logo_ocr_result = process_korean_string(ocr_result)
        comp_results = []
        print("EPG : " + epg_ocr_result)
        print("Logo : " + logo_ocr_result)
        comp_results, max_value, max_substring, max_index = compare_substrings(logo_ocr_result,epg_ocr_result,0)
        print("Similarity estimation with SQ: " + str(top_values(comp_results,3)))
# 둘다 영어인 경우

def imgocr_v2(imglst, img_path, unit, logorange, eng_filter,file_name,case):
    th = 30
    with open(img_path+file_name+'_'+str(th)+'_(조사 필터링).txt', 'w', encoding='utf-8') as f:
    # with open(img_path+file_name+'_okt'+str(th)+'.txt', 'w', encoding='utf-8') as f:
        cnt_true = 0
        cnt_false = 0
        for index, img in enumerate(imglst):
            start_time = time.time()
            max_value = 0
            image = cv2.imread(img_path + img)
            img_logo = crop.crop_logo(image,logorange)
            img_epg = crop.crop_epg(image)

            bgrLower = np.array([180, 180, 180])
            bgrUpper = np.array([255, 255, 255])
            img_mask = cv2.inRange(img_epg, bgrLower, bgrUpper)
            img_epg = cv2.bitwise_and(img_epg, img_epg, mask=img_mask)

            epg_easyocr = reader.readtext(img_epg)
            logo_easyocr = reader.readtext(img_logo)

            epg_ocr_result_seg = []
            logo_ocr_result_seg = []
            for i in epg_easyocr:
                epg_ocr_result_seg.append(str(i[1]))
            for i in logo_easyocr:
                logo_ocr_result_seg.append(str(i[1]))

            ocr_result_e = ' '.join(epg_ocr_result_seg)
            ocr_result_l = ' '.join(logo_ocr_result_seg)
            # #testcode
            # epg_ocr_result = process_korean_string_v2(ocr_result_e)
            # logo_ocr_result = process_korean_string_v2(ocr_result_l)
            # if eng_filter == True:
            #     epg_ocr_result = replace_english(epg_ocr_result)
            #     logo_ocr_result = replace_english(logo_ocr_result)
            # epg_ocr_result_lst = mecab.pos(epg_ocr_result)
            # logo_ocr_result_lst = mecab.pos(logo_ocr_result)
            # comp_results, max_comp_results, comp_epg, comp_logo = compare_substrings_v3(epg_ocr_result_lst, logo_ocr_result_lst)
            
            # # print(*[x[0] for x in comp_epg])
            # # print(*[x[0] for x in comp_logo])
            # #testcode

            if unit == "sentence":
                epg_ocr_result = process_korean_string(ocr_result_e)
                logo_ocr_result = process_korean_string(ocr_result_l)
                if eng_filter == True:
                    epg_ocr_result = replace_english(epg_ocr_result)
                    logo_ocr_result = replace_english(logo_ocr_result)
                comp_results, sub_strings, max_value, max_substring, max_index = compare_substrings(epg_ocr_result, logo_ocr_result,0)
                
            elif unit == "word":
                epg_ocr_result = process_korean_string_v2(ocr_result_e)
                logo_ocr_result = process_korean_string_v2(ocr_result_l)
                if eng_filter == True:
                    epg_ocr_result = replace_english(epg_ocr_result)
                    logo_ocr_result = replace_english(logo_ocr_result)
                comp_results, max_comp_results = compare_substrings_v2(epg_ocr_result, logo_ocr_result)
                
            elif unit == "morphs":
                epg_ocr_result = process_korean_string_v2(ocr_result_e)
                logo_ocr_result = process_korean_string_v2(ocr_result_l)
                if eng_filter == True:
                    epg_ocr_result = replace_english(epg_ocr_result)
                    logo_ocr_result = replace_english(logo_ocr_result)
                # epg_ocr_result_lst = mecab.morphs(epg_ocr_result)
                # logo_ocr_result_lst = mecab.morphs(logo_ocr_result)
                # epg_ocr_result_lst = okt.morphs(epg_ocr_result)
                # logo_ocr_result_lst = okt.morphs(logo_ocr_result)
                epg_ocr_result_lst = mecab.pos(epg_ocr_result)
                logo_ocr_result_lst = mecab.pos(logo_ocr_result)
                comp_results, max_comp_results = compare_substrings_v3(epg_ocr_result_lst, logo_ocr_result_lst)

            elif unit == "nouns":
                epg_ocr_result = process_korean_string_v2(ocr_result_e)
                logo_ocr_result = process_korean_string_v2(ocr_result_l)
                if eng_filter == True:
                    epg_ocr_result = replace_english(epg_ocr_result)
                    logo_ocr_result = replace_english(logo_ocr_result)
                # epg_ocr_result_lst = mecab.nouns(epg_ocr_result)
                # logo_ocr_result_lst = mecab.nouns(logo_ocr_result)    
                epg_ocr_result_lst = mecab.nouns(epg_ocr_result)
                logo_ocr_result_lst = mecab.nouns(logo_ocr_result)    
                comp_results, max_comp_results = compare_substrings_v4(epg_ocr_result_lst, logo_ocr_result_lst)

            f.write("EPG OCR processing result : " + epg_ocr_result + '\n')
            # text = epg_ocr_result
            # f.write("가공 후 문자열(epg) 기준\n")
            # morphs = mecab.morphs(text)
            # f.write("형태소 단위로 분석된 텍스트: " + str(morphs) + '\n')
            # nouns = mecab.nouns(text)
            # f.write("추출된 명사: " + str(nouns) + '\n')
            f.write("Logo OCR processing result : " + logo_ocr_result + '\n')
            # text = logo_ocr_result
            # f.write("가공 후 문자열(epg) 기준\n")
            # morphs = mecab.morphs(text)
            # f.write("형태소 단위로 분석된 텍스트: " + str(morphs) + '\n')
            # nouns = mecab.nouns(text)
            # f.write("추출된 명사: " + str(nouns) + '\n')

            # for item in comp_results:
            #     sub_pair, sim = item
            #     # f.write(f"{sub_pair}: {sim}\n")
            if unit != "sentence" :
                for max_item in max_comp_results:
                    max_sub_pair, max_sim = max_item
                    if max_sim > max_value:
                        max_value = max_sim
                    if max_sim > 0:
                        f.write(f"최대 유사 쌍 : {max_sub_pair}, 유사도 : {max_sim}\n")
                    else:
                        f.write(str(max_sub_pair) + '\n')
            if max_value>=th :
                cnt_true += 1
                f.write("일치\n")
            else : 
                cnt_false += 1
                f.write("불일치\n")
            elapsed_time = time.time() - start_time
            f.write(str(elapsed_time) + 'sec\n')
            f.write("\n")
        f.write("True : "+str(cnt_true)+"False : "+str(cnt_false)+"\n")
        print("GT:",case,file_name,"True:",cnt_true,"False:",cnt_false)
        return cnt_true,cnt_false

        
  

def imgocr_to_pic(kindofimg,imglst,target_img_path,save_result_path):
    for index,img in enumerate(imglst):
        # image = cv2.imread(img_path+path+img)
        image = cv2.imread(target_img_path+img)
        
        image_easyocr = image

        start_time = time.time()
        easyocr_result = reader.readtext(image)
        easyocr_elapsed_time = time.time()-start_time

        # image_pillow 및 draw 객체 초기화
        image_pillow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pillow = PIL.Image.fromarray(image)
        draw = PIL.ImageDraw.Draw(image_pillow)
        for i in easyocr_result :
            x = i[0][0][0] 
            y = i[0][0][1] 
            w = i[0][1][0] - i[0][0][0] 
            h = i[0][2][1] - i[0][1][1]
            if i[2] > 0:
                draw.text((x,y+h-10), str(i[1]), (0,0,255), font=font)
                draw.rectangle(((x,y),(x+w,y+h)),outline=(0,255,0),width=2)
                draw.text((5,5),str(easyocr_elapsed_time)+"sec",(0,0,255), font=font)
                image_easyocr = np.array(image_pillow)
        
        cv2.imwrite(easyocr_path+save_result_path+str(index)+"_"+kindofimg+".tif",image_easyocr)


def imgocr_to_pic_v2(kindofimg,imglst,target_img_path,save_result_path):
    for index,img in enumerate(imglst):
        # image = cv2.imread(img_path+path+img)
        image = cv2.imread(target_img_path+img)
        
        image_easyocr = image

        start_time = time.time()
        easyocr_result = reader.readtext(image)
        easyocr_elapsed_time = time.time()-start_time

        # image_pillow 및 draw 객체 초기화
        image_pillow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pillow = PIL.Image.fromarray(image)
        draw = PIL.ImageDraw.Draw(image_pillow)
        for i in easyocr_result :
            x = i[0][0][0] 
            y = i[0][0][1] 
            w = i[0][1][0] - i[0][0][0] 
            h = i[0][2][1] - i[0][1][1]
            if i[2] > 0:
                draw.text((x,y+h-10), str(i[1]), (0,0,255), font=font)
                draw.rectangle(((x,y),(x+w,y+h)),outline=(0,255,0),width=2)
                draw.text((5,5),str(easyocr_elapsed_time)+"sec",(0,0,255), font=font)
                image_easyocr = np.array(image_pillow)
        
        cv2.imwrite(easyocr_path+save_result_path+str(index)+"_"+kindofimg+".tif",image_easyocr)
# 참고
# logo path img_path+logo_path+img_logo
# epg path img_path+epg_path+img_epg
# entire path img_path+entire_path+img_entire


# imgocr(crop.img_lst, crop.img_path, crop.logo_path, crop.epg_path)
# imgocr_to_pic("logo",crop.logo_lst,crop.logo_path,logo_path)
# imgocr_to_pic("epg",crop.epg_lst,crop.epg_path,epg_path)
def write_ocr_results_to_file(imgocr_v2, img_sets, settings):
    """
    This function calculates and writes the OCR results to a file.
    :param imgocr_v2: The OCR function used for processing.
    :param img_sets: List of image sets and their configurations.
    :param settings: List of OCR settings (unit, crop behavior, English filter application).
    """
    with open(img_path+'result_summary.txt', 'w', encoding='utf-8') as f:
        last_unit = None
        accuracies = []
        error_rates = []
        max_accuracy = 0
        bset_desc = ""

        for setting in settings:
            unit, crop_expansion, english_filter = setting
            total_true_as_true = 0
            total_false_as_false = 0
            total_true_as_false = 0
            total_false_as_true = 0
            total_cases = 0
            total_true_cases = 0
            total_false_cases = 0
            for img_set in img_sets:
                _crop, true_label = img_set
                desc = f"{unit} (logo crop범위 {'확장' if crop_expansion else '비확장'})" + \
                       ("_영발음필터적용" if english_filter else "")
                true, false = imgocr_v2(_crop.img_lst, _crop.img_path, unit, 
                                                        "wide" if crop_expansion else "left", english_filter, desc, true_label)
                if true_label == "True":
                    total_true_as_true += true
                    total_true_as_false += false
                    total_true_cases += len(_crop.img_lst)
                else:
                    total_false_as_true += true
                    total_false_as_false += false
                    total_false_cases += len(_crop.img_lst)
                # total_cases += len(_crop.img_lst)  # Assuming img_lst provides the number of cases in this set
            total_cases = total_true_cases + total_false_cases
            accuracy_true = total_true_as_true / total_true_cases
            accuracy_false = total_false_as_false / total_false_cases
            overall_accuracy = (total_true_as_true + total_false_as_false) / total_cases
            error_rate_true = total_true_as_false / total_true_cases
            error_rate_false = total_false_as_true / total_false_cases
            overall_error_rate = (total_false_as_true + total_true_as_false) / total_cases

            if overall_accuracy > max_accuracy:
                max_accuracy = overall_accuracy
                best_desc = desc

            accuracies.append(overall_accuracy)
            error_rates.append(overall_error_rate)

            # When changing units, calculate averages for the previous unit
            if last_unit is not None and last_unit != unit:
                avg_accuracy = sum(accuracies) / len(accuracies)
                avg_error_rate = sum(error_rates) / len(error_rates)
                f.write(f"{last_unit} 단위에서의 Accuracy 평균: {avg_accuracy*100:.2f}%\n")
                f.write(f"{last_unit} 단위에서의 Error rate 평균: {avg_error_rate*100:.2f}%\n\n")
                accuracies = []
                error_rates = []

            last_unit = unit

            f.write(f"Results for {desc}:\n")
            f.write(f"Accuracy for True cases: {accuracy_true*100:.2f}%\n")
            f.write(f"Error rate for True cases: {error_rate_true*100:.2f}%\n")
            f.write(f"Accuracy for False cases: {accuracy_false*100:.2f}%\n")
            f.write(f"Error rate for False cases: {error_rate_false*100:.2f}%\n")
            f.write(f"Overall Accuracy: {overall_accuracy*100:.2f}%\n")
            f.write(f"Overall Error Rate: {overall_error_rate*100:.2f}%\n\n")


        # Calculate and write averages for the last unit
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_error_rate = sum(error_rates) / len(error_rates)
            f.write(f"{unit} 단위에서의 Accuracy 평균: {avg_accuracy:.2f}%\n")
            f.write(f"{unit} 단위에서의 Error rate 평균: {avg_error_rate:.2f}%\n")
        f.write(f"\nThe best setting with the highest accuracy is '{best_desc}' with an accuracy of {max_accuracy*100:.2f}%.\n")

# Example usage
img_sets = [
    (crop_true, "True"), 
    (crop_false, "False")
]
settings = [
    ("sentence", False, False), 
    ("sentence", True, False),
    ("sentence", False, True),
    ("sentence", True, True),
    
    ("word", False, False), 
    ("word", True, False),
    ("word", False, True),
    ("word", True, True),
    
    ("morphs", False, False), 
    ("morphs", True, False),
    ("morphs", False, True),
    ("morphs", True, True),
    
    ("nouns", False, False), 
    ("nouns", True, False),
    ("nouns", False, True),
    ("nouns", True, True),
]

write_ocr_results_to_file(imgocr_v2, img_sets, settings)


# imgocr_v2(crop.img_lst, crop.img_path)
# imgocr_v2(imglst, img_path, unit, logorange, eng_filter,file_name,case):



# print("Total case : 69")
# wide = (56,42,1900,232)
# left = (56,42,930,232)
# right = (1500,42,1900,232)
# # 추후 왼쪽 비교 후 불일치시 right 봐서 일치불일치 다시 판단하도록 로직 수정 필요
# error = 0


# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"sentence",left,False,"문장단위 (logo crop범위 비확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"sentence",left,False,"문장단위 (logo crop범위 비확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"sentence",wide,False,"문장단위 (logo crop범위 확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"sentence",wide,False,"문장단위 (logo crop범위 확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"sentence",left,True,"문장단위 (logo crop범위 비확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"sentence",left,True,"문장단위 (logo crop범위 비확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"sentence",wide,True,"문장단위 (logo crop범위 확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"sentence",wide,True,"문장단위 (logo crop범위 확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# # ###################################################################################################################################
# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"word",left,False,"단어단위 (logo crop범위 비확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"word",left,False,"단어단위 (logo crop범위 비확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"word",wide,False,"단어단위 (logo crop범위 확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"word",wide,False,"단어단위 (logo crop범위 확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"word",left,True,"단어단위 (logo crop범위 비확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"word",left,True,"단어단위 (logo crop범위 비확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"word",wide,True,"단어단위 (logo crop범위 확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"word",wide,True,"단어단위 (logo crop범위 확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# # ###################################################################################################################################error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"morphs",left,False,"형태소단위 (logo crop범위 비확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"morphs",left,False,"형태소단위 (logo crop범위 비확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"morphs",wide,False,"형태소단위 (logo crop범위 확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"morphs",wide,False,"형태소단위 (logo crop범위 확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# # a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"morphs",left,True,"조사테스트","True")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"morphs",left,True,"형태소단위 (logo crop범위 비확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"morphs",left,True,"형태소단위 (logo crop범위 비확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"morphs",wide,True,"형태소단위 (logo crop범위 확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"morphs",wide,True,"형태소단위 (logo crop범위 확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# # ###################################################################################################################################error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"nouns",left,False,"명사단위 (logo crop범위 비확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"nouns",left,False,"명사단위 (logo crop범위 비확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"nouns",wide,False,"명사단위 (logo crop범위 확장)","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"nouns",wide,False,"명사단위 (logo crop범위 확장)","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")


# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"nouns",left,True,"명사단위 (logo crop범위 비확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"nouns",left,True,"명사단위 (logo crop범위 비확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

# error = 0
# a,b = imgocr_v2(crop_true.img_lst, crop_true.img_path,"nouns",wide,True,"명사단위 (logo crop범위 확장)_영발음필터적용","True")
# c,d = imgocr_v2(crop_false.img_lst, crop_false.img_path,"nouns",wide,True,"명사단위 (logo crop범위 확장)_영발음필터적용","False")
# error = b+c
# errorrate = error/69
# print("Error:",error,f"error rate: {errorrate:.2f}")

###################################################################################################################################


#True : 63False : 0  True : 0False : 6  -> GT
#True : 30False : 33 True : 1False : 5  -> 문장단위 (logo crop범위 비확장) 34/69
#True : 25False : 38 True : 1False : 5  -> 문장단위 (logo crop범위 확장) 39/69
#True : 31False : 32 True : 1False : 5  -> 문장단위 (logo crop범위 비확장)_영발음필터적용 33/69
#True : 27False : 36 True : 1False : 5  -> 문장단위 (logo crop범위 확장)_영발음필터적용 37/69

#True : 47False : 16 True : 0False : 6  -> 단어단위 (logo crop범위 비확장) 16/69
#True : 49False : 14 True : 0False : 6  -> 단어단위 (logo crop범위 확장) 14/69
#True : 48False : 15 True : 0False : 6  -> 단어단위 (logo crop범위 비확장)_영발음필터적용 15/69
#True : 50False : 13 True : 0False : 6  -> 단어단위 (logo crop범위 확장)_영발음필터적용 13/69

#True : 52False : 11 True : 1False : 5  -> 형태소단위 (logo crop범위 비확장) 12/69
#True : 55False : 8  True : 2False : 4  -> 형태소단위 (logo crop범위 확장) 10/69
#True : 53False : 10 True : 1False : 5  -> 형태1소단위 (logo crop범위 비확장)_영발음필터적용 11/69
#True : 56False : 7  True : 2False : 4  -> 형태소단위 (logo crop범위 확장)_영발음필터적용  9/69

#True : 47False : 16 True : 1False : 5  -> 명사단위 (logo crop범위 비확장) 17/69
#True : 48False : 15 True : 2False : 4  -> 명사단위 (logo crop범위 확장) 17/69
#True : 47False : 16 True : 1False : 5  -> 명사단위 (logo crop범위 비확장)_영발음필터적용 17/69
#True : 49False : 14 True : 2False : 4  -> 명사단위 (logo crop범위 확장)_영발음필터적용 16/69

# 단어 길이가 한글자인 경우 스킵해야할까
# 로고 오른쪽에 있는 경우와 다음 방영 프로그램 정보에 대해 다 읽을수 있으려면 식별 박스를 y좌표만 설정하고 x 좌표는 width 전체로 해야하지 않을까
#           -> 특히 ocn같은 영화채널에서 이런 양상 심함

# 1.
# imgocr_v2와 imgocr 의 일치 불일치 구분 후
# 샘플 이미지 데이터에서 ground-truth 설정해놓고
# ground-truth와 일치율이 얼마나 되는지 비교
# 문장제거/단어단위/형태소단위 / 명사단위

# 2. 
# 비디오에 대해서
# 최대 유사쌍 글씨 색다르게 표시해서
# 각각 유사도 얼마나되어 유사하다고 판별되었는지 만들기

# 3. @ 다룰 이슈
# 홈쇼핑, 광고 어떻게 처리할것인가
# 홈쇼핑 -> 채널로?
# 광고 -> (광고중)다음 방영 프로그램 정보 로고와 다음 프로그램 EPG 데이터가 맞을 경우 일치 판정 
#                   -> 다음 방영 프로그램 정보 로고가 없을 경우에는?
#   Or -> 드라마나 예능 프로그램은 어떤 사람이 진행을 하거나 얘기를 하거나 하는 장면에 초점이 맞춰져 있기때문에
#         해당 scene이 유지되거나 그럴때 static scene인 빈도가 높은데
#         광고는 화면 전환이 매우 빈번하고 dynamic scene일 확률이 높음
#         이 특성을 이용하여 static scene일 경우에만 logo/epg 일치 불일치를 판별한다던가?
#                   ->(succsseive frame 간의 ssim 비교, 혹은 motion vector 비교)
#                           -> 이 방법같은 경우 전체 프레임 영역을 대상으로하면 동적정도가 배경이미지 등으로 인해 뭉개질때가 많아서 
#                              특정 영역을 설정해서 보는것이 바람직함
#         이 때 (광고중)다음 방영 프로그램 정보 로고와 다음 프로그램 EPG 데이터가 맞을 경우 일치 판정 이것과 5번에 있는 내용을 병합하여 적용하면
#         광고와 프로그램의 구분 및 EPG/logo comparsion 정확도를 향상시킬 수 있지 않을까

# 4. @ 다룰 이슈
# 로고와 같은 텍스트 데이터가 있는지 인식하는거..?
# 판별 불가능할 경우 알림 안가도록 스킵기능 넣으려면
# 판별 불가능할 경우를 케이스화해서 분류해야하는데
# 어떻게 분류할 것인지

# 5. @ 다룰 이슈
# 시간이 덜 채워저서 
# 현재 방영프로그램 - 다음 프로그램 EPG 데이터가 맞을 경우 &
# (광고중)다음 방영프로그램 정보 로고 - 현재 프로그램 EPG 데이터가 맞을 경우
# 예외 처리? -> 이미지 data 몇개 있음 잘 찾아봐라 정호야





#############################################알고리즘 수정 후 재활성화################################################################
# cap = cv2.VideoCapture(video_path+"test3.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(video_out+"output_video.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), 
#                       (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# # out = cv2.VideoWriter(video_out+'output.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 30, (int(cap.get(3)), int(cap.get(4))))
# frame_idx = 0  # 프레임 인덱스 초기화
# frame_skip = 30  # 30 프레임마다 한 번씩 OCR 수행

# transparent_layer = PIL.Image.new('RGBA', (int(cap.get(3)), int(cap.get(4))), (0, 0, 0, 0))
# draw = PIL.ImageDraw.Draw(transparent_layer)
# results = []
# # 동영상의 총 프레임 수
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# logo_x,logo_y,logo_xend,logo_yend = 56,42,600,232
# epg_x,epg_y,epg_xend,epg_yend = 666,848,1522,932
# outline = (0,0,0,0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # if not cap.grab():
#     #     break
#     # 현재 프레임의 위치를 백분율로 표시
#     progress = (frame_idx / total_frames) * 100
#     # print(f"Processing frame {frame_idx}/{total_frames} ({progress:.2f}%)")
#     # 일정 프레임마다 OCR 수행
#     if frame_idx % frame_skip == 2:
#         # ret, frame = cap.retrieve()
#         # if not ret:
#         #     continue
#         draw.rectangle([(0, 0), transparent_layer.size], fill=(0, 0, 0, 0))  # 투명 레이어 클리어
#          # 레이어를 클리어하고 새로운 OCR 정보로 업데이트
#         logo_ocr_result, epg_ocr_result, compare_result, max_substring, max_index, max_value = videoocr(frame)
#         if compare_result == True:
#             compare_txt = "일치"
#         else :
#             compare_txt = "불일치"
#         # for (bbox, text, prob) in results:
#         #     (top_left, _, bottom_right, _) = bbox
#         #     top_left = (int(min(top_left[0], bottom_right[0])), int(min(top_left[1], bottom_right[1])))
#         #     bottom_right = (int(max(top_left[0], bottom_right[0])), int(max(top_left[1], bottom_right[1])))
#         #     draw.rectangle([top_left, bottom_right], outline="green", width=2)
#         #     draw.text((top_left[0], top_left[1] - 20), text, fill="green")

#         if compare_txt == "일치":
#             outline = (0,255,0,255)
#         else :
#             outline = (255,0,0,255)
#         ###########################################################################    
#         # logo부분 텍스트 그리기
#         # 부분 문자열 시작 및 끝 인덱스 찾기
#         start_idx = logo_ocr_result.find(max_substring) if max_substring else -1
#         end_idx = start_idx + len(max_substring) if max_substring else -1
#         x= logo_x
#         # max_substring 이전의 문자열 그리기 (있을 경우)
#         if start_idx > 0:
#             pre_text = logo_ocr_result[:start_idx]
#             draw.text((x, logo_yend), pre_text, "red", font=font)
#             # 너비 계산하여 x 좌표 업데이트
#             bbox = draw.textbbox((x, logo_yend), pre_text, font=font)
#             pre_width = bbox[2] - bbox[0]
#             x += pre_width

#         # max_substring 그리기 (비어있지 않을 경우)
#         if start_idx != -1 and len(max_substring) > 0:
#             draw.text((x, logo_yend), max_substring, outline, font=font)
#             bbox = draw.textbbox((x, logo_yend), max_substring, font=font)
#             max_width = bbox[2] - bbox[0]
#             x += max_width

#         # max_substring 이후의 문자열 그리기 (있을 경우)
#         if end_idx < len(logo_ocr_result) and end_idx != -1:
#             post_text = logo_ocr_result[end_idx:]
#             draw.text((x, logo_yend), post_text, "red", font=font)

#         # max_substring이 비었을 경우, 전체 텍스트를 기본 색상으로 그리기
#         if not max_substring:
#             draw.text((x, logo_yend), logo_ocr_result, "red", font=font)

#         # draw.text((logo_x,logo_yend),logo_ocr_result, "red",font=font)
#         ##############################################################################

#         draw.text((epg_x,epg_yend),epg_ocr_result,"green",font=font)
#         draw.rectangle(((logo_x,logo_y),(logo_xend,logo_yend)),outline=outline,width=2)
#         draw.rectangle(((epg_x,epg_y),(epg_xend,epg_yend)),outline=outline,width=2)
#         draw.text((5,5),compare_txt+" "+str(max_value)+"%",outline, font=font_forresult)


#      # PIL 이미지를 OpenCV 형식으로 변환
#     # frame = cv2.cvtColor(np.array(transparent_layer), cv2.COLOR_RGBA2BGRA)

#     transparent_overlay = cv2.cvtColor(np.array(transparent_layer), cv2.COLOR_RGBA2BGR)
#     combined_frame = cv2.addWeighted(frame, 0.5, transparent_overlay, 1, 0)
    
#     out.write(combined_frame)  # 결과 파일에 저장
#     frame_idx += 1

# cap.release()
# out.release()



