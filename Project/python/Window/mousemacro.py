# import pyautogui as pg
# import time
# import keyboard
# import random
# from datetime import datetime

# # currentTime = datetime.now()
# # print('시작 시간 : ', currentTime, ', 현재 위치 : ', pyautogui.position())
# # while 1:
# #     recent = pyautogui.position()

# #     time.sleep(1)
# #     if(recent == pyautogui.position())  :
# #         height1 = random.randint(200,900)
# #         height2 = random.randint(200,900)
# #         print('현재 위치 : ', (recent))
# #         pyautogui.click(height1, height2, duration=2)
# #         print('변경 이동 위치 : ', pyautogui.position())
# #         time.sleep(5)
# while not keyboard.is_pressed("2"):
#     if keyboard.is_pressed("1"):
#         while True:
#             pg.click()
#             if keyboard.is_pressed("2"):
#                 break
import tkinter as tk
from tkinter import filedialog
import os

def browse_video(entry):
    # 파일 탐색기를 통해 비디오 파일 경로를 얻음
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mkv;*.avi")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def play_video(video_path):
    # ffplay를 사용해 비디오 재생
    os.system(f"ffplay '{video_path}'")

# GUI 초기화
root = tk.Tk()
root.title("Video Player")

# 비디오 파일 경로 입력 상자
entry_video1 = tk.Entry(root, width=50)
entry_video1.grid(row=0, column=1, padx=10, pady=10)
entry_video2 = tk.Entry(root, width=50)
entry_video2.grid(row=1, column=1, padx=10, pady=10)

# 파일 탐색 버튼
btn_browse1 = tk.Button(root, text="Specify video", command=lambda: browse_video(entry_video1))
btn_browse1.grid(row=0, column=2, padx=10, pady=10)
btn_browse2 = tk.Button(root, text="Specify video", command=lambda: browse_video(entry_video2))
btn_browse2.grid(row=1, column=2, padx=10, pady=10)

# 재생 버튼
btn_play = tk.Button(root, text="Play", command=lambda: [play_video(entry_video1.get()), play_video(entry_video2.get())])
btn_play.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()






