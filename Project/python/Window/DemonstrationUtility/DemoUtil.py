import tkinter as tk
from tkinter import Checkbutton, IntVar, filedialog, messagebox
import subprocess
import win32api
import keyboard

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.processes = []
        self.root_path = None
        self.setup_ui()
        keyboard.add_hotkey('esc', self.stop_all_videos)
        # keyboard.add_hotkey('a', self.dumy)
    def setup_ui(self):
        video_descriptions = [
            "Description for Button 1",
            "Description for Button 2",
            "Description for Button 3",
            "Description for Button 4"
        ]

        for i in range(4):
            label = tk.Label(self.root, text=video_descriptions[i])
            label.grid(row=i*4, column=0, padx=10, pady=5, sticky="ew")
            var = IntVar()
            btn_text = f"Play Video {i+1}"
            play_button = tk.Button(self.root, text=btn_text, command=lambda i=i, var=var: self.handle_play_left(i+1, var))
            play_button.grid(row=i*4+1, column=0, padx=10, pady=5, sticky="ew")

            if i < 3:
                check = Checkbutton(self.root, text="16:9", variable=var)
                check.grid(row=i*4+1, column=1, padx=5, pady=5, sticky="w")

        spacer = tk.Label(self.root, text="", width=10)
        spacer.grid(row=0, column=2, rowspan=16)

        right_video_descriptions = [
            "Description for Button 5",
            "Description for Button 6",
            "Description for Button 7",
            "Description for Button 8"
        ]

        for i in range(4):
            label = tk.Label(self.root, text=right_video_descriptions[i])
            label.grid(row=i*4, column=3, padx=10, pady=10, sticky="ew")
            btn_text = f"Play Video {5+i}"
            play_button = tk.Button(self.root, text=btn_text, command=lambda i=i: self.handle_play_right(5+i))
            play_button.grid(row=i*4+1, column=3, padx=10, pady=5, sticky="ew")
        #17
        
        entry_video1 = tk.Entry(root, width=50, state='readonly')
        entry_video1.place(x=110,rely=0.92)
        browse1 = tk.Button(self.root, text="Video path", command=lambda: self.browse_video(entry_video1))
        browse1.place(x=15,rely=0.91,width=80)

        
    def browse_video(self, entry):
        # 사용자에게 폴더 선택 요청
        directory_path = filedialog.askdirectory(title='Select a directory')
        if directory_path:
            # Entry 위젯의 상태를 일시적으로 'normal'로 변경
            entry.config(state='normal')
            entry.delete(0, tk.END)
            entry.insert(0, directory_path)
            self.root_path = directory_path
            entry.config(state='readonly')  # 다시 읽기 전용 상태로 설정


    def get_monitors(self):
        monitors = win32api.EnumDisplayMonitors()
        return [(monitor[2][0], monitor[2][1], monitor[2][2], monitor[2][3]) for monitor in monitors]

    def play_video(self, video_path, monitor):
        left, top, right, bottom = monitor
        width = right - left
        height = bottom - top
        video_path = self.root_path+video_path
        print(video_path)
        proc = subprocess.Popen(['ffplay', '-autoexit', '-fs', '-x', str(width), '-y', str(height), '-left', str(left), '-top', str(top), video_path],shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.processes.append(proc)
        # print(f"Started {video_path} with PID {proc.pid}")
        

    def handle_play_left(self, n, check_var):
        if self.root_path is None:
            messagebox.showwarning("Warning", "Please specify a directory path before playing.")
        else:
            monitors = self.get_monitors()
            video_path1 = f"/test{n}.mp4"
            # video_path1 = root_path+video_path1
            video_path2 = f"/test{n}-1.mp4" if check_var.get() == 0 else f"/test{n}-2.mp4"
            # video_path2 = root_path+video_path2
            self.play_video(video_path1, monitors[0])
            self.play_video(video_path2, monitors[1])

    def handle_play_right(self, n):
        if self.root_path is None:
            messagebox.showwarning("Warning", "Please specify a directory path before playing.")
        else:
            monitors = self.get_monitors()
            self.play_video(f"/test{n}.mp4", monitors[0 if n != 8 else 1])

    def stop_all_videos(self):
        # print("test")
        # print("Attempting to stop all videos...")
        for proc in self.processes:
            if proc.poll() is None:  # Check if the process is still running
                proc.terminate()  # Terminate the process
                # print(f"Terminated {proc.pid}")
        self.processes.clear()  # Clear the list of processes

root = tk.Tk()
root.geometry("480x400")
root.resizable(False, False)
app = VideoPlayer(root)

root.mainloop()



