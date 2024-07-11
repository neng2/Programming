import mmap
import os
import numpy as np
from multiprocessing import Process
import time

def write_to_mmap():
    data = np.random.rand(5000, 10000).astype(np.float64)
    data_bytes = data.tobytes()

    with open("mmap_file", "wb+") as f:
        f.write(data_bytes)  # 파일 크기를 확장

    with open("mmap_file", "r+b") as f:
        mm = mmap.mmap(f.fileno(), len(data_bytes), access=2)  # 쓰기 권한
        mm.write(data_bytes)
        mm.close()
# 36  1 2 3 4 6 9 12 18 36
def read_from_mmap():
    time.sleep(1)  # 쓰기 프로세스가 데이터를 쓸 시간을 확보

    with open("mmap_file", "r+b") as f:
        mm = mmap.mmap(f.fileno(), os.path.getsize(f.name), access=1)  # 읽기 권한
        data = np.frombuffer(mm, dtype=np.float64).reshape(5000, 10000)
        print("Data sum from mmap:", np.sum(data))
        mm.close()

def mmap_multiprocessing():
    start_time = time.time()

    writer = Process(target=write_to_mmap)
    reader = Process(target=read_from_mmap)

    writer.start()
    reader.start()

    writer.join()
    reader.join()

    print(f"Mmap multiprocessing time with large data: {time.time() - start_time} seconds")

if __name__ == "__main__":
    mmap_multiprocessing()
