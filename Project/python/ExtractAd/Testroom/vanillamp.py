from multiprocessing import Process, Queue
import time
import numpy as np

def worker(queue):
    while not queue.empty():
        item = queue.get()
        # 단순 계산으로 대체
        result = np.sum(item)

def vanilla_multiprocessing():
    queue = Queue()
    data = [np.random.rand(5000, 10000) for _ in range(10)]  # 큰 데이터 세트 생성

    start_time = time.time()
    processes = [Process(target=worker, args=(queue,)) for _ in range(2)]

    for item in data:
        queue.put(item)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print(f"Vanilla multiprocessing time with large data: {time.time() - start_time} seconds")

if __name__ == "__main__":
    vanilla_multiprocessing()