import threading
import time

def cpu_bound_task(x):
    result = sum(i * i for i in range(1000))
    print(f"Thread {x}: Result = {result}")

def main_threading():
    start_time = time.time()
    threads = []
    for i in range(4):  # 4개의 스레드 생성
        thread = threading.Thread(target=cpu_bound_task, args=(i,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print("Time taken with threading:", time.time() - start_time)

if __name__ == "__main__":
    main_threading()