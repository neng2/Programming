import subprocess
import time

def cpu_bound_task():
    result = sum(i * i for i in range(1000))
    print(f"Python Main Task: Result = {result}")

def c_task():
    # C 프로그램 실행
    result = subprocess.run(["./parallel_task"], capture_output=True, text=True)
    print(result.stdout)

def main_hybrid():
    start_time = time.time()

    # C 작업을 병렬로 실행
    c_thread = subprocess.Popen(["/mnt/Project/python/ExtractAd/Testroom/parallel_task"])
    
    # 파이썬 메인 작업 계속 수행
    cpu_bound_task()

    # C 작업 완료 대기
    c_thread.wait()

    print("Time taken with Python main task + C integration:", time.time() - start_time)

if __name__ == "__main__":
    main_hybrid()