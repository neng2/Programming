import zmq
from multiprocessing import Process
import time
import numpy as np

def zmq_receiver():
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://127.0.0.1:5555")

    for _ in range(10):
        message = receiver.recv_pyobj()
        result = np.sum(message)

def zmq_sender():
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://127.0.0.1:5555")
    data = [np.random.rand(5000, 10000) for _ in range(10)]

    start_time = time.time()
    for item in data:
        sender.send_pyobj(item)

    print(f"ZeroMQ sending time with large data: {time.time() - start_time} seconds")

def zmq_multiprocessing():
    receiver_process = Process(target=zmq_receiver)
    sender_process = Process(target=zmq_sender)

    receiver_process.start()
    sender_process.start()

    sender_process.join()
    receiver_process.terminate()
    receiver_process.join()

if __name__ == "__main__":
    zmq_multiprocessing()