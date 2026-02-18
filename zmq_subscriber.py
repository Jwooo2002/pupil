# test_sub.py
import zmq, time

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:6000")
sub.setsockopt_string(zmq.SUBSCRIBE, "")
time.sleep(0.1)   # 연결 안정화

print("=== SUBSCRIBER START ===")
while True:
    msg = sub.recv_json()
    print("SUB GOT:", msg)
