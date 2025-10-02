import socket, time, math, csv

HOST = "rp-f0b172.local"   # replace with your board IP
PORT = 5000

rp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rp.connect((HOST, PORT))
print("[DEBUG] Connected to Red Pitaya")

def send(cmd, wait=0.002, expect_reply=False):
    rp.send((cmd + "\n").encode())
    time.sleep(wait)
    if expect_reply:
        return rp.recv(65536).decode().strip()
    else:
        return None

def reset_integrator():
    send("PID:RST:I:CH1")

def set_pid(kp, ki, kd):
    print(f"[DEBUG] Setting PID: Kp={kp}, Ki={ki}, Kd={kd}")
    reset_integrator()
    send(f"PID:KP:CH1 {kp}")
    send(f"PID:KI:CH1 {ki}")
    send(f"PID:KD:CH1 {kd}")

def set_setpoint(v):
    send(f"PID:SET:CH1 {v}")
    print(f"[DEBUG] Setpoint={v}")

# -------- Oscilloscope capture ----------
def capture_trace(filename, dec=64, timeout=2.0):
    print("[DEBUG] Starting acquisition")
    send("ACQ:RST")
    send(f"ACQ:DEC {dec}")
    send("ACQ:DATA:FORMAT ASCII")
    send("ACQ:DATA:UNITS VOLTS")
    send("ACQ:START")
    time.sleep(1.0)   # let buffer fill (longer than 0.1s)

    # set trigger on CH1 positive edge at 50 mV
    send("ACQ:TRIG:LEV 0.05")
    send("ACQ:TRIG CH1_PE")

    # wait for trigger or force NOW
    t0 = time.time()
    while True:
        stat = send("ACQ:TRIG:STAT?", expect_reply=True)
        if stat == "TD":
            print("[DEBUG] Trigger detected")
            break
        if time.time() - t0 > timeout:
            print("[DEBUG] Trigger timeout, forcing NOW")
            send("ACQ:TRIG NOW")
            break
        time.sleep(0.05)

    # helper: read full response until newline terminator
    def read_data(cmd):
        rp.send((cmd + "\n").encode())
        chunks = []
        rp.settimeout(0.5)
        while True:
            try:
                chunk = rp.recv(4096).decode()
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith("\n"):
                break
        return "".join(chunks).strip()

    data1 = read_data("ACQ:SOUR1:DATA?")
    data2 = read_data("ACQ:SOUR2:DATA?")
    ch1 = [float(v) for v in data1.split(",") if v]
    ch2 = [float(v) for v in data2.split(",") if v]

    # sample interval
    dt = (1.0 / 125e6) * dec
    tspan = dt * min(len(ch1), len(ch2))
    print(f"[DEBUG] Samples: {len(ch1)} points, dt={dt:.2e}s, span={tspan:.3f}s")

    # save CSV
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "CH1_IN1", "CH2_OUT1"])
        for i in range(min(len(ch1), len(ch2))):
            w.writerow([i*dt, ch1[i], ch2[i]])

    print(f"[DEBUG] Saved trace to {filename}")


# -------- Demo sequences ------------
def demo1():
    print("=== Demo 1 ===")
    set_pid(0.30, 1.0, 0.0)
    set_setpoint(0.0); time.sleep(2)
    set_setpoint(0.2); time.sleep(4)
    set_setpoint(-0.1); time.sleep(4)
    capture_trace("demo1.csv")

def demo2():
    print("=== Demo 2 ===")
    set_pid(0.60, 2.0, 0.0)
    set_setpoint(0.0); time.sleep(2)
    set_setpoint(0.2); time.sleep(4)
    set_setpoint(-0.1); time.sleep(4)
    capture_trace("demo2.csv")

def demo3():
    print("=== Demo 3 ===")
    set_pid(0.55, 1.5, 0.02)
    set_setpoint(0.0); time.sleep(2)
    set_setpoint(0.2); time.sleep(4)
    set_setpoint(-0.1); time.sleep(4)
    capture_trace("demo3.csv")

def demo4():
    print("=== Demo 4 ===")
    set_pid(0.35, 0.8, 0.0)
    steps = 250
    for i in range(steps+1):
        v = 0.5*i/steps
        set_setpoint(v); time.sleep(0.02)
    steps = 350
    for i in range(steps+1):
        v = 0.5 - 0.7*i/steps
        set_setpoint(v); time.sleep(0.02)
    capture_trace("demo4.csv")

def demo5():
    print("=== Demo 5 ===")
    set_pid(0.40, 0.6, 0.0)
    T, dt = 10, 0.02
    N = int(T/dt)
    for k in range(N):
        t = k*dt
        v = 0.2*math.sin(2*math.pi*0.5*t)
        set_setpoint(v); time.sleep(dt)
    capture_trace("demo5.csv")

# -------- Main ------------
if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()
    demo5()
    rp.close()
    print("[DEBUG] Connection closed")
