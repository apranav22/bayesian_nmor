import socket
import time
import statistics

# --- Configuration ---
HOST = "rp-f0b172.local"   # replace with your PID/SCPI device hostname or IP
PORT = 5000                # SCPI port (5025 is common, check your device)
COMMAND = "*IDN?\n"        # simple SCPI query
N_REQUESTS = 1000          # number of iterations
# -----------------------

latencies = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    for _ in range(N_REQUESTS):
        start = time.perf_counter()
        s.sendall(COMMAND.encode())
        data = b""
        while not data.endswith(b"\n"):  # wait until newline terminator
            data += s.recv(4096)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # in ms

# Summary statistics
print(f"Count: {len(latencies)}")
print(f"Min: {min(latencies):.3f} ms")
print(f"Max: {max(latencies):.3f} ms")
print(f"Mean: {statistics.mean(latencies):.3f} ms")
print(f"Median: {statistics.median(latencies):.3f} ms")
if len(latencies) >= 100:
    print(f"95th percentile: {statistics.quantiles(latencies, n=100)[94]:.3f} ms")
