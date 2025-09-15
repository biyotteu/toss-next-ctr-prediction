import time

def stage(msg: str, i: int, n: int):
    print(f"\nSTAGE [{i}/{n}] {msg}")

class Timer:
    def __init__(self):
        self.t = time.time()
    def lap(self, tag: str = "lap"):
        dt = time.time() - self.t
        print(f"[TIMER] {tag}: {dt:.2f}s")
        self.t = time.time()