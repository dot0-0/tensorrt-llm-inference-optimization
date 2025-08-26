import time

def tokens_per_second(num_tokens, seconds):
    if seconds <= 0:
        return 0.0
    return float(num_tokens) / float(seconds)

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
