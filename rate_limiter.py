# rate_limiter.py
import time

class WebSocketRateLimiter:
    def __init__(self, max_requests: int = 3, window_seconds: int = 600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = []

    def is_allowed(self) -> tuple[bool, int]:
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.window_seconds]
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(now)
            return True, 0
        oldest = self.timestamps[0]
        retry_after = int(self.window_seconds - (now - oldest)) + 1
        return False, retry_after
