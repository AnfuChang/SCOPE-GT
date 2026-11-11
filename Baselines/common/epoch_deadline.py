"""Wall-clock deadline for one training epoch, without a PyTorch dependency."""
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


class EpochDeadline:
    """Terminate a stalled training process and retain an auditable OOT record.

    A timer thread also covers a blocked iterator or CUDA call. Exiting the
    process, rather than raising in the timer thread, prevents training from
    continuing after the deadline. This context must enclose training only.
    """

    def __init__(self, seconds, *, seed, epoch, record_dir):
        self.seconds = float(seconds)
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError('epoch deadline must be finite and nonnegative')
        self.seed, self.epoch = int(seed), int(epoch)
        self.record_dir = Path(record_dir)
        self.timer = None
        self.cancelled = threading.Event()

    def __enter__(self):
        self.cancelled.clear()
        self.started = time.monotonic()
        if self.seconds:
            self.timer = threading.Timer(self.seconds, self._expire)
            self.timer.daemon = True
            self.timer.start()
        return self

    def _expire(self):
        # Some platform timers wake slightly early. Classify OOT only after
        # the monotonic deadline, and let a completed epoch cancel this wait.
        remaining = self.seconds - (time.monotonic() - self.started)
        while remaining > 0:
            if self.cancelled.wait(remaining):
                return
            remaining = self.seconds - (time.monotonic() - self.started)
        if self.cancelled.is_set():
            return
        record = dict(status='epoch_timeout', exit_code=124, seed=self.seed,
                      epoch=self.epoch, epoch_timeout_seconds=self.seconds,
                      observed_epoch_seconds=time.monotonic() - self.started,
                      scope='training_epoch')
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            target = self.record_dir / f'epoch_timeout_seed{self.seed}_epoch{self.epoch}.json'
            target.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            print('[OOT] ' + json.dumps(record), file=sys.stderr, flush=True)
        finally:
            os._exit(124)

    def __exit__(self, exc_type, exc_value, traceback):
        self.cancelled.set()
        if self.timer:
            self.timer.cancel()
            self.timer.join()
        return False
