#!/usr/bin/env python3
"""
Operator as the press detector (no ROS). Docs 6.63.

Grip-point geometry misses arm_d presses (32/150, docs 6.61), but the E-stop
latch is visible by eye. Keys, each followed by Enter (works over ssh):
  Enter / w   pressed   -> next target
  n           skipped   -> next target
  q           stop      -> zeros, end of loop
A thread reads stdin into a queue; the bridge polls it once per step.
"""

import queue
import sys
import threading
import time

PRESSED, SKIP, STOP = "pressed", "skipped", "stop"
_KEYS = {"": PRESSED, "w": PRESSED, "n": SKIP, "q": STOP}


def parse(line):
    """Keyboard line -> event, or None for an unknown key."""
    return _KEYS.get(line.strip().lower())


class OperatorInput:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self.q = queue.Queue()
        self.alive = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while True:
            line = self.stream.readline()
            if line == "":                       # EOF: no terminal
                self.alive = False
                return
            ev = parse(line)
            if ev is None:
                print(f"[operator] unknown key {line.strip()!r}: Enter/w = pressed, "
                      f"n = skipped, q = stop", flush=True)
                continue
            self.q.put((ev, time.time()))

    def poll(self):
        """(event, wall time) or None; non-blocking."""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


def selftest():
    import io
    assert parse("\n") == PRESSED and parse(" W \n") == PRESSED
    assert parse("n\n") == SKIP and parse("q\n") == STOP and parse("x\n") is None
    op = OperatorInput(io.StringIO("\nx\nn\nq\n"))
    op.th.join(1.0)
    evs = []
    while (e := op.poll()) is not None:
        evs.append(e[0])
    assert evs == [PRESSED, SKIP, STOP], evs
    assert not op.alive
    print("operator_input: 3/3 tests OK")


if __name__ == "__main__":
    selftest()
