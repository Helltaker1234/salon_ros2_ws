"""Bound processing by sensor timestamps without coupling to a polling phase."""
class CloudSchedule:
    def __init__(self, hz):
        self.period_ns = max(1, round(1_000_000_000 / max(.1, float(hz))))
        self.tolerance_ns = min(10_000_000, self.period_ns // 10)
        self.last_stamp_ns = None
        self.next_due_ns = None

    def ready(self, stamp_ns):
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            return False
        return (self.next_due_ns is None or
                stamp_ns + self.tolerance_ns >= self.next_due_ns)

    def mark(self, stamp_ns):
        if not self.ready(stamp_ns):
            raise ValueError('Cloud is duplicate, out of order, or not due')
        self.last_stamp_ns = stamp_ns
        if self.next_due_ns is None:
            self.next_due_ns = stamp_ns + self.period_ns
        else:
            steps = max(1, (stamp_ns + self.tolerance_ns - self.next_due_ns)
                        // self.period_ns + 1)
            self.next_due_ns += steps * self.period_ns
