"""Independent biped gait measurements; never use discriminator scores for admission."""
from __future__ import annotations
import numpy as np


class BipedGaitMetrics:
    def __init__(self, count, dt=.02, warmup_s=2., contact_dwell_s=.06):
        self.count, self.dt = int(count), float(dt)
        self.warmup = int(round(warmup_s / dt))
        self.age = np.zeros(count, np.int64)
        self.samples = np.zeros(count, np.int64)
        self.stance = np.zeros((count, 2), np.int64)
        self.airborne = np.zeros(count, np.int64)
        self.landings = np.zeros((count, 2), np.int64)
        self.previous = np.ones((count, 2), bool)
        self.contact_dwell_steps = max(1, int(round(contact_dwell_s / dt)))
        self.candidate = self.previous.copy()
        self.candidate_steps = np.zeros((count, 2), np.int64)
        self.last_side = np.full(count, -1, np.int64)
        self.transitions = np.zeros(count, np.int64)
        self.alternations = np.zeros(count, np.int64)
        self.simultaneous = np.zeros(count, np.int64)
        self.last_landing = np.zeros((count, 2, 3))
        self.landing_seen = np.zeros((count, 2), bool)
        self.stride_sum = np.zeros((count, 2))
        self.stride_count = np.zeros((count, 2), np.int64)

    def update(self, contact, foot_position, active):
        contact = np.asarray(contact, bool)
        position = np.asarray(foot_position)
        active = np.asarray(active, bool)
        if contact.shape != (self.count, 2) or position.shape != (self.count, 2, 3):
            raise ValueError('biped gait metrics require two feet per environment')
        # MuJoCo can chatter at contact boundaries. Only sustained contact changes
        # count as a landing, rather than treating solver noise as a fast gait.
        changed = contact != self.candidate
        self.candidate_steps[changed & active[:, None]] = 0
        self.candidate[active] = contact[active]
        self.candidate_steps[active] += 1
        contact = np.where(self.candidate_steps >= self.contact_dwell_steps, self.candidate, self.previous)
        self.age[active] += 1
        valid = active & (self.age > self.warmup)
        new = contact & ~self.previous & valid[:, None]
        self.samples[valid] += 1
        self.stance[valid] += contact[valid]
        self.airborne[valid] += ~contact[valid].any(1)
        self.landings += new
        double = new.all(1)
        self.simultaneous += double
        single = new.any(1) & ~double
        side = new[:, 1].astype(np.int64)
        comparable = single & (self.last_side >= 0)
        self.transitions += comparable
        self.alternations += comparable & (side != self.last_side)
        self.last_side[single] = side[single]
        for foot in range(2):
            complete = new[:, foot] & self.landing_seen[:, foot]
            distance = np.linalg.norm(position[:, foot, :2] - self.last_landing[:, foot, :2], axis=1)
            self.stride_sum[complete, foot] += distance[complete]
            self.stride_count[complete, foot] += 1
            self.last_landing[new[:, foot], foot] = position[new[:, foot], foot]
            self.landing_seen[new[:, foot], foot] = True
        self.previous[active] = contact[active]

    def arrays(self):
        samples = np.maximum(self.samples, 1)
        count = np.maximum(self.stride_count.sum(1), 1)
        return {'cadence_hz': self.landings.sum(1) / (samples * self.dt),
                'duty_factor': self.stance / samples[:, None],
                'flight_fraction': self.airborne / samples,
                'alternation_fraction': self.alternations / np.maximum(self.transitions + self.simultaneous, 1),
                'stride_length_m': self.stride_sum.sum(1) / count,
                'landings': self.landings.sum(1),
                'measured_seconds': self.samples * self.dt}
