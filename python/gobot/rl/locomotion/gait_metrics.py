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


class QuadrupedGaitMetrics:
    """Estimate footfall phases from completed strides, independently of the policy clock."""
    def __init__(self, count, dt=.02, warmup_s=2.):
        self.dt, self.warmup = dt, int(warmup_s / dt)
        self.age = np.zeros(count, int)
        self.samples = np.zeros(count, int)
        self.previous = np.ones((count, 4), bool)
        self.last_time = np.full((count, 4), -1.)
        self.last_position = np.zeros((count, 4, 3))
        self.stance = np.zeros((count,4), int)
        self.landings = np.zeros((count,4), int)
        self.airborne = np.zeros(count, int)
        self.simultaneous = np.zeros(count, int)
        self.stride_sum = np.zeros(count)
        self.strides = np.zeros(count,int)
        self.cycles = np.zeros(count,int)
        self.phase_matches = np.zeros((count,3),int)

    def update(self, contact, foot_position, active):
        contact, positions, active = np.asarray(contact,bool), np.asarray(foot_position), np.asarray(active,bool)
        if contact.shape != self.previous.shape:
            raise ValueError('quadruped gait requires FR, FL, RR, RL contacts')
        self.age[active] += 1
        valid = active & (self.age > self.warmup)
        new = contact & ~self.previous & valid[:,None]
        # Reject contact chatter as a new stride (at least 100 ms between landings).
        new &= (self.age[:,None] * self.dt - self.last_time) >= .1
        self.samples[valid] += 1
        self.stance[valid] += contact[valid]
        self.airborne[valid] += ~contact[valid].any(1)
        self.simultaneous += new.sum(1) >= 3
        self.landings += new
        now = self.age * self.dt
        complete = new[:,0] & (self.last_time >= 0).all(1)
        period = now - self.last_time[:,0]
        complete &= (period >= .15) & (period <= 2.)
        phases = ((self.last_time - now[:,None]) / np.maximum(period[:,None], 1.e-6)) % 1
        # Left- and right-lead variants are equivalent natural gaits.
        templates = np.array(((0.,.5,.75,.25),(0.,.5,.5,0.),(.6,.5,.1,0.)))
        errors = []
        for template in templates:
            variants = []
            for t in (template, template[[1,0,3,2]]):
                t = (t-t[0]) % 1
                delta = np.abs(phases-t)
                variants.append(np.minimum(delta,1-delta).mean(1))
            errors.append(np.minimum(*variants))
        errors = np.stack(errors,1)
        kind = errors.argmin(1)
        self.cycles += complete
        for gait in range(3):
            self.phase_matches[:,gait] += complete & (kind == gait) & (errors[:,gait] <= .12)
        for foot in range(4):
            seen = new[:,foot] & (self.last_time[:,foot] >= 0)
            self.stride_sum[seen] += np.linalg.norm(positions[seen,foot,:2]-self.last_position[seen,foot,:2],axis=1)
            self.strides[seen] += 1
            self.last_time[new[:,foot],foot] = now[new[:,foot]]
            self.last_position[new[:,foot],foot] = positions[new[:,foot],foot]
        self.previous[active] = contact[active]

    def arrays(self):
        samples = np.maximum(self.samples,1)
        return {'cadence_hz': self.landings.sum(1)/(samples*self.dt),
                'duty_factor':self.stance/samples[:,None],
                'flight_fraction':self.airborne/samples,
                'stride_length_m':self.stride_sum/np.maximum(self.strides,1),
                'landings':self.landings.sum(1), 'measured_seconds':self.samples*self.dt,
                'cycles':self.cycles, 'walk_fraction':self.phase_matches[:,0]/np.maximum(self.cycles,1),
                'trot_fraction':self.phase_matches[:,1]/np.maximum(self.cycles,1),
                'gallop_fraction':self.phase_matches[:,2]/np.maximum(self.cycles,1),
                'simultaneous_landings':self.simultaneous}
