"""Torch inverse kinematics and trajectory optimization for Gobot scenes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
import math

import torch
from torch import Tensor

from ._geometry import clearance_mask as _clearance_mask, rotation_log
from ._scene import Capsule, PlanningScene
from ._solver import solve_rows
from ._trajectory import (collision_penalty, grid_substeps, interpolation, make_segment_constraints,
                          optimize, run_trajectories, sqp_segment_rows)

__all__ = ["BatchedIKResult", "BatchedTrajectoryResult", "Capsule", "IKResult", "Planner", "PlannerBatch",
           "PoseTarget", "TrajectoryResult"]

Link = tuple[str, str]
_SPEED_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PoseTarget:
    """World-space link origin in metres and optional 3×3 rotation matrix."""

    link: Link | Sequence[Link]
    position: Sequence[float] | Sequence[Sequence[float]] | Tensor | None = None
    rotation: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]] | Tensor | None = None


WaypointTargets = Mapping[int, PoseTarget | Tensor | Sequence[float]]


@dataclass(frozen=True)
class IKResult:
    positions: Tensor | None
    reason: str | None = None

    @property
    def success(self) -> bool:
        return self.positions is not None


@dataclass(frozen=True)
class TrajectoryResult:
    """Optimized waypoints and the seed used by TrajOpt, in joint coordinates."""

    positions: Tensor | None
    dt: float
    reason: str | None = None
    initial_positions: Tensor | None = None

    @property
    def success(self) -> bool:
        return self.positions is not None


def _check_count(value, name, minimum=1):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _check_scalar(value, name, minimum=0, *, inclusive=False):
    if not math.isfinite(value) or value < minimum or (value == minimum and not inclusive):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")


def _world_margin(margin, world_margin):
    _check_scalar(margin, "margin", inclusive=True)
    if world_margin is None:
        return margin
    _check_scalar(world_margin, "world_margin", inclusive=True)
    return world_margin


def _target_fields(target, field, scene):
    if target.position is None and target.rotation is None:
        raise ValueError("a pose target requires position, rotation or both")
    position = rotation = None
    if target.position is not None:
        position = field(target.position, (3,), "target position")
        if not torch.isfinite(position).all():
            raise ValueError("a target position must be finite")
    if target.rotation is not None:
        rotation = field(target.rotation, (3, 3), "target rotation")
        if (not torch.isfinite(rotation).all()
                or not torch.allclose(rotation.mT @ rotation, scene.tensor(torch.eye(3)), atol=1e-5)
                or not torch.allclose(torch.linalg.det(rotation), scene.tensor(1), atol=1e-5)):
            raise ValueError("a target rotation must be a proper 3×3 rotation matrix")
    return position, rotation


def _waypoint_error(scene, target):
    def field(value, shape, name):
        value = scene.tensor(value).detach()
        if value.shape != shape:
            raise ValueError(f"invalid {name} shape {tuple(value.shape)}")
        return value

    try:
        link = scene.links[target.link]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown target link {target.link!r}") from error
    position, rotation = _target_fields(target, field, scene)

    def error(q):
        p, r = scene.poses(q)
        components = []
        if position is not None:
            components.append(p[link] - position)
        if rotation is not None:
            components.append(rotation_log(rotation.mT @ r[link]))
        return torch.cat(components)

    return error


class Planner:
    """Plan with one or more articulations and fixed root poses from a snapshot.

    The robots mapping names each robot and its selected base links. Joint vectors
    use the order of joint_names, even when several robots share a joint name.
    """

    def __init__(
        self, snapshot: dict, robots: Mapping[str, str | Sequence[str]], *,
        ignored_link_pairs: Sequence[tuple[Link, Link]] = (),
        link_capsules: Mapping[Link, Sequence[Capsule]] | None = None,
        device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64,
        collision_backend: str = "torch",
    ):
        self._scene = PlanningScene(snapshot, robots, ignored_link_pairs, link_capsules or {},
                                    dtype, torch.device(device), collision_backend)

    @property
    def joint_names(self) -> tuple[tuple[str, str], ...]:
        return self._scene.joint_names

    @property
    def initial_positions(self) -> Tensor:
        return self._scene.initial.clone()

    @property
    def collision_pairs(self) -> tuple[tuple[Link, Link], ...]:
        """Link labels in collision_distances order, one entry per shape pair."""
        return self._scene.collision_pairs

    def _joint_tensor(self, values):
        q = self._scene.tensor(values)
        if q.ndim == 0 or q.shape[-1] != len(self.joint_names):
            raise ValueError("joint positions must follow Planner.joint_names")
        return q

    def _configuration(self, values):
        q = self._joint_tensor(values).detach()
        if q.ndim != 1 or not torch.isfinite(q).all():
            raise ValueError("expected a single finite joint configuration")
        return q

    def link_pose(self, positions: Tensor, link: Link) -> tuple[Tensor, Tensor]:
        """World position and rotation of a link for each input configuration."""
        index = self._scene.links[link]
        p, r = self._scene.poses(self._joint_tensor(positions))
        return p[..., index, :], r[..., index, :, :]

    def collision_distances(self, positions: Tensor) -> Tensor:
        """Differentiable distances in metres, positive for separated shapes."""
        return self._scene.distances(self._joint_tensor(positions))

    def _bounded(self, q):
        return bool(((q >= self._scene.lower) & (q <= self._scene.upper)).all())

    def _speed_bounded(self, speed):
        return bool((speed <= self._scene.speed * (1 + _SPEED_TOLERANCE)).all())

    @torch.no_grad()
    def solve_ik(
        self, targets: Sequence[PoseTarget], *, seed: Tensor | None = None,
        margin: float = 0.02, position_tolerance: float = 1e-4, rotation_tolerance: float = 1e-3,
        iterations: int = 30, world_margin: float | None = None,
    ) -> IKResult:
        """Solve one or several link poses together, subject to bounds and clearance."""
        batch = PlannerBatch(self)
        result = batch._solve_ik(targets, self._configuration(seed) if seed is not None else None,
                                margin, position_tolerance, rotation_tolerance, iterations, single=True,
                                world_margin=world_margin)
        if not result.success.item():
            return IKResult(None, result.reasons[0])
        return IKResult(result.positions[0])

    @torch.no_grad()
    def validate_trajectory(
        self, positions: Tensor, *, dt: float, margin: float = 0.02, collision_substeps: int = 16,
        world_margin: float | None = None, longest_valid_segment_length: float | None = None,
    ) -> bool:
        """Check joint limits, sampled clearance and swept world clearance for every segment."""
        _check_scalar(dt, "dt")
        world_margin = _world_margin(margin, world_margin)
        _check_count(collision_substeps, "collision_substeps")
        q = self._joint_tensor(positions)
        if q.ndim != 2 or q.shape[0] < 2:
            raise ValueError("a trajectory must contain at least two joint configurations")
        if longest_valid_segment_length is not None:
            _check_scalar(longest_valid_segment_length, "longest_valid_segment_length")
        return self._trajectory_failure(q, dt, margin, world_margin, collision_substeps,
                                        longest_valid_segment_length) is None

    def _trajectory_failure(self, q, dt, margin, world_margin, substeps, max_length=None):
        scene = self._scene
        if not torch.isfinite(q).all() or not self._bounded(q):
            return "trajectory violates joint limits"
        if not self._speed_bounded(q.diff(dim=0).abs() / dt):
            return "trajectory violates joint velocity limits"
        substeps = grid_substeps(q, substeps, max_length)
        samples = interpolation(q, substeps)

        def collision(ok, pairs, samples_per_segment):
            if ok.all():
                return None
            sample, pair = (~ok).nonzero()[0].tolist()
            segment = min(sample // samples_per_segment, len(q) - 2)
            return f"collision clearance on segment {segment}, pair {pairs[pair]}"

        p, r = scene.poses(samples)
        for _, ok, names in scene.original_checks(p, r, margin, world_margin):
            failure = collision(ok, names, substeps)
            if failure:
                return failure
        return (collision(scene.convex_clearance(p, r, scene.pair_margins(margin, world_margin)),
                          scene.distance_models[0].names if scene.distance_models else scene.collision_pairs, substeps)
                or collision(_clearance_mask(scene.swept_world_distances(q[:-1], q[1:]), world_margin),
                             scene.swept_pairs, 1))

    @torch.no_grad()
    def solve_trajopt(
        self, start: Tensor, goal: Tensor, *, dt: float, waypoints: int = 16,
        margin: float = 0.02, collision_substeps: int = 8, iterations: int = 20,
        seed: Tensor | None = None, world_margin: float | None = None, stop_at_endpoints: bool = False,
        solver: str = "lm", waypoint_targets: WaypointTargets | None = None,
        longest_valid_segment_length: float | None = None,
    ) -> TrajectoryResult:
        """Optimize a fixed-endpoint trajectory. CPU SQP accepts world poses or joint waypoints.

        margin is the self-collision clearance in metres. world_margin sets the
        environment clearance and defaults to margin. SQP considers collision
        rows within 0.05 m of their clearance threshold for the local QP.
        Trial evaluation and trajectory validation check all enabled pairs.

        Each requested position/rotation error vector must have norm at most 1e-4 (m/rad).
        longest_valid_segment_length bounds joint-space sampling intervals (Euclidean norm
        in the joint vector's native units). Collision verification checks the interpolants.
        """
        return self._solve_trajectories([(start, goal, seed, waypoint_targets)], dt, waypoints,
                                        margin, collision_substeps, iterations, world_margin,
                                        stop_at_endpoints, solver, longest_valid_segment_length)[0]

    def _solve_trajectories(self, requests, dt, waypoints, margin, collision_substeps, iterations,
                            world_margin, stop_at_endpoints, solver, longest_valid_segment_length):
        _check_scalar(dt, "dt")
        world_margin = _world_margin(margin, world_margin)
        _check_count(waypoints, "waypoints", 2)
        _check_count(collision_substeps, "collision_substeps")
        _check_count(iterations, "iterations")
        if longest_valid_segment_length is not None:
            _check_scalar(longest_valid_segment_length, "longest_valid_segment_length")
        if solver not in ("lm", "sqp"):
            raise ValueError("solver must be 'lm' or 'sqp'")
        if solver == "sqp" and self._scene.device.type != "cpu":
            raise ValueError("the SQP solver requires a CPU planner")
        fixed = 2 if stop_at_endpoints else 1
        if waypoints < 2 * fixed:
            raise ValueError("stopping at both endpoints requires at least four waypoints")
        constraints = make_segment_constraints(self._scene, dt, margin, world_margin)
        jobs = [self._trajectory_job(*request, dt, waypoints, margin, world_margin, collision_substeps,
                                     iterations, fixed, solver, longest_valid_segment_length, constraints)
                for request in requests]
        evaluate = partial(sqp_segment_rows, self._scene, dt, self._scene.pair_margins(margin, world_margin),
                           world_margin, constraints)
        return run_trajectories(jobs, evaluate)

    def _trajectory_job(self, start, goal, seed, waypoint_targets, dt, waypoints, margin, world_margin,
                        collision_substeps, iterations, fixed, solver, longest_valid_segment_length, constraints):
        if waypoint_targets and solver != "sqp":
            raise ValueError("intermediate waypoint targets require solver='sqp'")
        targets = []
        joint_anchors = {}
        for index, target in (waypoint_targets or {}).items():
            if type(index) is not int or not 0 <= index < waypoints:
                raise ValueError("target waypoint index is outside the trajectory")
            if isinstance(target, PoseTarget):
                error = _waypoint_error(self._scene, target)
            else:
                joint_target = self._configuration(target)
                if not self._bounded(joint_target):
                    return TrajectoryResult(None, dt, f"joint limits at waypoint {index}")
                error = partial(torch.sub, other=joint_target)
                if fixed <= index < waypoints - fixed:
                    joint_anchors[index] = joint_target
            targets.append((index, error))
        start, goal = self._configuration(start), self._configuration(goal)
        ends = torch.stack((start, goal))
        if not self._bounded(ends):
            return TrajectoryResult(None, dt, "start or goal violates joint limits")
        scene = self._scene
        p, r = scene.poses(ends)
        names = scene.distance_models[0].names if scene.distance_models else scene.collision_pairs
        checks = [(scene.convex_clearance(p, r, scene.pair_margins(margin, world_margin)), names)]
        checks.extend((ok, names) for _, ok, names in scene.original_checks(p, r, margin, world_margin))
        for ok, pairs in checks:
            if not ok.all():
                row, index = (~ok).nonzero()[0].tolist()
                segment = 0 if row == 0 else waypoints - 2
                return TrajectoryResult(None, dt, f"endpoint collision on segment {segment}, pair {pairs[index]}")
        moving_segments = waypoints - 2 * fixed + 1
        if not self._speed_bounded((goal - start).abs() / (moving_segments * dt)):
            return TrajectoryResult(None, dt, "joint velocity limits require a longer duration")

        if seed is None:
            anchors = [(fixed - 1, start), *sorted(joint_anchors.items()), (waypoints - fixed, goal)]
            pieces = []
            for (first, a), (last, b) in zip(anchors, anchors[1:]):
                fractions = torch.linspace(0, 1, last - first + 1, device=start.device, dtype=start.dtype)
                pieces.append(torch.lerp(a, b, fractions[:-1, None]))
            interior = torch.cat(pieces)[1:]
        else:
            initial = self._joint_tensor(seed).detach()
            if initial.shape != (waypoints, len(self.joint_names)) or not torch.isfinite(initial).all():
                raise ValueError("seed must contain one finite joint configuration per waypoint")
            interior = initial[fixed:-fixed].clamp(scene.lower, scene.upper)
        initial = torch.cat((start.expand(fixed, -1), interior, goal.expand(fixed, -1)))

        validation_substeps = 2 * collision_substeps
        failure = None

        def feasible(positions):
            nonlocal failure
            for index, error in targets:
                residual = error(positions[index])
                cartesian = isinstance(waypoint_targets[index], PoseTarget)
                magnitude = (residual.reshape(-1, 3).norm(dim=-1).max() if cartesian else residual.abs().max())
                if not magnitude <= 1e-4:
                    kind = "Cartesian pose" if cartesian else "joint position"
                    failure = f"{kind} tolerance at waypoint {index}"
                    return False
            failure = self._trajectory_failure(positions, dt, margin, world_margin, validation_substeps,
                                               longest_valid_segment_length)
            return failure is None

        if not interior.numel():
            q = initial if feasible(initial) else None
        else:
            # A failure between optimization samples gets one retry on the verification grid.
            for substeps in (collision_substeps, validation_substeps):
                q = yield from optimize(scene, initial, dt, margin, substeps, iterations, feasible,
                                        fixed, solver, targets, longest_valid_segment_length, constraints)
                if q is not None or not scene.has_collision_pairs:
                    break
        if q is None:
            return TrajectoryResult(None, dt, failure or "trajectory optimization found no feasible path from this seed",
                                    initial_positions=initial)
        return TrajectoryResult(q, dt, initial_positions=initial)

    def plan_poses(
        self, start: Sequence[PoseTarget], goal: Sequence[PoseTarget], *, dt: float,
        waypoints: int = 16, margin: float | None = None, collision_substeps: int = 8,
        start_seed: Tensor | None = None, goal_seed: Tensor | None = None, seed: Tensor | None = None,
        world_margin: float | None = None, stop_at_endpoints: bool | None = None,
        solver: str = "lm", waypoint_targets: WaypointTargets | None = None,
        longest_valid_segment_length: float | None = None,
    ) -> TrajectoryResult:
        """Solve endpoint IK and optimize a joint trajectory between the solutions."""
        _check_scalar(dt, "dt")
        _check_count(waypoints, "waypoints", 2)
        _check_count(collision_substeps, "collision_substeps")
        if margin is None:
            margin = 0.02
            world_margin = 0.05 if world_margin is None else world_margin
        world_margin = _world_margin(margin, world_margin)
        batch = PlannerBatch(self)
        start_problem = batch._prepare_ik(start, start_seed, margin, 1e-4, 1e-3, True, world_margin)
        goal_problem = batch._prepare_ik(goal, goal_seed if goal_seed is not None else start_seed,
                                         margin, 1e-4, 1e-3, True, world_margin)
        endpoints = self._solve_endpoints(start_problem, goal_problem, margin, world_margin)
        if endpoints is None:
            # Separate solves can escape a stationary point of the coupled problem.
            start_ik = self.solve_ik(start, seed=start_seed, margin=margin, world_margin=world_margin)
            if not start_ik.success:
                return TrajectoryResult(None, dt, f"start {start_ik.reason}")
            goal_ik = self.solve_ik(goal, seed=start_ik.positions if goal_seed is None else goal_seed,
                                    margin=margin, world_margin=world_margin)
            if not goal_ik.success:
                return TrajectoryResult(None, dt, f"goal {goal_ik.reason}")
            endpoints = start_ik.positions, goal_ik.positions
        return self.solve_trajopt(
            *endpoints, dt=dt, waypoints=waypoints, margin=margin,
            collision_substeps=collision_substeps, seed=seed, world_margin=world_margin,
            solver=solver, waypoint_targets=waypoint_targets, longest_valid_segment_length=longest_valid_segment_length,
            stop_at_endpoints=(waypoints >= 4 if stop_at_endpoints is None else stop_at_endpoints),
        )

    @torch.no_grad()
    def _solve_endpoints(self, start_problem, goal_problem, margin, world_margin):
        scene = self._scene
        seeds, residuals, feasible = zip(start_problem, goal_problem)
        initial = torch.cat(seeds, dim=-1)
        joint_count = len(self.joint_names)
        n_self = scene.self_pair_count
        margins = scene.pair_margins(margin, world_margin)

        def terms(x):
            start, goal = x.split(joint_count, dim=-1)
            soft = [goal - start]
            constraints = []
            for q, evaluate in zip((start, goal), residuals):
                objective, clearance = evaluate(q)
                distance = margins - clearance
                pose = objective[:, joint_count:]
                soft.extend((0.001 * (q - scene.initial), pose,
                             5 * collision_penalty(distance[:, :n_self], margin),
                             10 * collision_penalty(distance[:, n_self:], world_margin)))
                # Preserve the IK pose tolerances as the joint-similarity cost
                # pulls the two configurations together.
                constraints.extend((pose, -pose, clearance))
            return torch.cat(soft, dim=-1), torch.cat(constraints, dim=-1)

        def valid(x):
            start, goal = x.split(joint_count, dim=-1)
            return feasible[0](start) & feasible[1](goal)

        x, ok = solve_rows(initial, scene.lower.repeat(2), scene.upper.repeat(2), terms, valid, 30)
        return x[0].split(joint_count) if ok.item() else None


@dataclass(frozen=True)
class BatchedIKResult:
    """Joint rows and a success mask. Failed rows contain NaN."""

    positions: Tensor
    success: Tensor
    reasons: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class BatchedTrajectoryResult:
    """Joint trajectories [batch, waypoints, joints] in input order and per-row status.

    Failed positions and unavailable initial trajectories are filled with NaN.
    """

    positions: Tensor
    success: Tensor
    reasons: tuple[str | None, ...]
    dt: float
    initial_positions: Tensor


class PlannerBatch:
    """Independent trajectories in a shared scene, or IK with shared or per-row models."""

    def __init__(self, planners: Planner | Sequence[Planner]):
        self._shared_planner = planners if isinstance(planners, Planner) else None
        if isinstance(planners, Planner):
            self._scene = planners._scene
            self._size = None
        else:
            if not planners:
                raise ValueError("provide at least one planner")
            self._scene = PlanningScene.stack([planner._scene for planner in planners])
            self._size = len(planners)

    def collision_distances(self, positions: Tensor) -> Tensor:
        """Compute all active pair distances for each row of joint positions."""
        q = self._scene.tensor(positions)
        if q.ndim != 2 or q.shape[-1] != len(self._scene.joint_names):
            raise ValueError("expected a batch of joint configurations")
        if self._size is not None and self._size != q.shape[0]:
            raise ValueError("provide one joint configuration per planner")
        return self._scene.distances(q)

    @torch.no_grad()
    def solve_trajopt(
        self, start: Tensor, goal: Tensor, *, dt: float, waypoints: int = 16,
        waypoint_targets: Sequence[WaypointTargets] | None = None, seed: Tensor | None = None,
        margin: float = 0.02, world_margin: float | None = None, collision_substeps: int = 8,
        iterations: int = 20, stop_at_endpoints: bool = False,
        longest_valid_segment_length: float | None = None, chunk_size: int = 8,
    ) -> BatchedTrajectoryResult:
        """Plan independent CPU SQP trajectories using one shared Planner.

        start and goal have shape [B, J], seed has shape [B, T, J]. Each row may
        specify its own mapping from waypoint indices to joint vectors or world-space
        PoseTargets. The maximum joint error or position/rotation norm is 1e-4 in
        joint units or metres/radians. chunk_size bounds the number of concurrent paths.
        """
        planner = self._shared_planner
        if planner is None:
            raise ValueError("batch trajectory optimization requires PlannerBatch(shared_planner)")
        _check_count(chunk_size, "chunk_size")
        _check_count(waypoints, "waypoints", 2)
        start, goal = planner._joint_tensor(start), planner._joint_tensor(goal)
        if start.ndim != 2 or not len(start) or goal.shape != start.shape:
            raise ValueError("start and goal must have matching nonempty [batch, joints] shapes")
        size, joints = start.shape
        if seed is not None:
            seed = planner._joint_tensor(seed)
            if seed.shape != (size, waypoints, joints):
                raise ValueError("seed must have shape [batch, waypoints, joints]")
        if waypoint_targets is not None and (len(waypoint_targets) != size or
                                             not all(isinstance(row, Mapping) for row in waypoint_targets)):
            raise ValueError("provide one waypoint target mapping per trajectory")
        results = []
        for offset in range(0, size, chunk_size):
            group = [(start[i], goal[i], None if seed is None else seed[i],
                      None if waypoint_targets is None else waypoint_targets[i])
                     for i in range(offset, min(offset + chunk_size, size))]
            results.extend(planner._solve_trajectories(group, dt, waypoints, margin, collision_substeps,
                                                       iterations, world_margin, stop_at_endpoints, "sqp",
                                                       longest_valid_segment_length))
        empty = start.new_full((waypoints, joints), torch.nan)
        return BatchedTrajectoryResult(
            torch.stack([r.positions if r.success else empty for r in results]),
            torch.tensor([r.success for r in results], device=start.device),
            tuple(r.reason for r in results), dt,
            torch.stack([r.initial_positions if r.initial_positions is not None else empty for r in results]),
        )

    @torch.no_grad()
    def solve_ik(
        self, targets: Sequence[PoseTarget], *, seed: Tensor | None = None,
        margin: float = 0.02, position_tolerance: float = 1e-4, rotation_tolerance: float = 1e-3,
        iterations: int = 30,
    ) -> BatchedIKResult:
        """Each link, target position, rotation and seed can be shared or specified per row."""
        return self._solve_ik(targets, seed, margin, position_tolerance, rotation_tolerance, iterations)

    def _solve_ik(self, targets, seed, margin, position_tolerance, rotation_tolerance, iterations,
                  single=False, world_margin=None):
        _check_count(iterations, "iterations")
        failures = {}
        q0, terms, feasible = self._prepare_ik(targets, seed, margin, position_tolerance, rotation_tolerance,
                                               single, world_margin, failures)
        q, success = solve_rows(q0, self._scene.lower, self._scene.upper, terms, feasible, iterations)
        generic = "IK found no configuration satisfying pose, joint limits and clearance"
        reasons = tuple(None if ok else failures.get(i, generic) for i, ok in enumerate(success.tolist()))
        return BatchedIKResult(q.masked_fill(~success[:, None], torch.nan), success, reasons)

    def _prepare_ik(self, targets, seed, margin, position_tolerance, rotation_tolerance, single, world_margin, failures=None):
        world_margin = _world_margin(margin, world_margin)
        _check_scalar(position_tolerance, "position_tolerance")
        _check_scalar(rotation_tolerance, "rotation_tolerance")
        if not targets:
            raise ValueError("provide at least one pose target")
        scene = self._scene
        sizes = set()

        def field(value, shape, name):
            value = scene.tensor(value).detach()
            if value.shape == shape:
                return value
            if not single and value.ndim == len(shape) + 1 and value.shape[1:] == shape and value.shape[0]:
                sizes.add(value.shape[0])
                return value
            raise ValueError(f"invalid {name} shape {tuple(value.shape)}")

        joints = len(scene.joint_names)
        q0 = field(scene.initial if seed is None else seed, (joints,), "seed")
        if not torch.isfinite(q0).all():
            raise ValueError("joint configurations must be finite")

        positions, rotations = [], []
        for target in targets:
            if isinstance(target.link, tuple) and len(target.link) == 2 and isinstance(target.link[0], str):
                links = (target.link,)
            elif not single and target.link:
                links = tuple(target.link)
                sizes.add(len(links))
            else:
                raise ValueError("expected a (robot, link) target")
            try:
                index = torch.tensor([scene.links[link] for link in links], device=scene.device)
            except (KeyError, TypeError) as error:
                raise ValueError(f"unknown target link {target.link!r}") from error
            position, rotation = _target_fields(target, field, scene)
            if position is not None:
                positions.append((index, position))
            if rotation is not None:
                rotations.append((index, rotation))
        if self._size is not None:
            sizes.add(self._size)
        if len(sizes) > 1:
            raise ValueError("batch sizes of models, links, targets and seeds must match")
        size = next(iter(sizes), 1)
        rows = torch.arange(size, device=scene.device)
        q0 = q0.expand(size, joints).clamp(scene.lower, scene.upper)
        margins = scene.pair_margins(margin, world_margin)

        def errors(p, r):
            translations = [p[rows, index] - goal for index, goal in positions]
            angles = [rotation_log(goal.mT @ r[rows, index]) for index, goal in rotations]
            return translations, angles

        def terms(q):
            p, r = scene.poses(q)
            translations, angles = errors(p, r)
            residual = torch.cat([0.001 * (q - q0), *(10 * e for e in translations), *angles], dim=-1)
            return residual, margins - scene.pose_distances(p, r)

        def feasible(q):
            p, r = scene.poses(q)
            translations, angles = errors(p, r)
            valid = scene.convex_clearance(p, r, margins).all(-1)
            for error in translations:
                valid = valid & (torch.linalg.vector_norm(error, dim=-1) <= position_tolerance)
            for error in angles:
                valid = valid & (torch.linalg.vector_norm(error, dim=-1) <= rotation_tolerance)
            for batch, ok, names in scene.original_checks(p, r, margin, world_margin):
                accepted = ok.all(-1)
                if failures is not None:
                    for row in (valid[batch] & ~accepted).nonzero().flatten().tolist():
                        pair = (~ok[row]).nonzero()[0].item()
                        failures[(batch.start or 0) + row] = f"IK original collision clearance failed for pair {names[pair]}"
                valid[batch] &= accepted
            return valid

        return q0, terms, feasible
