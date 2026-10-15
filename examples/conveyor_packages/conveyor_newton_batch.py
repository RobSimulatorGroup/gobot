"""Headless Newton workcell validation and Gobot-rendered video capture."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import gobot

from conveyor_config import WORKTABLE_TOP_Z, BELT_CENTER_Y, BELT_WIDTH
from conveyor_newton_runtime import create, default_config, prepare_snapshot, scene_sync
from gobot.sim import SimulationSession

SUPPORTED_TRANSPORT_PHASES = ("place_hold", "release", "clear", "settle_after_release", "outfeed")


def _maximum_windowed_vertical_speed(samples, name, window_s):
    """Diagnostic RMS trend; acceptance also requires the instantaneous peak."""
    times = np.array([sample["time"] for sample in samples])
    squared = np.array([sample["parcels"][name]["vertical_rms_speed_m_s"]**2 for sample in samples])
    maximum = 0.
    for i, end in enumerate(times):
        start = max(times[0], end - window_s)
        if end == start:
            mean_square = squared[i]
        else:
            first = np.searchsorted(times, start, side="right")
            t = np.r_[start, times[first:i+1]]
            values = np.r_[np.interp(start, times[:i+1], squared[:i+1]), squared[first:i+1]]
            mean_square = np.sum(np.diff(t) * .5 * (values[:-1] + values[1:])) / (end-start)
        maximum = max(maximum, float(np.sqrt(mean_square)))
    return maximum


class TrialMetrics:
    def __init__(self, snapshot):
        self.samples = []
        self.peak_penetration = 0.
        self.minimum_volume_ratio = 1.
        self.rest_tets = {}
        self.press_region = {}
        self.press_pads = {}
        self.quiet_reference = {}
        self.quiet_previous = {}
        self.transport_previous = {}
        self.unattended_names = tuple(body["name"] for body in snapshot["deformables"]
                                      if body["name"].startswith("soft_pouch_yellow"))
        for body in snapshot["deformables"]:
            if body["model"] == 1:
                local = np.asarray(body["vertices"])
                mask = (np.abs(local[:, 0]) < .10) & (np.abs(local[:, 1]) < .055)
                mask[:len(local)//2] = False
                self.press_region[body["name"]] = mask
                if body["name"] == "soft_mailer_blue":
                    transform = np.asarray(body["transform"]["matrix"])
                    world = local @ transform[:3, :3].T + transform[:3, 3]
                    # Six regular fingertips, each with a 26 x 30 mm contact
                    # footprint. Retain the broader central-patch measurement
                    # separately rather than mixing untouched film into the press.
                    centers = np.array((.025, .070, .115, .205, .250, .295))
                    pads = (np.min(np.abs(world[:, :1] - centers), axis=1) < .013) & (np.abs(world[:, 1] + .375) < .015)
                    pads[:len(local)//2] = False
                    if not pads.any():
                        raise ValueError("the press trajectory does not overlap the authored mailer")
                    self.press_pads[body["name"]] = pads
            if body["tetrahedra"]:
                indices = np.asarray(body["tetrahedra"]).reshape(-1, 4)
                vertices = np.asarray(body["vertices"])[indices]
                determinant = np.linalg.det(vertices[:, 1:] - vertices[:, :1])
                self.rest_tets[body["name"]] = (indices, determinant)

    def update(self, state, phase):
        self.peak_penetration = max(self.peak_penetration, state["peak_soft_penetration"])
        self.minimum_volume_ratio = min(self.minimum_volume_ratio, state["minimum_tet_volume_ratio"])
        parcels = {}
        for name, body in state["deformables"].items():
            q = body["positions"]
            parcels[name] = {"center": q.mean(0).tolist(), "min": q.min(0).tolist(),
                             "max": q.max(0).tolist(), "height": float(np.ptp(q[:, 2])),
                             "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(body["velocities"] ** 2, axis=1)))),
                             "vertical_rms_speed_m_s": float(np.sqrt(np.mean(body["velocities"][:, 2] ** 2))),
                             "max_speed_m_s": float(np.linalg.norm(body["velocities"], axis=1).max())}
            if phase in SUPPORTED_TRANSPORT_PHASES:
                previous = self.transport_previous.get(name, q)
                dz = q[:, 2] - previous[:, 2]
                parcels[name]["sample_max_vertex_rise_m"] = float(max(0., dz.max()))
                parcels[name]["sample_rms_vertical_displacement_m"] = float(np.sqrt(np.mean(dz**2)))
            self.transport_previous[name] = q.copy()
            if name in self.unattended_names and state["time"] >= 1.:
                reference = self.quiet_reference.setdefault(name, q.copy())
                previous = self.quiet_previous.get(name, q)
                parcels[name]["rest_rms_displacement_m"] = float(np.sqrt(np.mean(np.sum((q - reference) ** 2, axis=1))))
                parcels[name]["sample_max_displacement_m"] = float(np.linalg.norm(q - previous, axis=1).max())
                self.quiet_previous[name] = q.copy()
            if name in self.press_region:
                parcels[name]["press_region_height"] = float(np.quantile(q[self.press_region[name], 2], .9) - WORKTABLE_TOP_Z)
            if name in self.press_pads:
                parcels[name]["press_pad_height"] = float(np.mean(q[self.press_pads[name], 2]) - WORKTABLE_TOP_Z)
            if name in self.rest_tets:
                indices, rest = self.rest_tets[name]
                points = q[indices]
                ratio = np.linalg.det(points[:, 1:] - points[:, :1]) / rest
                self.minimum_volume_ratio = min(self.minimum_volume_ratio, float(ratio.min()))
        sample = {"time": state["time"], "phase": phase, "parcels": parcels,
                  "soft_contact_count": state["soft_contact_count"],
                  "max_soft_penetration_m": state["max_soft_penetration"]}
        self.samples.append(sample)
        return sample

    def report(self, *, drop_only=False):
        blue = "soft_mailer_blue"
        def heights(phase, field="press_pad_height"):
            return [s["parcels"][blue][field] for s in self.samples if s["phase"] == phase]
        settled = heights("settle_and_carton_drop")
        pressed = heights("press") + heights("press_hold")
        recovered = heights("recover")
        baseline = settled[-1] if settled else None
        compression = 1 - min(pressed) / baseline if baseline and pressed else None
        recovery = recovered[-1] / baseline if baseline and recovered else None
        # A launched or oscillating bag can have a large apparent "recovery".
        # Require the shell and contents to settle in place after withdrawing.
        rest_samples = [s for s in self.samples if s["phase"] == "settle_and_carton_drop"]
        recovery_samples = [s for s in self.samples if s["phase"] == "recover"]
        recovery_drift = recovery_speed = recovery_max_speed = None
        if rest_samples and recovery_samples:
            rest_center = np.asarray(rest_samples[-1]["parcels"][blue]["center"][:2])
            recovery_drift = max(float(np.linalg.norm(np.asarray(s["parcels"][blue]["center"][:2]) - rest_center))
                                 for s in recovery_samples)
            tail = [s for s in recovery_samples if s["time"] >= recovery_samples[-1]["time"] - .2]
            recovery_speed = max(s["parcels"][name]["rms_speed_m_s"]
                                 for s in tail for name in (blue, blue + "_fill"))
            recovery_max_speed = max(s["parcels"][name]["max_speed_m_s"]
                                     for s in tail for name in (blue, blue + "_fill"))
        central_settled = heights("settle_and_carton_drop", "press_region_height")
        central_pressed = heights("press", "press_region_height") + heights("press_hold", "press_region_height")
        central_compression = (1 - min(central_pressed) / central_settled[-1]
                               if central_settled and central_pressed else None)
        holds = [s for s in self.samples if s["phase"] == "hold"]
        hold_time = 0.
        hold_start = None
        final_hold_clearance = None
        for sample in holds:
            lowest = min(sample["parcels"][name]["min"][2] for name in (blue, blue + "_fill"))
            final_hold_clearance = lowest - WORKTABLE_TOP_Z
            if lowest < WORKTABLE_TOP_Z + .1:
                hold_start = None
            else:
                if hold_start is None:
                    hold_start = sample["time"]
                hold_time = max(hold_time, sample["time"] - hold_start)
        checks = {"press_compression": compression is not None and compression >= .1,
                  "height_recovery": recovery is not None and recovery >= .8,
                  "settled_recovery": (recovery_drift is not None and recovery_drift <= .02
                                       and recovery_speed <= .03 and recovery_max_speed <= .25),
                  "lift_and_hold": hold_time >= 1. and final_hold_clearance >= .1,
                  "contact_penetration": self.peak_penetration <= .001,
                  "positive_tet_volumes": self.minimum_volume_ratio > 0.}
        outfeed = [s for s in self.samples if s["phase"] == "outfeed"]
        transfer = None
        if outfeed:
            first = np.asarray(outfeed[0]["parcels"][blue]["center"])
            last = np.asarray(outfeed[-1]["parcels"][blue]["center"])
            transfer = {"final_center_y_m": float(last[1]), "outfeed_travel_m": float(last[0] - first[0])}
        checks["conveyor_transfer"] = (transfer is not None
            and transfer["final_center_y_m"] >= BELT_CENTER_Y - .5 * BELT_WIDTH
            and transfer["outfeed_travel_m"] >= .2)
        transport = [s for s in self.samples if s["phase"] in SUPPORTED_TRANSPORT_PHASES]
        placed = [s for s in self.samples if s["phase"] == "place_hold"]
        transport_stability = None
        if placed and transport:
            # Intentional carrying/lowering has vertical motion. Once laid
            # down, check every sample through opening, withdrawal and outfeed.
            # Do not wait for the release or settling phase to finish.
            reference = placed[0]["parcels"][blue]
            transport_stability = {
                "maximum_center_rise_m": max(s["parcels"][blue]["center"][2] - reference["center"][2] for s in transport),
                "maximum_top_rise_m": max(s["parcels"][blue]["max"][2] - reference["max"][2] for s in transport),
                "maximum_bottom_clearance_m": max(s["parcels"][blue]["min"][2] - WORKTABLE_TOP_Z for s in transport),
                "maximum_vertical_rms_speed_m_s": max(s["parcels"][blue]["vertical_rms_speed_m_s"] for s in transport),
                "vertical_speed_window_s": .2,
                "maximum_windowed_vertical_rms_speed_m_s": _maximum_windowed_vertical_speed(transport, blue, .2),
                "maximum_sample_vertex_rise_m": max(s["parcels"][blue]["sample_max_vertex_rise_m"] for s in transport),
                "maximum_sample_rms_vertical_displacement_m": max(s["parcels"][blue]["sample_rms_vertical_displacement_m"] for s in transport),
            }
        # A falling envelope can hide an upward-moving fold. Check individual
        # vertices as well as the bounds, and retain instantaneous velocity:
        # averaging away a contact impulse does not establish stable motion.
        checks["conveyor_stability"] = (transport_stability is not None
            and transport_stability["maximum_center_rise_m"] <= .015
            and transport_stability["maximum_top_rise_m"] <= .025
            and transport_stability["maximum_bottom_clearance_m"] <= .005
            and transport_stability["maximum_vertical_rms_speed_m_s"] <= .15
            and transport_stability["maximum_sample_vertex_rise_m"] <= .005
            and transport_stability["maximum_sample_rms_vertical_displacement_m"] <= .002)
        # The second parcel is untouched throughout this recipe. Checking only
        # the manipulated parcel misses slow self-excited oscillations that can
        # begin several seconds after an apparently quiet initial settling.
        unattended = None
        if self.unattended_names:
            quiet = [s for s in self.samples if s["time"] >= 1.]
            if quiet:
                drift = max(float(np.linalg.norm(np.asarray(s["parcels"][name]["center"])
                                                  - quiet[0]["parcels"][name]["center"]))
                            for s in quiet for name in self.unattended_names)
                height_change = max(float(np.ptp([s["parcels"][name]["height"] for s in quiet]))
                                    for name in self.unattended_names)
                rms = max(s["parcels"][name]["rms_speed_m_s"] for s in quiet for name in self.unattended_names)
                maximum = max(s["parcels"][name]["max_speed_m_s"] for s in quiet for name in self.unattended_names)
                rest_motion = max(s["parcels"][name]["rest_rms_displacement_m"] for s in quiet for name in self.unattended_names)
                sample_motion = max(s["parcels"][name]["sample_max_displacement_m"] for s in quiet for name in self.unattended_names)
                unattended = {"duration_s": quiet[-1]["time"] - quiet[0]["time"],
                              "maximum_center_drift_m": drift, "height_variation_m": height_change,
                              "maximum_rms_speed_m_s": rms, "maximum_vertex_speed_m_s": maximum,
                              "maximum_rest_rms_displacement_m": rest_motion,
                              "maximum_sample_vertex_displacement_m": sample_motion}
            # An instantaneous substep velocity can spike on a single corner
            # without visible motion. Retain it as a diagnostic and bound the
            # actual per-vertex motion between samples, plus accumulated shape
            # drift, to catch both flutter and slowly growing oscillations.
            checks["unattended_parcel_stability"] = (unattended is not None
                    and unattended["duration_s"] >= 2. and drift <= .003
                    and height_change <= .002 and rms <= .01
                    and rest_motion <= .002 and sample_motion <= .001)
        if drop_only:
            checks = {key: checks[key] for key in ("contact_penetration", "positive_tet_volumes")}
        return {"acceptance": checks, "accepted": all(checks.values()),
                "press_compression_fraction": compression, "height_recovery_fraction": recovery,
                "recovery_horizontal_drift_m": recovery_drift,
                "recovery_rms_speed_m_s": recovery_speed, "recovery_max_speed_m_s": recovery_max_speed,
                "central_top_compression_fraction": central_compression,
                "lift_hold_seconds": hold_time, "final_hold_clearance_m": final_hold_clearance,
                "maximum_soft_penetration_m": self.peak_penetration,
                "minimum_tet_volume_ratio": self.minimum_volume_ratio,
                "unattended_parcel": unattended, "conveyor_transfer": transfer,
                "conveyor_stability": transport_stability, "samples": self.samples}


def run(args, *, tick_observer=None, failure_observer=None):
    context = gobot.app.create_context()
    session = writer = None
    previous_raster_settings = None
    try:
        path = args.scene.expanduser().resolve()
        context.set_project_path(str(path.parent))
        root = context.load_scene("res://" + path.name)
        snapshot = prepare_snapshot(context.compile_physics_scene_snapshot())
        config = replace(default_config(), device=args.device, iterations=args.iterations, substeps=args.substeps)
        start = time.perf_counter()
        components = create(snapshot, asdict(config), drop_only=getattr(args, "drop_only", False))
        provider, controller = components.provider, components.controller
        session = SimulationSession(provider, controller=controller)
        initialization = time.perf_counter() - start
        steps = args.steps or int(round(controller.duration / config.fixed_time_step))
        for _ in range(args.warmup_steps):
            session.step()
        session.reset()
        initial = provider.snapshot_arrays(provider.scene_fields, (0,))
        metrics = TrialMetrics(snapshot)
        sync, fields = scene_sync(context, root, snapshot)
        fps = args.fps
        next_frame = 0.
        if args.video:
            import imageio.v2 as imageio
            previous_raster_settings = gobot.render.get_raster_settings()
            gobot.render.set_raster_settings(replace(previous_raster_settings,
                shadow_quality="high", shadow_distance=6.))
            video = args.video.expanduser().resolve()
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video, fps=fps, codec="libx264", quality=8)
        start = time.perf_counter()
        previous_phase = None
        failure = None
        completed_steps = 0
        try:
            for tick in range(steps):
                session.step()
                completed_steps = tick + 1
                phase = controller.phase
                completed_time = (tick + 1) * config.fixed_time_step
                observe = tick % 20 == 0 or phase != previous_phase or tick == steps - 1 or tick_observer is not None
                if observe:
                    state = provider.read_state()
                    sample = metrics.update(state, phase)
                    sample["tick"] = tick + 1
                    if phase != previous_phase or tick % 500 == 0:
                        print(f"Newton VBD {phase} | t={completed_time:.3f}s | "
                              f"penetration={sample['max_soft_penetration_m']*1000:.3f}mm", flush=True)
                    if metrics.minimum_volume_ratio <= 0:
                        raise RuntimeError(f"Newton VBD tetrahedral inversion at t={completed_time:.3f}s (minimum volume ratio {metrics.minimum_volume_ratio:.5f})")
                if writer is not None and completed_time >= next_frame:
                    values = provider.snapshot_arrays(fields, (0,))
                    sync(SimpleNamespace(environments=(0,), fields=tuple(values), buffer=values.__getitem__))
                    wide = phase == "settle_and_carton_drop"
                    eye, target, fov = (((1.35, -1.9, 1.55), (.0, -.1, .65), 55.) if wide else
                                        ((.60, -1.00, 1.04), (.16, -.375, .64), 50.))
                    if phase in ("lift", "hold"):
                        eye = (.8, .6, 1.6)  # Show the printed face as the pouch hangs.
                        target, fov = (.15, -.38, .83), 52.
                    if phase in ("transfer", "place_contact", "place", *SUPPORTED_TRANSPORT_PHASES):
                        # Keep the parcel large enough to inspect as it enters
                        # the belt and travels across the wider workcell.
                        center = sample["parcels"]["soft_mailer_blue"]["center"]
                        target = tuple(center)
                        eye = (center[0] + .95, center[1] - .90, center[2] + .70)
                        fov = 48.
                        if phase in ("release", "clear", "settle_after_release", "outfeed"):
                            # View from upstream and across the belt so the
                            # preceding cartons do not hide the soft parcel.
                            eye = (center[0] - .55, center[1] + .85, 1.16)
                            target, fov = (center[0], center[1], .62), 44.
                    rgb = gobot.render.capture_rgb(root=root, width=args.width, height=args.height,
                              eye=eye, target=target, fov_y=fov, z_far=8.)
                    from PIL import Image, ImageDraw, ImageFont
                    frame = Image.fromarray(rgb)
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, args.width, 46), fill=(18, 22, 28))
                    draw.text((16, 12), f"Newton VBD   |   {phase.replace('_', ' ')}   |   {completed_time:.2f} s",
                              font=ImageFont.load_default(size=18), fill=(235, 238, 242))
                    writer.append_data(np.asarray(frame))
                    next_frame += 1. / fps
                previous_phase = phase
                if tick_observer is not None and tick_observer(sample) is False:
                    break
        except Exception as error:
            failure = {"tick": completed_steps, "phase": controller.phase,
                       "error": f"{type(error).__name__}: {error}"}
            if failure_observer is not None:
                failure_observer(failure)
        elapsed = time.perf_counter() - start
        session.reset()
        restored = provider.snapshot_arrays(provider.scene_fields, (0,))
        reset_error = max(float(np.max(np.abs(initial[name] - restored[name]), initial=0.)) for name in initial)
        report = {"backend": "NewtonVBD", "device": str(provider.model.device), "config": asdict(config),
                  "versions": {"newton": provider.newton.__version__, "warp": provider.wp.__version__},
                  "initialization_seconds": initialization, "elapsed_seconds": elapsed, "steps": completed_steps, "requested_steps": steps,
                  "completed": failure is None and completed_steps == steps, "failure": failure,
                  "exact_effort_limits": provider.exact_effort_limits,
                  "simulation_seconds": completed_steps * config.fixed_time_step,
                  "simulation_seconds_per_wall_second": completed_steps * config.fixed_time_step / elapsed,
                  "reset_max_error": reset_error, **metrics.report(drop_only=controller.drop_only)}
        report["acceptance"]["completed"] = report["completed"]
        report["acceptance"]["reset"] = reset_error <= 1.e-6
        report["accepted"] = all(report["acceptance"].values())
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        if writer is not None:
            writer.close()
        if previous_raster_settings is not None:
            gobot.render.set_raster_settings(previous_raster_settings)
        if writer is not None:
            gobot.render.shutdown()
        if session is not None:
            session.close()
        context.clear_scene()
