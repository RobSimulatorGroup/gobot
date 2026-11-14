"""Render saved tracking poses side by side, without advancing any physics.

The left panel shows recorded policy execution; the right shows the exact
reference poses at the same timestamps. Both views use the same camera.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(trajectory, evaluation, output, stride=2):
    trajectory = Path(trajectory)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if stride < 1:
        raise ValueError("stride must be positive")
    result = json.loads(Path(evaluation).read_text())
    with np.load(trajectory) as saved:
        trace = {name: saved[name] for name in saved.files}
    model = mujoco.MjModel.from_binary_path(str(trajectory.with_suffix(".mjb")))
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 600
    model.stat.extent = 4.0
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 135
    camera.elevation = -18
    camera.distance = 3.2
    camera.lookat[:] = (0, 0, 0.8)
    options = mujoco.MjvOption()
    options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    options.sitegroup[:] = 0
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 25)
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 16)
    step_dt = float(trace["step_dt"])
    fps = 1 / (step_dt * stride)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE,
    )
    indices = list(range(0, len(trace["qpos"]), stride))
    if indices[-1] != len(trace["qpos"]) - 1:
        indices.append(len(trace["qpos"]) - 1)
    poster_index = min(len(indices) // 2, int(10 * fps))
    last_image = None
    try:
        with mujoco.Renderer(model, height=600, width=640) as renderer:
            for count, index in enumerate(indices):
                time_s = float(trace["time_s"][index])
                root = trace["reference_qpos"][index, :3]
                camera.lookat[:] = (root[0], root[1], 0.8)
                panels = []
                for is_reference, qpos in enumerate(
                    (trace["qpos"][index], trace["reference_qpos"][index])
                ):
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    if is_reference:
                        for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                            if (geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                                    and geom.objid >= 0
                                    and model.geom_bodyid[geom.objid] > 0):
                                geom.rgba[:3] = (0.35, 0.75, 0.58)
                    panels.append(Image.fromarray(renderer.render()))
                image = Image.new("RGB", (1280, 720), (13, 20, 34))
                image.paste(panels[0], (0, 80))
                image.paste(panels[1], (640, 80))
                draw = ImageDraw.Draw(image)
                draw.text((22, 12), "MJLAB POLICY  |  30,000 updates", font=title_font,
                          fill=(224, 234, 246))
                draw.text((662, 12), "REFERENCE  |  dance2_subject1", font=title_font,
                          fill=(115, 223, 174))
                draw.text((22, 48), "Closed-loop physics rollout", font=small,
                          fill=(159, 178, 201))
                draw.text((662, 48), "Kinematic target at the same time", font=small,
                          fill=(159, 178, 201))
                draw.line((640, 80, 640, 680), fill=(47, 65, 87), width=2)
                error = float(trace["anchor_position_error_m"][index])
                draw.text((22, 691), f"t = {time_s:6.2f} / {result['reference_duration_s']:.2f} s"
                          f"     torso position error = {error * 100:.1f} cm",
                          font=small, fill=(220, 231, 242))
                draw.text((910, 691), "1x speed  |  no resets", font=small,
                          fill=(159, 178, 201))
                if count == len(indices) - 1:
                    label = ("STOPPED: " + ", ".join(result["termination_reasons"])) \
                        if result["terminated"] else "FULL MOTION COMPLETED"
                    draw.rectangle((18, 620, 1262, 674), fill=(13, 20, 34))
                    draw.text((34, 635), label, font=font,
                              fill=(255, 159, 120) if result["terminated"] else (115, 223, 174))
                assert encoder.stdin is not None
                encoder.stdin.write(image.tobytes())
                if count == poster_index:
                    image.save(output.with_suffix(".png"))
                if count % 250 == 0:
                    print(f"Rendering {count + 1}/{len(indices)}: {time_s:.1f}s", flush=True)
                last_image = image
        # Freeze the terminal state for two seconds so its reason is legible.
        assert encoder.stdin is not None and last_image is not None
        for _ in range(round(2 * fps)):
            encoder.stdin.write(last_image.tobytes())
        last_image.save(output.with_name(output.stem + "-final.png"))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    finally:
        if encoder.poll() is None:
            if encoder.stdin is not None:
                encoder.stdin.close()
            encoder.terminate()
            encoder.wait()
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    render(**vars(parser.parse_args()))
