"""Render recorded MuJoCo simulation states from the anchored robot's top view."""

import argparse
from pathlib import Path

import imageio_ffmpeg
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reaching_env import ReachingEnv, read_config
from collision import PLANE


def draw_segment(scene, a, b, radius, color):
    if np.linalg.norm(a - b) < 1e-8:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.full(3, radius), a,
                    np.eye(3).ravel(), np.asarray(color))
    mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
    scene.ngeom += 1


def draw_bezier(scene, points, display_y):
    def world(planar):
        xyz = np.asarray(planar) @ PLANE
        xyz[..., 1] = display_y
        return xyz

    polygon = world(points)
    for a, b in zip(polygon, polygon[1:]):
        draw_segment(scene, a, b, .0014, [.44, .70, .80, 1.])
    t = np.linspace(0., 1., 25)[:, None]
    curve = (1. - t)**2 * polygon[0] + 2. * t * (1. - t) * polygon[1] + t**2 * polygon[2]
    for a, b in zip(curve, curve[1:]):
        draw_segment(scene, a, b, .0023, [.15, .96, .87, 1.])
    for index, point in enumerate(polygon):
        color = [0., .95, .99, 1.] if index == 1 else [.48, .76, .87, 1.]
        radius = .014 if index == 1 else .007
        mj.mjv_initGeom(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, radius), point, np.eye(3).ravel(), np.asarray(color))
        scene.ngeom += 1


def record(env, trace, stem, video):
    fps, width, height = (video[key] for key in ("fps", "width", "height"))
    env.use_boxes(trace["boxes"])
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0., env.height, -.5]
    camera.azimuth, camera.elevation, camera.distance = 90., 0., 1.47
    goal = PLANE.T @ trace["goal"]
    options = mj.MjvOption()
    options.sitegroup[:] = 0
    t = trace["time_s"]
    has_bezier = "bezier_points_LT" in trace
    has_clearance = "obstacle_clearance_m" in trace
    frames_at = np.r_[np.arange(0., t[-1], 1 / fps), t[-1], np.full(fps // 2, t[-1])]
    indices = np.clip(np.searchsorted(t, frames_at, side="right") - 1, 0, len(t) - 1)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    writer = imageio_ffmpeg.write_frames(str(stem.with_suffix(".mp4")), (width, height), fps=fps,
                                        codec="libx264", quality=8, pix_fmt_out="yuv420p")
    writer.send(None)
    gif = []
    try:
        with mj.Renderer(env.m, height=height, width=width) as renderer:
            for idx in indices:
                env.set_state(trace["q"][idx])
                env.d.qvel[env.v_indices] = trace["dq"][idx]
                env.d.time = t[idx]
                mj.mj_forward(env.m, env.d)
                renderer.update_scene(env.d, camera=camera, scene_option=options)
                scene = renderer.scene
                marker = PLANE.T @ trace["tip"][idx]
                display_y = env.height + np.copysign(.13, scene.camera[0].pos[1] - env.height)
                marker[1] = goal[1] = display_y
                if has_bezier:
                    draw_bezier(scene, trace["bezier_points_LT"][idx], display_y)
                if has_clearance:
                    trail = np.vstack((trace["tip"][:idx:5], trace["tip"][idx])) @ PLANE
                    trail[:, 1] = display_y
                    for a, b in zip(trail, trail[1:]):
                        draw_segment(scene, a, b, .0013, [1., .38, .62, .8])
                for position, color, radius in ((goal, [.98, .8, .14, 1.], .017),
                                                (marker, [1., .17, .50, 1.], .013)):
                    mj.mjv_initGeom(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                                    np.full(3, radius), position, np.eye(3).ravel(), np.asarray(color))
                    scene.ngeom += 1
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                status = str(trace["status"]).upper() if idx == len(t) - 1 else "RUNNING"
                draw.rectangle((0, 0, width, 43), fill=(15, 25, 38))
                draw.text((16, 10), f"{status}   |   {t[idx]:.2f} s   |   M1-to-goal {100*trace['distance_m'][idx]:.1f} cm",
                          font=font, fill=(245, 250, 255))
                if has_clearance:
                    draw.text((width - 232, 10), f"Clearance {1000*trace['obstacle_clearance_m'][idx]:.1f} mm",
                              font=font, fill=(160, 245, 248))
                elif "tip_speed_m_s" in trace:
                    draw.text((width - 220, 10), f"M1 speed {100*trace['tip_speed_m_s'][idx]:.1f} cm/s",
                              font=font, fill=(160, 245, 248))
                sign_map = str(trace["output_sign_label"]) if "output_sign_label" in trace else None
                footer = 85 if sign_map else 62 if has_bezier else 36
                draw.rectangle((0, height - footer, width, height), fill=(15, 25, 38))
                legend = ("Pink: M1   Yellow: goal   Cyan: Bezier   Large cyan dot: P1" if has_bezier
                          else "Pink marker: M1       Yellow sphere: goal")
                if has_clearance:
                    legend = f"{trace['ablation_group']}  |  Pink: M1 and its path  |  Yellow: goal"
                draw.text((16, height - footer + 5), legend, font=font, fill=(245, 250, 255))
                if has_bezier:
                    p1, p2 = trace["bezier_points_LT"][idx, 1:] * 1000.
                    coords = f"P1 (L,T) = ({p1[0]:.1f}, {p1[1]:.1f}) mm    P2 = ({p2[0]:.1f}, {p2[1]:.1f}) mm"
                    draw.text((16, height - footer + 31), coords, font=font, fill=(125, 243, 246))
                if sign_map:
                    draw.text((16, height - 28), sign_map, font=font, fill=(245, 250, 255))
                writer.send(frame.tobytes())
                gif.append(frame.resize((576, 480), Image.Resampling.LANCZOS))
    finally:
        writer.close()
    durations = [round((i + 1) * 100 / fps) * 10 - round(i * 100 / fps) * 10 for i in range(len(gif))]
    gif[0].save(stem.with_suffix(".gif"), save_all=True, append_images=gif[1:], duration=durations, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config = read_config(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.trace) as trace:
        record(ReachingEnv(config, "A_fixed"), trace, args.out, config["video"])


if __name__ == "__main__":
    main()
