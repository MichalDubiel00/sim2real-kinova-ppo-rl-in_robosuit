#!/usr/bin/env python3
"""
Webots demo recording: Gen3 wrist camera (with ArUco overlay) | Spot Kinect, to MP4.
Runs beside bridge_node.py without affecting its loop.

  ~/miniconda3/envs/robosuite/bin/python ros2_bridge/record_demo.py --out docs/video/x.mp4 --seconds 60
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aruco_board as AB


def to_bgr(m):
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
    if m.encoding in ("bgra8", "bgr8"):
        return np.ascontiguousarray(a[:, :, :3])
    return cv2.cvtColor(np.ascontiguousarray(a[:, :, :3]), cv2.COLOR_RGB2BGR)


def label(img, text, y=28, scale=0.7):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()

    rclpy.init()
    n = Node("record_demo")
    st = {"wrist": None, "kinect": None, "info": None}
    n.create_subscription(Image, "/Gen3/kinova_color/image_color", lambda m: st.__setitem__("wrist", m), 1)
    n.create_subscription(CameraInfo, "/Gen3/kinova_color/camera_info", lambda m: st.__setitem__("info", m), 1)
    n.create_subscription(Image, "/Spot/kinect_color/image_color", lambda m: st.__setitem__("kinect", m), 1)

    writer, last, frames, t0 = None, None, 0, time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(n, timeout_sec=0.01)
        w = st["wrist"]
        if w is None or st["kinect"] is None or st["info"] is None:
            continue
        stamp = (w.header.stamp.sec, w.header.stamp.nanosec)
        if stamp == last:
            continue
        last = stamp
        wimg = to_bgr(w)
        K = np.array(st["info"].k).reshape(3, 3)
        D = np.array(st["info"].d) if len(st["info"].d) else np.zeros(5)
        r = AB.detect(wimg, K, D)
        if r is not None:
            cands, corners = r
            p, R, err = min(cands, key=lambda c: c[2])
            cv2.aruco.drawDetectedMarkers(wimg, [corners.reshape(1, 4, 2).astype(np.float32)],
                                          np.array([[AB.MARKER_ID]]))
            rv, _ = cv2.Rodrigues(R)
            cv2.drawFrameAxes(wimg, K, D, rv, p, 0.03)
            label(wimg, f"ArUco id {AB.MARKER_ID}: {100 * np.linalg.norm(p):.1f} cm from the camera", y=58, scale=0.6)
        else:
            label(wimg, "ArUco: out of view", y=58, scale=0.6)
        label(wimg, "Gen3 wrist camera")
        k = to_bgr(st["kinect"])
        k = cv2.resize(k, (int(k.shape[1] * wimg.shape[0] / k.shape[0]), wimg.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        label(k, "Kinect on the Spot body")
        sim_t = w.header.stamp.sec + w.header.stamp.nanosec * 1e-9
        frame = np.hstack([wimg, k])
        label(frame, f"Webots | arm_d policy (robosuite) via ROS 2 | t_sim {sim_t:7.2f} s",
              y=frame.shape[0] - 12, scale=0.6)
        if writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        frames += 1
    if writer is not None:
        writer.release()
    print(f"saved {frames} frames -> {args.out}")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
