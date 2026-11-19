#!/usr/bin/env python3
"""
rosbag2 (sqlite3) -> npz with arm joint states, twist commands and /goto_position events.
Run with the system ROS Python, not conda:

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 ros2_bridge/tools/bag_to_npz.py logs/hw/2026-11-18/step out.npz

npz: t (bag receive time [s]), stamp, pos[6], vel[6], ct, cmd[6] (lin xyz, ang xyz), goto (t, name).
"""

import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String, Bool

JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def main(bag, out):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    T, S, P, V, CT, C, G = [], [], [], [], [], [], []
    idx = None
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == "/joint_states":
            m = deserialize_message(data, JointState)
            if idx is None:
                idx = [list(m.name).index(j) for j in JOINTS]
            T.append(t); S.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            P.append([m.position[i] for i in idx])
            V.append([m.velocity[i] for i in idx] if len(m.velocity) == len(m.name) else [np.nan] * 6)
        elif topic == "/twist_controller/commands":
            tw = deserialize_message(data, TwistStamped).twist
            CT.append(t); C.append([tw.linear.x, tw.linear.y, tw.linear.z, tw.angular.x, tw.angular.y, tw.angular.z])
        elif topic == "/goto_position":
            G.append((t * 1e-9, deserialize_message(data, String).data))
        elif topic == "/goto_position/result":
            G.append((t * 1e-9, f"result {deserialize_message(data, Bool).data}"))
    np.savez(out, t=np.array(T) * 1e-9, stamp=np.array(S), pos=np.array(P), vel=np.array(V),
             ct=np.array(CT) * 1e-9, cmd=np.array(C).reshape(-1, 6), goto=np.array(G, dtype=object))
    print(f"{len(T)} joint states, {len(CT)} twists, {len(G)} goto events -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
