"""
Joint poses on the Gen3 through the Kortex API, without the Steam Deck (run on max1 or the laptop).

    read                     current joint angles + Kortex tool pose; --log FILE appends JSON lines
                             (Enter = next sample, q = quit) for the FK check (checklist point 6)
    list                     saved REACH_JOINT_ANGLES actions
    goto-angles A1 .. A6     move to joint angles [deg, Web App convention] WITHOUT saving them;
                             slow (--speed deg/s of the largest joint), asks for confirmation
    goto NAME                run a saved action (same as the lab's /goto_position)
    save NAME A1 .. A6       save a REACH_JOINT_ANGLES action; refuses an existing name

Ctrl+C during a move sends Stop(). Keep the E-stop in reach. Opens its own session next to the
lab driver; does not change servoing mode, limits or admittance.

    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    python3 kortex_poses.py read
    python3 kortex_poses.py goto-angles 0 84.2 231.6 179.9 244.3 180 --speed 10
"""

import argparse
import json
import threading
import time

from kortex_api.RouterClient import RouterClient
from kortex_api.TCPTransport import TCPTransport
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2, Base_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

CONTINUOUS = (0, 3, 5)  # j1, j4, j6 of the 6DOF arm (shortest path is taken by Kortex)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def joints(base):
    return [j.value for j in base.GetMeasuredJointAngles().joint_angles]


def find_action(base, name):
    req = Base_pb2.RequestedActionType(action_type=Base_pb2.REACH_JOINT_ANGLES)
    for a in base.ReadAllActions(req).action_list:
        if a.name == name:
            return a
    return None


def angles_of(action):
    return [j.value for j in action.reach_joint_angles.joint_angles.joint_angles]


def make_action(name, target, duration=None):
    action = Base_pb2.Action()
    action.name = name
    action.application_data = ""
    for i, v in enumerate(target):
        ja = action.reach_joint_angles.joint_angles.joint_angles.add()
        ja.joint_identifier = i
        ja.value = v % 360.0
    if duration is not None:
        action.reach_joint_angles.constraint.type = Base_pb2.JOINT_CONSTRAINT_DURATION
        action.reach_joint_angles.constraint.value = duration
    return action


def print_plan(cur, target):
    print("joint   now[deg]  target   delta (Kortex path)")
    for i, (c, t) in enumerate(zip(cur, target)):
        d = wrap180(t - c) if i in CONTINUOUS else (t % 360 - c % 360)
        flag = "  <- near 180, direction ambiguous" if i in CONTINUOUS and abs(abs(d) - 180) < 5 else ""
        print(f"  j{i + 1}  {wrap180(c):8.1f}  {wrap180(t):7.1f}  {d:+7.1f}{flag}")


def max_delta(cur, target):
    return max(abs(wrap180(t - c)) if i in CONTINUOUS else abs(t % 360 - c % 360)
               for i, (c, t) in enumerate(zip(cur, target)))


def run(base, start_fn, timeout):
    done, result = threading.Event(), {}

    def cb(n):
        if n.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            result["event"] = Base_pb2.ActionEvent.Name(n.action_event)
            result["abort"] = n.abort_details
            done.set()

    handle = base.OnNotificationActionTopic(cb, Base_pb2.NotificationOptions())
    t0 = time.time()
    try:
        start_fn()
        if not done.wait(timeout):
            base.Stop()
            print(f"timeout {timeout:.0f} s -> Stop()")
            return False
    except KeyboardInterrupt:
        base.Stop()
        print("\nCtrl+C -> Stop()")
        return False
    finally:
        base.Unsubscribe(handle)
    ok = result["event"] == "ACTION_END"
    print(f"{result['event']} after {time.time() - t0:.1f} s" +
          ("" if ok else f" (abort details {result['abort']})"))
    return ok


def confirm(msg):
    return input(f"{msg} Type 'yes' to move: ").strip() == "yes"


def check_servoing(base):
    mode = base.GetServoingMode().servoing_mode
    if mode != Base_pb2.SINGLE_LEVEL_SERVOING:
        raise SystemExit(f"servoing mode {Base_pb2.ServoingMode.Name(mode)}: high-level actions "
                         "would fail; not changing it (lab driver/trajectory in use?)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read", "list", "goto-angles", "goto", "save"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--ip", default="192.168.50.9")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--speed", type=float, default=10.0, help="[deg/s] of the largest joint (goto-angles)")
    ap.add_argument("--log", help="read: append samples as JSON lines")
    a = ap.parse_args()

    tcp = TCPTransport()
    router = RouterClient(tcp, RouterClient.basicErrorCallback)
    tcp.connect(a.ip, 10000)
    info = Session_pb2.CreateSessionInfo()
    info.username, info.password = a.user, a.password
    info.session_inactivity_timeout = 60000
    info.connection_inactivity_timeout = 2000
    sess = SessionManager(router)
    sess.CreateSession(info)
    base = BaseClient(router)
    try:
        if a.cmd == "read":
            cyc = BaseCyclicClient(router)
            while True:
                q = joints(base)
                p = base.GetMeasuredCartesianPose()
                fb = cyc.RefreshFeedback()
                rec = {"t": time.time(), "joints_deg": q,
                       "tool_pose": [p.x, p.y, p.z, p.theta_x, p.theta_y, p.theta_z],
                       "joint_vel_deg_s": [act.velocity for act in fb.actuators][:6],
                       "joint_torque_nm": [act.torque for act in fb.actuators][:6]}
                print("joints wrap180 [deg]:", [round(wrap180(v), 1) for v in q])
                print("tool pose (base) [m, deg]:", [round(v, 4) for v in rec["tool_pose"]])
                if a.log:
                    with open(a.log, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"  -> {a.log}")
                if not a.log or input("Enter = next sample, q = quit: ").strip() == "q":
                    break
        elif a.cmd == "list":
            req = Base_pb2.RequestedActionType(action_type=Base_pb2.REACH_JOINT_ANGLES)
            for act in base.ReadAllActions(req).action_list:
                print(f"{act.name:12s}", [round(v, 1) for v in angles_of(act)])
        elif a.cmd == "goto-angles":
            target = [float(v) for v in a.args]
            assert len(target) == 6, "need 6 angles"
            check_servoing(base)
            cur = joints(base)
            print_plan(cur, target)
            dur = max(max_delta(cur, target) / a.speed, 2.0)
            if confirm(f"Duration {dur:.1f} s."):
                run(base, lambda: base.ExecuteAction(make_action("kortex_poses_tmp", target, dur)),
                    dur + 15.0)
                print("now:", [round(wrap180(v), 1) for v in joints(base)])
        elif a.cmd == "goto":
            act = find_action(base, a.args[0])
            if act is None:
                raise SystemExit(f"no saved action '{a.args[0]}'")
            check_servoing(base)
            print_plan(joints(base), angles_of(act))
            if confirm("Saved-action speed (Angular Trajectory limit)."):
                run(base, lambda: base.ExecuteActionFromReference(act.handle), 60.0)
                print("now:", [round(wrap180(v), 1) for v in joints(base)])
        elif a.cmd == "save":
            name, target = a.args[0], [float(v) for v in a.args[1:]]
            assert len(target) == 6, "need NAME + 6 angles"
            if find_action(base, name) is not None:
                raise SystemExit(f"action '{name}' exists; not overwriting")
            base.CreateAction(make_action(name, target))
            print(f"saved '{name}':", [round(v % 360, 1) for v in target])
    finally:
        sess.CloseSession()
        router.SetActivationStatus(False)
        tcp.disconnect()


if __name__ == "__main__":
    main()
