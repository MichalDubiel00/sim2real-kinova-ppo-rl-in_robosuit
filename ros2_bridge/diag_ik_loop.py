#!/usr/bin/env python3
"""
Warstwa A, pomiar W PETLI — odpowiednik training/diag_action_scale.py, ale
przez /compute_ik zamiast OSC.

Roznica wobec diag_ik_move.py (pojedynczy dojazd): tutaj co krok liczona jest
NOWA poza docelowa (biezacy TCP + delta) i wysylana zanim poprzedni dojazd
zdazy wyhamowac. To odpowiada temu, co robi polityka przy 20 Hz, i dopiero to
jest porownywalne z tabela z robosuite.

ODNIESIENIE — robosuite OSC_POSE, 20 Hz, akcja 1.0:
    os X: 0.01126 m/krok ustalone, 90 % po 12 krokach, sprzezenie  2.8 %
    os Y: 0.00722 m/krok,           90 % po  3 krokach, sprzezenie 13.4 %
    os Z: 0.01317 m/krok,           90 % po  2 krokach, sprzezenie  8.4 %

URUCHOMIENIE (bez condy):
    source /opt/ros/humble/setup.bash
    source ~/Batchelor/webots_ros2_spot/install/setup.bash
    python3 ros2_bridge/diag_ik_loop.py --axis z --delta 0.05 --steps 40
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

BASE_FRAME = "gen3_base_link"
EE_FRAME = "end_effector_link"
GROUP = "manipulator"
JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
CMD_TOPIC = "/kinova_joint_trajectory_controller/joint_trajectory"
AXES = {"x": 0, "y": 1, "z": 2}


class IkLoop(Node):
    def __init__(self, time_from_start):
        super().__init__("diag_ik_loop")
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, self)
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.pub = self.create_publisher(JointTrajectory, CMD_TOPIC, 10)
        self.tfs = time_from_start

    def spin_for(self, sec):
        """Spinuj przez `sec` — NIE time.sleep(), bo ten nie przetwarza wiadomosci."""
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.005)

    def read_tcp(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.02)
            try:
                tr = self.tfb.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time())
            except Exception:
                continue
            t, r = tr.transform.translation, tr.transform.rotation
            return np.array([t.x, t.y, t.z]), (r.x, r.y, r.z, r.w)
        return None, None

    def solve_ik(self, pos, quat):
        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = EE_FRAME
        req.ik_request.timeout.sec = 1
        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos)
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = map(float, quat)
        req.ik_request.pose_stamped = ps

        fut = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            return None, (res.error_code.val if res else None)
        js = res.solution.joint_state
        try:
            return [js.position[js.name.index(j)] for j in JOINTS], 1
        except ValueError:
            return None, None

    def send(self, joints):
        if self.pub.get_subscription_count() == 0:
            self.get_logger().warn("kontroler nie subskrybuje — pomijam publikacje")
            return False
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(q) for q in joints]
        pt.time_from_start.sec = int(self.tfs)
        pt.time_from_start.nanosec = int((self.tfs % 1) * 1e9)
        msg.points.append(pt)
        self.pub.publish(msg)
        return True

    def wait_for_controller(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="z", choices=list(AXES))
    ap.add_argument("--delta", type=float, default=0.05,
                    help="delta na krok [m]; 0.05 = output_max OSC przy akcji 1.0")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--rate", type=float, default=20.0, help="Hz — tak dziala polityka")
    ap.add_argument("--tfs", type=float, default=0.05, help="time_from_start [s]")
    args = ap.parse_args()

    rclpy.init()
    node = IkLoop(args.tfs)

    if not node.ik.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("/compute_ik niedostepne — czy dziala move_group?")
        return
    if not node.wait_for_controller():
        node.get_logger().error("kontroler nie subskrybuje CMD_TOPIC")
        return

    d = np.zeros(3)
    d[AXES[args.axis]] = args.delta
    period = 1.0 / args.rate
    ai = AXES[args.axis]

    pos, quat = node.read_tcp()
    if pos is None:
        node.get_logger().error("brak TF")
        return
    print(f"\nos {args.axis.upper()}  delta {args.delta:+.3f} m/krok  "
          f"{args.rate:.0f} Hz  time_from_start {args.tfs}s")
    print(f"start TCP {np.round(pos, 4)}\n")
    print(f"{'krok':>4} {'dx':>9} {'dy':>9} {'dz':>9} {'|d|':>9} {'IK':>4}")

    prev, rows, fails, first_fail = pos.copy(), [], 0, None
    for k in range(args.steps):
        t_start = time.time()
        cur, q = node.read_tcp(timeout=0.5)
        if cur is None:
            break
        joints, code = node.solve_ik(cur + d, q)
        if joints is None:
            fails += 1
            if first_fail is None:
                first_fail = len(rows)
            print(f"{k:4d}  IK FAIL (kod {code})")
        else:
            node.send(joints)
        node.spin_for(max(0.0, period - (time.time() - t_start)))
        new, _ = node.read_tcp(timeout=0.5)
        if new is None:
            break
        inc = new - prev
        rows.append(inc)
        print(f"{k:4d} {inc[0]:+9.5f} {inc[1]:+9.5f} {inc[2]:+9.5f} "
              f"{np.linalg.norm(inc):9.5f} {'ok' if joints else 'X':>4}")
        prev = new

    r = np.array(rows)
    # Okno ustalone: od kroku 3 (po stanie przejsciowym) do PIERWSZEJ porazki IK.
    # Poprzednia wersja brala ostatnie 8 krokow — czyli dokladnie strefe, w ktorej
    # ramie dochodzi do granicy zasiegu, przyrost spada i rosnie sprzezenie.
    # Pomiary zerowe to artefakty odczytu TF (ta sama wartosc dwa razy) — odrzucane.
    end = first_fail if first_fail is not None else len(r)
    sign = np.sign(args.delta) or 1.0
    idx = [i for i in range(3, end) if abs(r[i, ai]) > 1e-6]
    if len(idx) >= 5:
        win = r[idx]
        ss = float(np.median(win[:, ai]))
        others = [j for j in range(3) if j != ai]
        coup = float(np.mean(np.abs(win[:, others]))) / abs(ss) if ss else 0.0
        thr = 0.9 * abs(ss)
        i90 = next((i for i, v in enumerate(r[:, ai]) if abs(v) >= thr), None)
        REF = {"x": 0.01126, "y": 0.00722, "z": 0.01317}
        ref = REF[args.axis] * sign
        print(f"\n--- WYNIK, os {args.axis.upper()} ---")
        print(f"  okno ustalone     : kroki 3..{end - 1} ({len(idx)} probek)")
        print(f"  przyrost (mediana): {ss:+.5f} m/krok  (odch. std {np.std(win[:, ai]):.5f})")
        print(f"  robosuite OSC     : {ref:+.5f} m/krok")
        print(f"  stosunek Webots/robosuite : {abs(ss / ref):.2f}x")
        print(f"  krok do 90 % ustalonego   : {i90}")
        print(f"  sprzezenie na inne osie   : {coup:.1%}")
        print(f"  nieudane IK               : {fails}/{args.steps}"
              + (f"  (pierwsza @{first_fail})" if first_fail is not None else ""))
        if first_fail is not None:
            print("  UWAGA: ramie doszlo do granicy zasiegu — kroki po pierwszym\n"
                  "  IK FAIL sa odrzucone. Zmniejsz --steps albo --delta.")
    else:
        print("\nZa malo poprawnych probek do wyznaczenia wartosci ustalonej.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
