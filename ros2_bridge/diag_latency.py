#!/usr/bin/env python3
"""Loop latency: policy -> ROS 2 -> controller -> ROS 2 -> observation.

robosuite has zero dead time; one step of delay at 20 Hz is a full 50 ms. Measure
before randomising: a guessed range already failed once (gain_rnd, docs 6.29).

Measures three things, keep them apart:
  1. observation age  = now - header.stamp of /joint_states when read
  2. command dead time = command publish -> first reaction in /joint_states
  3. jitter of both; constant delay is learnable, variable delay is not

Run in a ROS terminal (not conda):
    source /opt/ros/humble/setup.bash        # not under `set -u`
    python3 ros2_bridge/diag_latency.py --tor twist --prob 20
Commands are tiny: the arm should twitch, not travel.
"""
import argparse
import time
import statistics as st

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger

CMD_TOPIC = "/kinova_joint_trajectory_controller/joint_trajectory"
# Command frame must be robot_link_command_frame from moveit_servo_config.yaml
# (gen3_base_link, NOT base_link); otherwise commands are silently dropped (09.10).
BASE_FRAME = "gen3_base_link"
TWIST_TOPIC = "/servo_node/delta_twist_cmds"
STAWY = [f"joint_{i}" for i in range(1, 7)]
KROK_STEROWANIA = 0.05          # 20 Hz, as trained


class Pomiar(Node):
    def __init__(self, tor: str):
        super().__init__("diag_latency")
        self.tor = tor
        # Webots /joint_states stamps are SIM time, time.time() is Unix epoch: use
        # get_clock() everywhere and enable use_sim_time when /clock exists.
        self.hist = []            # (t_recv, t_header, [q1..q6], [v1..v6])
        self.ma_predkosci = None  # does /joint_states provide .velocity
        self.create_subscription(JointState, "/joint_states", self._js, 50)
        self.twist_pub = self.create_publisher(TwistStamped, TWIST_TOPIC, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, CMD_TOPIC, 10)
        self.start_cli = self.create_client(Trigger, "/servo_node/start_servo")

    def czekaj_na_discovery(self, sekundy=3.0):
        """Give DDS time to discover peers; otherwise subscriber counts and topic
        lists are random and commands go nowhere (09.10)."""
        t0 = self.teraz()
        while self.teraz() - t0 < sekundy:
            rclpy.spin_once(self, timeout_sec=0.05)

    def uruchom_servo(self, czekaj=3.0) -> str:
        """MoveIt Servo ignores twists until /servo_node/start_servo."""
        if not self.start_cli.wait_for_service(timeout_sec=czekaj):
            return "no /servo_node/start_servo service"
        fut = self.start_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=czekaj)
        if fut.result() is None:
            return "no response"
        return f"start_servo: {'OK' if fut.result().success else 'odmowa'} ({fut.result().message})"

    def teraz(self) -> float:
        """ROS clock (sim time in Webots), for stamps comparable with headers."""
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def sciana() -> float:
        """Monotonic wall clock, for loop durations. With use_sim_time the ROS clock
        reads 0 until the first /clock and then jumps (09.10)."""
        return time.monotonic()

    def czekaj_na_zegar(self, limit=5.0):
        """Wait until the ROS clock starts (starts at zero with use_sim_time)."""
        t0 = self.sciana()
        a = self.teraz()
        while self.sciana() - t0 < limit:
            rclpy.spin_once(self, timeout_sec=0.05)
            b = self.teraz()
            if b > 0 and b != a:
                return True
            a = b
        return False

    def _js(self, m: JointState):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(s in idx for s in STAWY):
            return
        # joint order is 1,2,4,5,3,6: map by NAME
        q = [float(m.position[idx[s]]) for s in STAWY]
        ma_v = bool(m.velocity) and len(m.velocity) > max(idx[s] for s in STAWY)
        v = [float(m.velocity[idx[s]]) for s in STAWY] if ma_v else [0.0] * 6
        if self.ma_predkosci is None:
            self.ma_predkosci = ma_v
        t_hdr = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.hist.append((self.teraz(), t_hdr, q, v))
        if len(self.hist) > 4000:
            del self.hist[:2000]


    # ---- measurements 1 and 3: observation age and stream regularity ----
    def zmierz_wiek(self, sekundy=3.0):
        self.hist.clear()
        t0 = self.sciana()
        while self.sciana() - t0 < sekundy:
            rclpy.spin_once(self, timeout_sec=0.01)
        if len(self.hist) < 10:
            return None
        wiek = [(t - h) * 1000 for t, h, _, _ in self.hist if h > 0]
        odstepy = [(self.hist[i][0] - self.hist[i - 1][0]) * 1000
                   for i in range(1, len(self.hist))]
        return dict(n=len(self.hist), wiek=wiek, odstepy=odstepy,
                    hz=len(self.hist) / sekundy)

    # ---- measurement 2: command dead time ----
    def _szum(self, sekundy=1.5):
        """Detection threshold from standstill noise, on POSITIONS (Webots
        /joint_states has no .velocity)."""
        self.hist.clear()
        t0 = self.sciana()
        while self.sciana() - t0 < sekundy:
            rclpy.spin_once(self, timeout_sec=0.01)
        if len(self.hist) < 10:
            return None
        kol = list(zip(*[q for _, _, q, _ in self.hist]))
        # 5 sigma of position noise, at least 2e-4 rad
        return [max(5 * st.pstdev(k), 2e-4) for k in kol]

    def _wyslij(self, znak):
        if self.tor == "twist":
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = BASE_FRAME
            m.twist.linear.x = 0.05 * znak      # small, short command
            self.twist_pub.publish(m)
        else:
            if not self.hist:
                return
            q = list(self.hist[-1][2])      # CURRENT pose
            msg = JointTrajectory()
            msg.joint_names = STAWY
            pt = JointTrajectoryPoint()
            # Positions RELATIVE to the current pose (absolute zeros would send
            # every joint to zero).
            pt.positions = q
            pt.positions[0] = q[0] + 0.01 * znak
            pt.time_from_start.sec = 0
            pt.time_from_start.nanosec = int(0.05 * 1e9)
            msg.points = [pt]
            self.traj_pub.publish(msg)

    def zmierz_martwy_czas(self, prob, prog, limit=1.0):
        wyniki, nieudane = [], 0
        for i in range(prob):
            znak = 1.0 if i % 2 == 0 else -1.0
            # let the arm settle
            t0 = self.sciana()
            while self.sciana() - t0 < 0.5:
                rclpy.spin_once(self, timeout_sec=0.01)
            baza = len(self.hist)
            t_pub = self.teraz()
            # Servo needs a stream, not a single packet
            t0 = self.sciana()
            wykryte = None
            while self.sciana() - t0 < limit:
                self._wyslij(znak)
                rclpy.spin_once(self, timeout_sec=0.005)
                if not self.hist[:baza]:
                    continue
                q0 = self.hist[baza - 1][2] if baza else self.hist[0][2]
                for t, _, q, _ in self.hist[baza:]:
                    if any(abs(q[j] - q0[j]) > prog[j] for j in range(6)):
                        wykryte = (t - t_pub) * 1000
                        break
                if wykryte is not None:
                    break
            if wykryte is None:
                nieudane += 1
            else:
                wyniki.append(wykryte)
        return wyniki, nieudane


def opis(nazwa, dane, jednostka="ms"):
    if not dane:
        print(f"{nazwa:28} NO DATA")
        return
    d = sorted(dane)
    med = st.median(d)
    p10, p90 = d[int(0.1 * len(d))], d[min(int(0.9 * len(d)), len(d) - 1)]
    print(f"{nazwa:28} median {med:7.1f} {jednostka}   p10-p90 {p10:6.1f}-{p90:6.1f}   "
          f"min-max {d[0]:6.1f}-{d[-1]:6.1f}   n={len(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tor", choices=["twist", "trajektoria"], default="twist",
                    help="command path (the hardware model uses the twist path)")
    ap.add_argument("--prob", type=int, default=20, help="dead-time repetitions")
    a = ap.parse_args()

    rclpy.init()
    n = Pomiar(a.tor)
    print("waiting for DDS discovery (3 s)...")
    n.czekaj_na_discovery()
    # With /clock the node clock must match the header stamps.
    ma_clock = any(t == "/clock" for t, _ in n.get_topic_names_and_types())
    if ma_clock:
        try:
            n.set_parameters([rclpy.parameter.Parameter(
                "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
        except Exception as e:                      # noqa: BLE001
            print(f"  (could not set use_sim_time: {e})")
    print(f"path: {a.tor}, policy control step: {KROK_STEROWANIA*1000:.0f} ms")
    print(f"/clock present: {'yes -> use_sim_time=true' if ma_clock else 'no -> wall clock'}")
    if ma_clock and not n.czekaj_na_zegar():
        print("  ⚠ ROS clock not advancing - is the simulation PAUSED in Webots?")
    sub = (n.twist_pub.get_subscription_count() if a.tor == "twist"
           else n.traj_pub.get_subscription_count())
    tnazwa = TWIST_TOPIC if a.tor == "twist" else CMD_TOPIC
    print(f"subscribers of {tnazwa}: {sub}")
    if a.tor == "twist":
        print(n.uruchom_servo())
    if sub == 0:
        print("  ⚠ NOBODY LISTENS on this topic - the dead-time measurement is meaningless.")
        print("    twist needs a running MoveIt Servo; check `ros2 node list`.")
    print()

    print("--- 1/3: observation age and /joint_states regularity (3 s) ---")
    w = n.zmierz_wiek()
    if w is None:
        print("NO /joint_states - is the simulation or hardware running?")
        rclpy.shutdown(); return
    print(f"{'czestosc strumienia':28} {w['hz']:7.1f} Hz   (n={w['n']})")
    print(f"{'/joint_states ma .velocity':28} {'tak' if n.ma_predkosci else 'NIE (detekcja po pozycji)'}")
    opis("observation age", w["wiek"])
    opis("inter-message gap", w["odstepy"])

    print("\n--- 2/3: detection threshold from standstill noise (1.5 s) ---")
    prog = n._szum()
    if prog is None:
        print("not enough data for a threshold"); rclpy.shutdown(); return
    print("threshold per joint [rad/s]:", " ".join(f"{p:.4f}" for p in prog))

    print(f"\n--- 3/3: command dead time, {a.prob} repetitions ---")
    wyniki, nieudane = n.zmierz_martwy_czas(a.prob, prog)
    opis("command dead time", wyniki)
    if wyniki:
        med = st.median(wyniki) / 1000.0
        print(f"{'in control steps':28} {med/KROK_STEROWANIA:7.2f} steps of 50 ms")
        rozrzut = (max(wyniki) - min(wyniki)) / 1000.0
        print(f"{'rozrzut martwego czasu':28} {rozrzut*1000:7.1f} ms = "
              f"{rozrzut/KROK_STEROWANIA:.2f} steps")
        print("\nA CONSTANT dead time can be absorbed by the policy (phase shift).")
        print("A VARIABLE one cannot. If the spread is comparable to the median,")
        print("delay randomisation in training is REQUIRED, not optional.")
    if nieudane:
        print(f"\nNOTE: {nieudane}/{a.prob} trials without a detected reaction - the command may be")
        print("too small, the path inactive (Servo needs a stream) or the controller not listening.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
