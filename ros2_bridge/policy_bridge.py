#!/usr/bin/env python3
"""
URUCHOMIENIE (conda robosuite — tam jest torch/SB3, rclpy dziala bo Python 3.10):
  source /opt/ros/humble/setup.bash
  source ~/Batchelor/webots_ros2_spot/install/setup.bash
  python3 ros2_bridge/policy_bridge.py --dry-run --steps 100
"""

import sys
sys.modules["tensorflow"] = None   # PRZED SB3 — inaczej segfault (docs 7.6)

import argparse
import os
import time

import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped, TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs import (State, BASE_FRAME, EE_FRAME, JOINTS, N_BUTTONS, quat2mat,
                 quat_mul, quat_conj, axisangle2quat, QUAT_SITE_REL, GRIP_OFFSET)

# MODEL NA SPRZET (docs 6.40, n=100: 92/100 w obwiedni sprzetowej) — trenowany
# z limitami wg instrukcji Gen3 i E-stopem 20 N / 6 mm. Poprzedni domyslny
# (sweep/ft_sc1_v23_long, 15.91M) byl uczony BEZ limitow predkosci.
MODEL = ("logs/board_6dof/kinova_ft_lr2/checkpoints/ppo_board_28023552_steps.zip")
GROUP = "manipulator"
CMD_TOPIC = "/kinova_joint_trajectory_controller/joint_trajectory"
TWIST_TOPIC = "/servo_node/delta_twist_cmds"

# --- droga przez MoveIt Servo (regulator kartezjanski, jak OSC) ---------------
# Servo przyjmuje twist BEZWYMIAROWY [-1,1] (command_in_type: "unitless")
# w ramce gen3_base_link i sam zamyka petle przez jakobian. Tym sie rozni od
# IK + JointTrajectory, ktore interpoluje w przestrzeni STAWOW i przez to ma
# wzmocnienie zalezne od osi i konfiguracji (docs 6.28).
#
# UWAGA NA SENS KOMENDY. Przy IK zadawalismy POZE i regulator dochodzil do niej
# z opoznieniem — robosuite realizuje ok. 26 % zadanej delty na krok. Servo
# zadaje PREDKOSC i wykonuje ja wprost, wiec komenduje sie to, co robosuite
# faktycznie OSIAGA, a nie to, o co prosi.
SERVO_PUBLISH_PERIOD = 0.01   # publish_period z moveit_servo_config.yaml [s];
                              # Servo liczy nim przyrost pozycji (dt calkowania)
SERVO_SCALE_LIN = 5.0     # scale.linear z moveit_servo_config.yaml [m/s]
SERVO_SCALE_ROT = 2.0     # scale.rotational [rad/s]
# Wartosci dobrane pomiarowo (docs 6.29). Servo nasyca sie na 0.32x tempa
# robosuite dla translacji i 0.10x dla obrotu — wiecej sie nie da wycisnac,
# wiec to NIE jest dopasowanie, tylko maksimum osiagalne ta droga.
TARGET_LIN = 1.00         # unitless 0.20; powyzej komenda nie daje juz przyrostu
TARGET_ROT = 1.99         # unitless 0.995; sufit zakresu [-1,1]

# output_max z konfiguracji OSC_POSE (robosuite .../composite/basic.json).
# SKALIBROWANE pod mechanizm Webotsa (IK + JointTrajectoryController), a NIE
# przepisane z konfiguracji OSC. Kryterium: zrealizowany ruch na krok ma sie
# zgadzac z robosuite (0.01295 m/krok, 4.05 st./krok od pozy home).
#
#   tfs    transl. +Z   x robosuite   obrot +X   x robosuite
#   0.02     0.00992        0.77        4.10st      1.01
#   0.03     0.00962        0.74        2.87st      0.71
#   0.05     0.00718        0.55        2.76st      0.68
#   0.08     0.00455        0.35        2.40st      0.59
#
# tfs = 0.02 trafia w obrot; brakujaca translacje nadrabia POS_SCALE
# (mechanizm jest liniowy w amplitudzie — diag_action_scale):
#   POS_SCALE 0.050 -> 0.01090 m/krok (0.84x)
#   POS_SCALE 0.065 -> 0.01359 m/krok (1.05x)
POS_SCALE = 0.065     # m na jednostke akcji (OSC ma 0.05 — patrz wyzej)
POS_SCALE_CAL = POS_SCALE  # wartosc skalibrowana przy OSC 0.05; --pos-max skaluje proporcjonalnie
#   Obrot mierzony KATEM GEODEZYJNYM pelnej orientacji (arccos((tr(Ra^T Rb)-1)/2)),
#   a nie przechyleniem osi Z — ta druga miara ZANIZA obrot wokol osi narzedzia
#   i dala wczesniej falszywe 1.01x dla +X. Prawdziwe wzmocnienia przy tfs=0.02,
#   20 krokow od pozy home:
#     os    robosuite   most    stosunek   IK-bledy
#     +X      3.946      3.416    0.87        0
#     -X      3.975      3.570    0.90        0
#     +Y      4.143      2.961    0.71        7/20
#     -Y      3.299      2.864    0.87        4/20
#     +Z      5.138      2.797    0.54        0
#     -Z      5.137      2.871    0.56        0
#   Jedna skala NIE dopasuje sie do zakresu 0.54-0.90. Robosuite sam jest
#   anizotropowy (+Y 4.14 vs -Y 3.30), wiec wzmocnienia zaleza od konfiguracji
#   i powyzsze liczby sa lokalne dla pozy home.
ROT_SCALE = 0.5       # rad na jednostke akcji — NIEDOPASOWANY, patrz tabela wyzej
# ROT_SCALE MUSI byc rowny osc_rot_output_max z treningu modelu (docs 6.35):
# modele do 14.09 — 0.5, modele trenowane pod limit 0.8727 — 0.1. Ustawiane
# flaga --rot-max; tabela kalibracji wyzej byla mierzona przy 0.5.
RATE_HZ = 20.0        # tak byla trenowana polityka; nienegocjowalne
TIME_FROM_START = 0.02

# Poza resetu z robosuite — ButtonBoardEnv daje ZAWSZE te same katy (std = 0
# na wszystkich 6 stawach, 8 ziaren). Polityka nie widziala w treningu
# zadnego innego startu, wiec most musi tu dojechac przed oddaniem sterowania.
# TCP wychodzi wtedy [0.4265, 0.0014, 0.0396] w ramce BASE_FRAME.
# Kolejnosc = JOINTS z obs.py.
HOME_QPOS = [0.0, 0.65, -1.89, 0.0, -0.6, -1.5708]
HOME_TIME = 3.0       # NIE 0.05 — to przejazd 0.46 m, nie krok sterowania


class PolicyBridge(State):
    def __init__(self, model_path, dry_run=True):
        super().__init__()
        self.dry_run = dry_run
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.pub = self.create_publisher(JointTrajectory, CMD_TOPIC, 10)
        self.twist_pub = self.create_publisher(TwistStamped, TWIST_TOPIC, 10)

        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device="cpu")

    # ------------------------------------------------------------------ IK --
    def solve_ik(self, pos, quat):
        """
          - req.ik_request.{group_name, ik_link_name, timeout.sec}
          - pose_stamped z header.frame_id = BASE_FRAME
          - call_async + spin_until_future_complete
          - filtruj 6 stawow PO NAZWIE (odpowiedz ma ~28)
        """
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

    def send(self, joints,duration = TIME_FROM_START):
        if self.pub.get_subscription_count() == 0:
            self.get_logger().warn("kontroler nie subskrybuje ")
            return False
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(q) for q in joints]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration % 1) * 1e9)
        msg.points.append(pt)
        self.pub.publish(msg)
        return True

    # ------------------------------------------------------------ bazowanie --
    def home(self, qpos=None, duration=HOME_TIME, tol=0.02):
        """Dojedz do pozy resetu z robosuite. Zwroc True, gdy sie udalo."""


    
        qpos = list(HOME_QPOS if qpos is None else qpos)
        if self.wait_for_controller is False:
            return False
        self.send(qpos, duration)
        self.spin_for(duration + duration)
        pos, vel = self.read_joints()
        pos = np.asarray(pos, dtype=np.float64)
        
        dif_vector = np.abs(qpos - pos)
        max_dif = np.max(dif_vector)

        if max_dif < tol: 
            return True
        else: 
            print(dif_vector)
            return False
    # ------------------------------------------------------------- twist --
    def send_twist(self, action):
        """Akcja -> twist dla Servo. Zwraca False, gdy nikt nie slucha.

        Nie ma tu ani IK, ani GRIP_OFFSET: Servo steruje ramka ee_frame_name
        i samo liczy jakobian. Znika wiec cala klasa bledow z action_to_target.
        """
        if self.twist_pub.get_subscription_count() == 0:
            self.get_logger().warn("servo nie subskrybuje — czy /servo_node/start_servo?")
            return False
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = BASE_FRAME     # robot_link_command_frame z configu
        a = np.asarray(action, dtype=np.float64)
        # skala translacji Servo kalibrowana przy OSC 0.05 m — proporcjonalnie do --pos-max
        m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = \
            a[0:3] * (TARGET_LIN / SERVO_SCALE_LIN) * (POS_SCALE / POS_SCALE_CAL)
        # skala obrotu Servo kalibrowana przy ROT_SCALE 0.5 — proporcjonalnie do --rot-max
        m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = \
            a[3:6] * (TARGET_ROT / SERVO_SCALE_ROT) * (ROT_SCALE / 0.5)
        self.twist_pub.publish(m)
        return True

    def wait_for_controller(self, timeout=10.0):
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.pub.get_subscription_count() > 0:
                    return True
                rclpy.spin_once(self, timeout_sec=0.1)
            return False
    # -------------------------------------------------------------- polityka --
    def act(self, flat):
        """Zwroc akcje [7] z polityki. deterministic=True jak przy ewaluacji."""
        a, _ = self.model.predict(np.asarray(flat, dtype=np.float32),
                                  deterministic=True)
        return np.asarray(a, dtype=np.float64)

    def action_to_target(self, action, eef_pos, eef_quat):
        """Akcja -> docelowa poza KOLNIERZA w ramce BASE_FRAME.

        Orientacja NIE jest opcjonalna. Strategia tej polityki polega na
        obroceniu nadgarstka: `eef_align` schodzi z 89.9 st. do ok. 24 st.
        w 30 krokow, przy prawie stalym dystansie. Z wyzerowanym a[3:6]
        dystans w robosuite ucieka 0.045 -> 0.331 (docs 6.28).

        Trzy pulapki:
          - robosuite sklada obrot jako R_cel = R_delta @ R_biezace, czyli
            mnozenie Z LEWEJ (input_ref_frame = "base"). Odwrotna kolejnosc
            daje inny wynik.
          - delte skladamy na `eef_quat` (konwencja robosuite), a IK oczekuje
            konwencji TF Webotsa — stad powrot przez odwrotnosc QUAT_SITE_REL.
          - GRIP_OFFSET idzie wzdluz osi Z pozy DOCELOWEJ, nie biezacej.
            read_tcp() zwraca punkt chwytu, /compute_ik oczekuje kolnierza.
        """
        q_delta = axisangle2quat(np.asarray(action[3:6], dtype=np.float64) * ROT_SCALE)
        q_goal = quat_mul(q_delta, eef_quat)                      # R_delta @ R_biezace
        quat_target = quat_mul(q_goal, quat_conj(QUAT_SITE_REL))  # -> konwencja TF

        z_axis = quat2mat(quat_target)[:, 2]
        target_pos = eef_pos + action[0:3] * POS_SCALE - GRIP_OFFSET * z_axis
        return target_pos, quat_target

    # -------------------------------------------------------------- nadzorca --
    def pick_target(self):
        """Ktory przycisk jest celem.

        TODO 4 (na B2, w B1 zostaw 0): w env robil to `random_order` + licznik
        wcisnietych. Tutaj decyduje nadzorca. Na start: staly 0.
        Docelowo: przelacz na nastepny, gdy wykryjesz wcisniecie — a wcisniecia
        w Webots NIE DA SIE wykryc, bo przyciski nie maja fizyki (docs 6.24).
        """
        return 0


def run_axis_test(node, args):
    """Znak i kierunek realizacji akcji, os po osi."""
    osie = [("+X", 0, +1.0), ("-X", 0, -1.0), ("+Y", 1, +1.0),
            ("-Y", 1, -1.0), ("+Z", 2, +1.0), ("-Z", 2, -1.0),
            ("+RX", 3, +1.0), ("-RX", 3, -1.0), ("+RY", 4, +1.0),
            ("-RY", 4, -1.0), ("+RZ", 5, +1.0), ("-RZ", 5, -1.0)]
    print(f"\n{'os':>5} {'przesuniecie [mm]':>28} {'dlugosc':>9} {'mm/krok':>9} "
          f"{'obrot [st]':>11} {'st/krok':>9} {'os obrotu':>22} {'ZGODNOSC':>9}")
    for nazwa, idx, znak in osie:
        if not node.home():
            print(f"{nazwa:>4}  bazowanie nieudane — pomijam"); continue
        node.spin_for(0.3)
        if node.build(target_idx=0) is None:   # build() odswieza eef_pos/eef_quat z TF
            print(f"{nazwa:>4}  build() zwrocil None — pomijam"); continue
        p0 = np.asarray(node.eef_pos, dtype=np.float64).copy()
        R0 = quat2mat(np.asarray(node.eef_quat, dtype=np.float64))
        a = np.zeros(7)   # 6 osi OSC + chwytak; send_twist i action_to_target czytaja a[0:6]
        a[idx] = znak
        for _ in range(args.axis_steps):
            if node.build(target_idx=0) is None:
                break
            if args.servo:
                node.send_twist(a)
            else:
                tgt_pos, tgt_quat = node.action_to_target(a, node.eef_pos, node.eef_quat)
                joints, _ = node.solve_ik(tgt_pos, tgt_quat)
                if joints is not None:
                    node.send(joints, args.ik_horizon or TIME_FROM_START)
            node.spin_for(1.0 / args.rate)
        node.build(target_idx=0)
        p1 = np.asarray(node.eef_pos, dtype=np.float64).copy()
        R1 = quat2mat(np.asarray(node.eef_quat, dtype=np.float64))
        d = p1 - p0
        # Kat GEODEZYJNY pelnej orientacji. Przechylenie osi Z zaniza obrot
        # wokol osi narzedzia i raz juz dalo falszywe 1.01x (docs 6.28).
        dR = R0.T @ R1
        ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
        # os obrotu z macierzy wzglednej, wyrazona w ramce bazy
        v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        nv = np.linalg.norm(v)
        osobr = (R0 @ (v / nv)) if nv > 1e-9 else np.zeros(3)
        dom = int(np.argmax(np.abs(d)))
        if idx < 3:
            zgod = "TAK" if (dom == idx and np.sign(d[idx]) == znak) else "NIE"
        else:
            k = idx - 3
            zgod = ("TAK" if (nv > 1e-9 and int(np.argmax(np.abs(osobr))) == k
                    and np.sign(osobr[k]) == znak) else "NIE")
        print(f"{nazwa:>5} {np.round(d*1000, 1)!s:>28} {np.linalg.norm(d)*1000:8.1f} "
              f"{np.linalg.norm(d)*1000/args.axis_steps:8.2f} {ang:10.1f} "
              f"{ang/args.axis_steps:8.2f} {np.round(osobr, 2)!s:>22} {zgod:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="B1: nie wysylaj akcji, tylko wypisuj")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--servo", action="store_true",
                    help="sterowanie przez MoveIt Servo (twist) zamiast IK + trajektoria")
    ap.add_argument("--home", action="store_true",
                    help="dojedz do pozy resetu z robosuite przed pomiarem")
    ap.add_argument("--rate", type=float, default=RATE_HZ)
    ap.add_argument("--pos-max", type=float, default=0.05,
                    help="osc_pos_output_max, z ktorym trenowano model [m/krok]; POS_SCALE "
                         "= 0.065 * pos_max / 0.05 (kalibracja przy 0.05; docs 6.36)")
    ap.add_argument("--log-csv", type=str, default=None, metavar="PLIK.csv",
                    help="zrzut na krok: pozycja chwytaka w gen3_base_link, dystans, "
                         "align, dp, dR, czy IK dalo rozwiazanie. Potrzebne do "
                         "atrybucji krokow stojacych (kontroler czy kontakt z korpusem)")
    ap.add_argument("--axis-test", action="store_true",
                    help="zamiast polityki: na kazdej osi z osobna zadaj STALA akcje "
                         "+-1 przez --axis-steps krokow i zmierz, DOKAD ramie pojechalo. "
                         "Sprawdza ZNAK i kierunek realizacji akcji — czego B1 nie "
                         "testuje (B1 porownuje sam wektor akcji, nie jego interpretacje)")
    ap.add_argument("--axis-steps", type=int, default=20)
    ap.add_argument("--ik-horizon", type=float, default=None, metavar="S",
                    help="time_from_start punktu trajektorii [s] (TIME_FROM_START). "
                         "Gdy jest KROTSZY od okresu sterowania (0.05 s przy 20 Hz), "
                         "kontroler nie zdaza i jedzie z maksymalna predkoscia az do "
                         "wywlaszczenia nastepna komenda — zrealizowany ruch przestaje "
                         "zalezec od amplitudy (zmierzone 2026-10-01)")
    ap.add_argument("--exec-pos-scale", type=float, default=None, metavar="M",
                    help="KALIBRACJA MECHANIZMU (nie skala treningu): metry na jednostke "
                         "akcji dla drogi IK, czyli POS_SCALE. Kryterium: zrealizowany "
                         "ruch na krok ma sie zgadzac z robosuite (7.92 mm/krok). "
                         "Nie mylic z --pos-max, ktore mowi, czym trenowano model")
    ap.add_argument("--exec-rot-scale", type=float, default=None, metavar="RAD",
                    help="KALIBRACJA MECHANIZMU: radiany na jednostke akcji dla drogi IK "
                         "(ROT_SCALE). Kryterium: 2.35 st/krok jak w robosuite")
    ap.add_argument("--servo-substeps", type=int, default=1, metavar="N",
                    help="ile razy powtorzyc ten sam twist w jednym kroku polityki, "
                         "co publish_period Servo (0.01 s). Servo liczy przyrost pozycji "
                         "jako v*publish_period, wiec przy 20 Hz jeden twist pokrywa "
                         "10 z 50 ms — N=5 pokrywa caly krok sterowania")
    ap.add_argument("--servo-vel-lin", type=float, default=None, metavar="M/S",
                    help="predkosc liniowa przy |a|=1 [m/s] (nadpisuje TARGET_LIN). "
                         "Tempo robosuite to 0.158 m/s, limit twist Gen3 0.5")
    ap.add_argument("--servo-vel-rot", type=float, default=None, metavar="RAD/S",
                    help="predkosc katowa przy |a|=1 [rad/s] (nadpisuje TARGET_ROT). "
                         "Tempo robosuite to 0.82 rad/s, limit admitancji Gen3 0.8727")
    ap.add_argument("--dump-obs", type=str, default=None, metavar="PLIK.npy",
                    help="zapisz wektor obserwacji z pierwszego kroku (walidacja B1 "
                         "wobec wzorca robosuite — porownanie blok po bloku)")
    ap.add_argument("--rot-max", type=float, default=0.5,
                    help="osc_rot_output_max, z ktorym trenowano model [rad/krok]; "
                         "0.5 = modele do 14.09, 0.1 = modele pod limit 0.8727 (docs 6.35)")
    args = ap.parse_args()
    global ROT_SCALE, POS_SCALE, TARGET_LIN, TARGET_ROT
    ROT_SCALE = float(args.rot_max)
    POS_SCALE = POS_SCALE_CAL * float(args.pos_max) / 0.05
    # TARGET_LIN/ROT to po skroceniu SERVO_SCALE_* wprost predkosc przy |a|=1
    # (unitless * scale = TARGET), wiec knob jest w m/s i rad/s.
    if args.servo_vel_lin is not None:
        TARGET_LIN = float(args.servo_vel_lin)
    if args.servo_vel_rot is not None:
        TARGET_ROT = float(args.servo_vel_rot)
    # Kalibracja mechanizmu IK nadpisuje wartosci wyliczone z --pos-max/--rot-max.
    # Te dwa knoby znacza co innego: --pos-max to skala, ktora trenowano polityke,
    # --exec-pos-scale to ile metrow na jednostke akcji trzeba ZADAC, zeby Webots
    # zrealizowal tyle samo co robosuite (docs 6.28, 6.36).
    if args.exec_pos_scale is not None:
        POS_SCALE = float(args.exec_pos_scale)
    if args.exec_rot_scale is not None:
        ROT_SCALE = float(args.exec_rot_scale)
    if args.servo:
        print(f"[servo] |a|=1 -> {TARGET_LIN:.3f} m/s, {TARGET_ROT:.3f} rad/s, "
              f"podkrokow {args.servo_substeps}")
    else:
        print(f"[ik] POS_SCALE {POS_SCALE:.4f} m/jedn., ROT_SCALE {ROT_SCALE:.4f} rad/jedn., "
              f"horyzont {args.ik_horizon or TIME_FROM_START:.3f} s, okres {1.0/args.rate:.3f} s")

    rclpy.init()
    node = PolicyBridge(args.model, dry_run=args.dry_run)

    if not node.wait_ready(timeout=15.0):
        node.get_logger().error("brak /joint_states albo TF — czy dziala symulacja?")
        return
    if not args.dry_run and not args.servo:
        if not node.ik.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("/compute_ik niedostepne"); return
        if not node.wait_for_controller():
            node.get_logger().error("kontroler nie subskrybuje"); return

    # Bazowanie MUSI byc po wait_ready() (potrzebne joint_states), a przed
    # petla. Bez niego polityka dostaje dystans ~0.5 m, a w treningu nie
    # widziala nigdy wiecej niz 0.296 m — dziala wtedy w ekstrapolacji.
    if args.home:
        if not node.home():
            node.get_logger().error("bazowanie nieudane"); return

    if args.axis_test:
        run_axis_test(node, args)
        node.destroy_node(); rclpy.shutdown(); return

    period = 1.0 / args.rate
    print(f"\n{'krok':>4} {'|a| sr.':>8} {'% nasyc.':>9} "
          f"{'a[0:3] (pozycja)':>26} {'dystans':>8} {'kierunek':>9} {'align':>8} "
          f"{'dp [mm]':>8} {'dR [st]':>8}")

    csv_f = csv_w = None
    if args.log_csv:
        import csv as _csv
        csv_f = open(args.log_csv, "w", newline="")
        csv_w = _csv.writer(csv_f)
        csv_w.writerow(["krok", "eef_x", "eef_y", "eef_z", "dystans", "align_st",
                        "dp_mm", "dR_st", "ik_ok", "abs_a", "nasycenie"])

    sat_hist, nan_steps = [], 0
    # ZREALIZOWANY ruch na krok — ta sama miara co wzorzec robosuite
    # (logs/board_6dof/eval_2026-10-01/b2_robosuite_trace.py): przyrost pozycji
    # EEF i kat geodezyjny orientacji. Bez tego nie da sie powiedziec, czy most
    # nie nadaza, czy polityka inaczej decyduje.
    dp_hist, dR_hist = [], []
    p_prev = R_prev = None
    # Atrybucja krokow stojacych: osobno "IK nie dalo rozwiazania" i "IK dalo,
    # a ramie i tak nie ruszylo" (= nie zdazyl kontroler). Bez tego podzialu
    # jedna trzecia krokow bez ruchu w biegu F nie ma wytlumaczenia (docs 6.41).
    ik_codes = {}
    ik_ok_flags = []         # czy IK dalo rozwiazanie w kroku k
    dp_by_step = []          # ruch zmierzony NA POCZATKU kroku k (skutek komendy k-1)
    req_hist = []            # zadana delta kartezjanska [mm] na krok
    for k in range(args.steps):
        t0 = time.time()
        target_idx = node.pick_target()
        flat = node.build(target_idx=target_idx)
        if flat is None:
            node.get_logger().warn("build() zwrocil None"); break

        if not np.all(np.isfinite(flat)):
            nan_steps += 1
            node.get_logger().error(f"krok {k}: NaN/inf w obserwacji!")
            break

        if args.dump_obs and k == 0:
            np.save(args.dump_obs, flat)
            print(f"[dump] obserwacja kroku 0 -> {args.dump_obs}")

        a = node.act(flat)
        rel = flat[47:50]          # target_relative_pos
        dist = float(flat[50])
        sat = float(np.mean(np.abs(a) > 0.99))
        sat_hist.append(sat)

        # Kontrola kierunku: czy polityka ciagnie W STRONE celu?
        # dodatni iloczyn skalarny = zgodnie, ujemny = w przeciwna strone
        agree = float(np.dot(a[0:3] / (np.linalg.norm(a[0:3]) + 1e-9),
                             rel / (np.linalg.norm(rel) + 1e-9)))
        # align (obs[59]) jest teraz glownym sygnalem: polityka realizuje
        # zadanie OBROTEM nadgarstka, 89.9 st. -> ok. 24 st. (docs 6.28)
        align = np.degrees(float(flat[59]))

        p_now = np.asarray(node.eef_pos, dtype=np.float64).copy()
        R_now = quat2mat(np.asarray(node.eef_quat, dtype=np.float64))
        if p_prev is None:
            dp = dR = float("nan")
        else:
            dp = float(np.linalg.norm(p_now - p_prev)) * 1000.0
            cosang = (np.trace(R_prev.T @ R_now) - 1.0) / 2.0
            dR = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            dp_hist.append(dp); dR_hist.append(dR)
        dp_by_step.append(dp)
        p_prev, R_prev = p_now, R_now

        if csv_w is not None:
            csv_w.writerow([k, f"{p_now[0]:.5f}", f"{p_now[1]:.5f}", f"{p_now[2]:.5f}",
                            f"{dist:.5f}", f"{align:.2f}",
                            ("" if dp != dp else f"{dp:.3f}"),
                            ("" if dR != dR else f"{dR:.3f}"),
                            "", f"{np.abs(a).mean():.4f}", f"{sat:.3f}"])

        print(f"{k:4d} {np.abs(a).mean():8.3f} {100*sat:8.0f}% "
              f"{np.round(a[0:3],3)!s:>26} {dist:8.3f} {agree:+9.2f} {align:7.1f}st "
              f"{dp:8.2f} {dR:8.2f}")

        if not node.dry_run:
            if args.servo:
                # Servo integruje twist przez publish_period (0.01 s), a nie przez
                # krok polityki (0.05 s). Jeden twist na krok pokrywa wiec 1/5 czasu
                # i ramie stoi przez pozostale 40 ms — powtarzamy komende.
                for _ in range(max(1, args.servo_substeps)):
                    node.send_twist(a)
                    if args.servo_substeps > 1:
                        node.spin_for(SERVO_PUBLISH_PERIOD)
            else:
                tgt_pos, tgt_quat = node.action_to_target(a, node.eef_pos, node.eef_quat)
                joints, code = node.solve_ik(tgt_pos, tgt_quat)
                ik_codes[code] = ik_codes.get(code, 0) + 1
                # ZADANA delta: roznica wobec celu dla akcji ZEROWEJ, nie wobec
                # biezacej pozycji. action_to_target zwraca poze KOLNIERZA, a
                # node.eef_pos to punkt chwytu — roznica tych dwoch to po prostu
                # GRIP_OFFSET (0.1448 m) i mierzenie jej nie mowi nic o komendzie.
                noop_pos, _ = node.action_to_target(
                    np.zeros(6), node.eef_pos, node.eef_quat)
                req_hist.append(float(np.linalg.norm(
                    np.asarray(tgt_pos, dtype=np.float64)
                    - np.asarray(noop_pos, dtype=np.float64))) * 1000.0)
                ik_ok_flags.append(joints is not None)
                if joints is not None:
                    node.send(joints, args.ik_horizon or TIME_FROM_START)

        node.spin_for(max(0.0, period - (time.time() - t0)))

    if csv_f is not None:
        csv_f.close()
        print(f"  CSV zapisany: {args.log_csv}")

    if sat_hist:
        print(f"\n--- PODSUMOWANIE ---")
        print(f"  srednie nasycenie akcji : {100*np.mean(sat_hist):.0f}%")
        print(f"  (robosuite dla tego modelu: ~71 %. Duzo wiecej = obserwacja")
        print(f"   poza rozkladem, cos w wektorze sie nie zgadza)")
        print(f"  krokow z NaN            : {nan_steps}")
        if ik_codes:
            # MoveIt MoveItErrorCodes: 1 SUCCESS, -31 NO_IK_SOLUTION,
            # -10 START_STATE_IN_COLLISION, -12 GOAL_IN_COLLISION, None = brak odpowiedzi
            nazwy = {1: "SUCCESS", -31: "NO_IK_SOLUTION", -10: "START_IN_COLLISION",
                     -12: "GOAL_IN_COLLISION", None: "brak odpowiedzi/zly joint_state"}
            tot = sum(ik_codes.values())
            print(f"  compute_ik: {tot} wywolan")
            for c, n in sorted(ik_codes.items(), key=lambda kv: -kv[1]):
                print(f"    {str(c):>5} {nazwy.get(c, 'inny kod'):<32} {n:4d} ({100*n/tot:.0f}%)")
            # komenda z kroku k daje ruch zmierzony na poczatku kroku k+1
            par = [(ok, dp_by_step[k+1]) for k, ok in enumerate(ik_ok_flags)
                   if k + 1 < len(dp_by_step) and dp_by_step[k+1] == dp_by_step[k+1]]
            n_ok = sum(1 for ok, _ in par if ok)
            n_ok_still = sum(1 for ok, d in par if ok and d < 0.1)
            n_bad = sum(1 for ok, _ in par if not ok)
            if n_ok:
                print(f"  IK OK, a ramie STALO (dp<0.1mm): {n_ok_still}/{n_ok} "
                      f"({100*n_ok_still/n_ok:.0f}%)")
            print(f"  krokow bez komendy (IK padlo): {n_bad}/{len(par)} "
                  f"({100*n_bad/max(len(par),1):.0f}%)")
            if req_hist:
                print(f"  mediana ZADANEJ delty    : {np.median(req_hist):.2f} mm/krok")
        if dp_hist:
            print(f"  mediana dp              : {np.median(dp_hist):.2f} mm/krok "
                  f"(robosuite 7.92)")
            print(f"  mediana dR              : {np.median(dR_hist):.2f} st/krok "
                  f"(robosuite 2.35)")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
