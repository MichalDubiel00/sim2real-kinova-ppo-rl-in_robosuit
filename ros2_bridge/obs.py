#!/usr/bin/env python3
"""
B0: zbuduj wektor obserwacji [120] z ROS i WYPISZ GO. Bez polityki, bez ruchu.

Kontrakt: envs/obs_contract.json (regeneracja: training/diag_obs_map.py --json)

UKLAD (Kinova3_6DOF, 120 = 60 unikalnych x2):
   0:6   joint_pos              /joint_states, filtr joint_1..6
   6:12  joint_pos_cos          cos(joint_pos)
  12:18  joint_pos_sin          sin(joint_pos)
  18:24  joint_vel              /joint_states .velocity
  24:27  eef_pos                TF gen3_base_link -> end_effector_link
  27:31  eef_quat               ten sam TF, xyzw
  31:35  eef_quat_site          quat * [0,0,-0.7071,0.7071]  (patrz nizej)
  35:41  gripper_qpos           Robotiq85 != 2f_140 — patrz TODO 4
  41:47  gripper_qvel           j.w.
  47:50  target_relative_pos    poza gniazda - eef_pos   [uklad SWIATA]
  50:51  target_distance        norma powyzszego
  51:56  target_onehot          ktory przycisk (decyzja nadzorcy)
  56:59  target_press_axis      -normalna scianki
  59:60  eef_align_with_target  arccos(|dot(os_Z_chwytaka, press_axis)|)
  60:120 DUPLIKAT 0:60          agregaty robot0_proprio-state + object-state

REFERENCJA robosuite (poza domowa, Kinova3_6DOF, chwytak zamkniety):
  joint_pos      [ 0.      0.65   -1.89    0.     -0.6    -1.5708]
  eef_pos        [0.3265 0.0014 0.6559]
  eef_quat       [-1.      0.     -0.0008 -0.    ]
  eef_quat_site  [0.7071 0.7071 0.0006 0.0006]

URUCHOMIENIE (conda robosuite — tam jest torch/SB3, a rclpy importuje sie
poprawnie, bo to tez Python 3.10):
  source /opt/ros/humble/setup.bash
  source ~/Batchelor/webots_ros2_spot/install/setup.bash
  ~/miniconda3/envs/robosuite/bin/python ros2_bridge/obs.py
"""

import sys
sys.modules["tensorflow"] = None   # PRZED SB3 — inaczej segfault (docs 7.6)

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

BASE_FRAME = "gen3_base_link"
EE_FRAME = "end_effector_link"
JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
N_BUTTONS = 5

# Staly obrot -90 st. wokol lokalnej osi Z (kwaternion xyzw).
# Wystepuje DWA RAZY w lancuchu przeliczen (docs 6.26):
#
#   ramka `end_effector_link` w Webots jest obrocona o 90 st. wokol osi
#   narzedzia wzgledem `eef_quat` robosuite:
#       eef_quat      = quat_Webots * REL
#       eef_quat_site = quat_Webots * REL * REL
#
#   Zweryfikowane: zgodnosc 0.02 st. dla eef_quat i quat_site jednoczesnie.
#   Osie Z wszystkich trzech ramek POKRYWAJA SIE — dlatego GRIP_OFFSET liczony
#   wzdluz Z dziala niezaleznie od tej roznicy.
QUAT_SITE_REL = np.array([0.0, 0.0, -0.70710678, 0.70710678])   # xyzw

# Punkt chwytu robosuite (Robotiq85) lezy 0.1448 m od kolnierza wzdluz osi
# narzedzia. Zmierzone testem kinematyki prostej: te same katy w obu
# symulatorach, roznica [0.1444, 0.0004, -0.0102] = 0.1448 m wzdluz lokalnej Z.
# Webots ma 2f_140 (wkladki na 0.167 m), ale LABORATORIUM MA 85 — wiec ta
# wartosc odtwarza i trening, i realny robot. Patrz docs 6.26.
GRIP_OFFSET = 0.1448

# Poza bazy ramienia w ukladzie SWIATA treningowego (ButtonBoardEnv).
# obs[24:27] `eef_pos` jest w robosuite BEZWZGLEDNE, nie wzgledem bazy — a to
# wlasnie bezwzgledna poza EEF zabila transfer 7DOF przy przeniesieniu bazy
# (docs 6.22). `target_relative_pos` jest odporny (roznica wektorow), ale
# `eef_pos` nie.
BASE_IN_WORLD = np.array([-0.10, 0.0, 0.62])   # base_x_offset, 0, KINOVA_BASE_Z


def _sign_w_nonneg(q):
    """Konwencja robosuite `mat2quat`: wymusza w >= 0 (zweryfikowane 200/200)."""
    return -q if q[3] < 0 else q


def _sign_max_negative(q):
    """Konwencja surowego MuJoCo `body_xquat` dla tego robota: najwieksza co do
    modulu skladowa jest UJEMNA (zweryfikowane 200/200 poz).

    q i -q opisuja TEN SAM obrot, ale siec widzi LICZBY. Pomiar: odwrocenie
    znaku zmienia akcje polityki srednio o 0.50 na zakresie [-1,1], w 100 %
    krokow (docs 6.26). Znaku NIE wolno zostawic przypadkowi."""
    return -q if q[np.argmax(np.abs(q))] > 0 else q

TRAINING_POS = np.array([[0.4670,  0.0000,  0.0200],
                        [0.5225, -0.1022,  0.1222],   
                        [0.5225,  0.1022,  0.1222],   
                        [0.5225,  0.1022, -0.0822],   
                        [0.5225, -0.1022, -0.0822]])
TRAINING_NORMAL_POS = np.array([[-1.0000,  0.0000,  0.0000],
                            [-0.7071, -0.5000,  0.5000],
                            [-0.7071,  0.5000,  0.5000],
                            [-0.7071,  0.5000, -0.5000],
                            [-0.7071, -0.5000, -0.5000] ])

# Chwytak: robosuite ma Robotiq85, Webots 2f_140 — wartosci NIEporownywalne.
# Przy force_gripper_closed=True chwytak byl zamkniety przez caly trening,
# wiec mediana z robosuite jest uczciwym zastepnikiem.
#
# Mediana z 6 epizodow finalnego modelu (2977 krokow) przy DOMYSLNYM szumie
# srodowiska, czyli w warunkach treningu.
#
# UWAGA: poprzednie wartosci qvel [-2.05, -22.43, -1.39, -0.056, -10.07, -0.073]
# byly ODCHYLENIEM STANDARDOWYM ze znakiem minus, nie mediana. Chwytak drga
# wokol zera: std = [2.48, 24.99, 1.68, 0.14, 11.36, 0.14], mediana ~ 0.
# Blad podnosil |a| z 0.786 na 0.450 i zbijal nasycenie z 71% na 0%.
# Polityka reaguje na ten blok mocniej, niz sugeruje jego 12 wymiarow.
GRIPPER_QPOS_MED = np.array([0.4301, -0.2016, 0.2189, 0.6243, 0.1544, 0.2708])
GRIPPER_QVEL_MED = np.array([-0.0009, -0.0103, 0.0017, 0.0011, -0.0050, 0.0001])



def quat_mul(a, b):
    """Zlozenie kwaternionow xyzw (a potem b)."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_conj(q):
    """Sprzezenie kwaternionu xyzw. Dla jednostkowego = obrot odwrotny."""
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def axisangle2quat(v):
    """Wektor os-kat (dlugosc = kat w rad) -> kwaternion xyzw.

    Taka reprezentacje ma a[3:6] polityki po przeskalowaniu przez ROT_SCALE.
    Przy dlugosci ~0 zwraca identycznosc — bez tego dzielenie przez zero
    przy nieruchomym nadgarstku.
    """
    v = np.asarray(v, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = v / angle
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def quat2mat(q):
    """Kwaternion xyzw -> macierz 3x3. Kolumna 2 = lokalna os Z."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class State(Node):
    def __init__(self):
        super().__init__("obs_builder")
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, self)
        self.js = None            # ostatni JointState
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.joint_names = JOINTS
        self.current_joint_positions = [None] * len(self.joint_names)
        self.current_joint_velocities = [None] * len(self.joint_names)
        self.joint_state_received = False
        self.eef_pos, self.quat_webots, self.eef_quat = None, None, None

    def _js_cb(self, msg):
        try:
            # Only update for joints in self.joint_names (ignore gripper/fingers)
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    joint_idx = msg.name.index(joint_name)
                    self.current_joint_positions[i] = msg.position[joint_idx]
                    if len(msg.velocity) > joint_idx:
                        self.current_joint_velocities[i] = msg.velocity[joint_idx]
            if all([ v is not None for v in self.current_joint_positions]) :
                self.joint_state_received = True
        except Exception as e:
            self.get_logger().warn(f"Error updating joint state: {e}")


    def spin_for(self, sec):
        """NIE time.sleep() — ten nie przetwarza wiadomosci (docs 6.26)."""
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_ready(self, timeout=10.0):
        """Czekaj az przyjda i joint_states, i TF."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joint_state_received is False:
                continue
            try:
                self.tfb.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time())
                return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    def read_joints(self):        
        return self.current_joint_positions,self.current_joint_velocities

    def read_tcp(self, timeout=5.0):
        """Zwroc (pos[3], quat_xyzw[4]) PUNKTU CHWYTU w ukladzie BASE_FRAME.

        UWAGA — ramki EEF znacza co innego w obu symulatorach (docs 6.26):
          robosuite `gripper0_right_grip_site`  punkt CHWYTU miedzy palcami
          Webots    `end_effector_link`         KOLNIERZ narzedzia
        Roznica 0.1448 m wzdluz osi narzedzia. Wstawienie kolnierza wprost
        przesunieloby caly `target_relative_pos` o 14 cm, przy
        `percep_pos_bias = 0.015`.

        Kwaternion zwracany SUROWY (ramka Webotsa) — przeliczenie na konwencje
        robosuite robi build(), bo tam widac caly lancuch. Obrot kolnierza
        i punktu chwytu jest ten sam, rozni sie tylko poczatek ukladu.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.02)
            try:
                tr = self.tfb.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time())
            except Exception:
                continue
            t, r = tr.transform.translation, tr.transform.rotation
            flange = np.array([t.x, t.y, t.z])
            quat = np.array([r.x, r.y, r.z, r.w])
            # punkt chwytu = kolnierz + GRIP_OFFSET wzdluz LOKALNEJ osi Z
            z_axis = quat2mat(quat)[:, 2]
            return flange + GRIP_OFFSET * z_axis, quat
        return None, None

    # ------------------------------------------------------------------
    def button_pose(self, idx):
        """Zwroc (pozycja_gniazda[3] w ukladzie BASE_FRAME, normalna[3]).

            3: na razie PRAWDA, nie ArUco — o to chodzi w warstwie B.
          Gniazda `estop` w swiecie Webotsa (docs 6.24), uklad swiata:
            btn0 (-4.550, -3.885, 0.639)   srodkowy, wysuniety ~0.034 do przodu
            btn1 (-4.589, -3.810, 0.714)
            btn2 (-4.578, -3.820, 0.574)
            btn3 (-4.592, -3.963, 0.561)
            btn4 (-4.578, -3.963, 0.716)
            btn        pozycja [m]    geometria treningowa
            [0.4670,  0.0000,  0.0200]   
            [0.5225, -0.1022,  0.1222]   
            [0.5225,  0.1022,  0.1222]   
            [0.5225,  0.1022, -0.0822]   
            [0.5225, -0.1022, -0.0822] 

            [-1.0000,  0.0000,  0.0000]
            [-0.7071, -0.5000,  0.5000]
            [-0.7071,  0.5000,  0.5000]
            [-0.7071,  0.5000, -0.5000]
            [-0.7071, -0.5000, -0.5000]  
          Normalne scianek: btn0 czolowa, btn1..4 skosne 45 st. (docs 6.24).

          UWAGA: to sa wspolrzedne SWIATA Webotsa, a obs oczekuje ukladu
          zgodnego z trenowanym. Musisz je przeliczyc do BASE_FRAME — albo
          przez TF (jesli tablica ma ramke), albo recznie znajac poze Spota.
          NAJPIERW ustal, w jakim ukladzie ma byc target_relative_pos: w env
          jest to (site_xpos - eef_pos), oba w ukladzie SWIATA MuJoCo, gdzie
          baza ramienia stoi w (base_x_offset, 0, KINOVA_BASE_Z).
        """
        pos,pos_normal = TRAINING_POS,TRAINING_NORMAL_POS
        return pos[idx],pos_normal[idx]

    # ------------------------------------------------------------------
    def build(self, target_idx=0):
        jp, jv = self.read_joints()
        self.eef_pos, self.quat_webots = self.read_tcp()

        # UWAGA: kwaternion z TF to ramka WEBOTSA, obrocona o 90 st. wokol osi
        # narzedzia wzgledem robosuite. Trzeba go przeliczyc — patrz QUAT_SITE_REL.
        eef_quat = _sign_max_negative(quat_mul(self.quat_webots, QUAT_SITE_REL))
        self.eef_quat = eef_quat   # potrzebny w policy_bridge do zlozenia delty obrotu
        eef_quat_site = _sign_w_nonneg(quat_mul(eef_quat, QUAT_SITE_REL))

        # obs[24:27] jest BEZWZGLEDNE — przesun z ramki bazy do swiata treningowego.
        # eef_pos (wzgledny) zostaje do policzenia target_relative_pos, bo tam
        # roznica wektorow i tak skraca poczatek ukladu.
        eef_pos_world = BASE_IN_WORLD + self.eef_pos

        
        #   (a) stale z robosuite (GRIPPER_QPOS_MED / QVEL_MED) — proste,
        #       broni sie tym, ze chwytak byl zamkniety przez caly trening
        gripper_qpos = GRIPPER_QPOS_MED
        gripper_qvel = GRIPPER_QVEL_MED

        btn_pos, btn_normal = self.button_pose(target_idx)

        #   target_relative_pos = btn_pos - eef_pos
        #   target_distance     = norma powyzszego (shape (1,)!)
        #   target_onehot       = wektor 5, jedynka na target_idx, float32
        #   target_press_axis   = -btn_normal, ZNORMALIZOWANY
        #   eef_align           = arccos(|dot(eef_z, press_axis)|), shape (1,)
        #     gdzie eef_z = quat2mat(eef_quat)[:, 2]
        #   Uwaga: env liczy align na eef_quat, NIE na quat_site.
        # Wektor EEF->cel. W env jest tu jeszcze szum percepcji (bias per epizod
        # + szum per krok) — w warstwie B go NIE dodajemy, bo mamy prawde.
        # Uwaga: eef_pos MUSI byc ten sam, ktory idzie do obs[24:27], inaczej
        # obserwacja jest wewnetrznie sprzeczna.
        target_relative_pos = btn_pos - self.eef_pos
        target_distance = np.array([np.linalg.norm(target_relative_pos)])
        target_onehot = np.zeros(N_BUTTONS, dtype=np.float32)
        target_onehot[target_idx] = 1.0

        axis = -np.asarray(btn_normal, dtype=np.float64)
        target_press_axis = axis / np.linalg.norm(axis)

        # Kat miedzy osia Z chwytaka a osia wciskania. abs() w srodku daje
        # zakres 0..pi/2 zamiast 0..pi — chwytak "tylem" liczy sie jak
        # wyrownany. Polityka uczyla sie na wersji z abs().
        # clip chroni przed arccos(1.0000001) = NaN.
        eef_z = quat2mat(eef_quat)[:, 2]
        cos_a = np.clip(float(np.dot(eef_z, target_press_axis)), -1.0, 1.0)
        eef_align = np.array([np.arccos(abs(cos_a))])

        #   unique = np.concatenate([...])          # 60
        #   flat   = np.concatenate([unique, unique])  # 120
        #   Duplikat NIE jest bledem — robosuite publikuje kazda wartosc dwa
        #   razy (osobne klucze + agregaty *-state). Bez niego model dostanie
        #   wektor dlugosci 60 i odmowi; z blednym — przyjmie i zacznie bredzic.
        jp_a,jv_a = np.asarray(jp,dtype=np.float64),np.asarray(jv,dtype=np.float64)
        jp_cos = np.cos(jp_a)
        jp_sin = np.sin(jp_a)
        unique = np.concatenate([       
                jp_a,                 # 0:6   joint_pos              /joint_states, filtr joint_1..6            
                jp_cos,               # 6:12  joint_pos_cos          cos(joint_pos)
                jp_sin,               #12:18  joint_pos_sin          sin(joint_pos)    
                jv_a,                 #18:24  joint_vel              /joint_states .velocity    
                eef_pos_world,        #24:27  eef_pos                TF gen3_base_link -> end_effector_link
                eef_quat,             #27:31  eef_quat               ten sam TF, xyzw        
                eef_quat_site,        #31:35  eef_quat_site          quat * [0,0,-0.7071,0.7071]  (patrz nizej)    
                gripper_qpos,         #35:41  gripper_qpos           Robotiq85 != 2f_140        
                gripper_qvel,         #41:47  gripper_qvel           j.w.        
                target_relative_pos,  #47:50  target_relative_pos    poza gniazda - eef_pos   [uklad SWIATA]        
                target_distance,      #50:51  target_distance        norma powyzszego                
                target_onehot,        #51:56  target_onehot          ktory przycisk (decyzja nadzorcy)
                target_press_axis,    #56:59  target_press_axis      -normalna scianki             
                eef_align             #59:60  eef_align_with_target  arccos(|dot(os_Z_chwytaka, press_axis)|)          
            ])
        flat = np.concatenate([unique,unique]) #60:120 DUPLIKAT 0:60          agregaty robot0_proprio-state + object-state 
        return flat

def main():
    rclpy.init()
    node = State()
    if not node.wait_ready():
        node.get_logger().error("brak /joint_states albo TF — czy dziala symulacja?")
        return

    flat = node.build(target_idx=0)


    flat_np = np.asarray(flat,dtype=np.float64)
    print(flat_np.size)
    print(np.allclose(flat_np[60:120], flat_np[0:60]))
    print(np.all(np.isfinite(flat_np)))
    print(flat_np[0:6])
    print(flat_np[24:27])


    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
