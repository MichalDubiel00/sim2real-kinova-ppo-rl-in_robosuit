"""
ButtonBoard-v0
==============
Kinova Gen3 (on Spot base) presses 5 Emergency Stop buttons
mounted in a cross (plus) pattern on the Omniboard wall panel.

Button layout (RoboCup omniboard): the cross-shaped block tapers toward
the robot; the centre button sits on the small front face pointing straight
at the robot, and the 4 arm buttons sit on the 45-degree bevelled faces of
the arms, each pointing diagonally outward (up/right/down/left + forward):

        [1]
    [4] [0] [2]      0 = front centre, then top / right / bottom / left
        [3]

Task: press all 5 buttons in sequence (order randomised when
random_order=True — the current target is exposed via target_onehot).

Sim2real observation model: the policy only sees quantities reproducible
on the real robot — target position/axis as estimated by an external
perception module (board pose from e.g. AprilTag), so both carry a
per-episode constant bias + per-step noise. Button depressions are NOT
observed (real board is not wired); press detection on hardware comes
from EEF force + push depth, and the task state machine (which button
is next) is the supervisor's job — here simulated by the env itself.

Press model (reward v2.2/v2.3): a press only counts when a gripper FINGER
geom is in contact with the button cap at the moment of activation —
mirroring the hardware supervisor, which detects presses from
force/kinematics at the TCP and would never register a press made with the
wrist or arm. ARM-link contact with any button is penalised like a board
collision (wrist-side pressing risks the Gen3 wrist camera and breaks the
TCP-based force/depth estimates). Gripper housing/knuckle contact is NOT
penalised (v2.3): it inevitably accompanies legitimate fingertip presses —
penalising it (v2.2) taught the policy to avoid buttons altogether
(0/100 successes). The gripper is also forced closed (action dim stays 7;
the gripper command is overridden) so the policy presses with a stiff,
well-defined fingertip contact.

Contact quality (reward v2.4, FAILED — see v2.5): v2.3 successfully unblocked
pressing but left NOTHING discouraging clumsy contact — buttons were hit at up
to 1 m/s (visible as gripper jaws splaying apart on hard/off-axis impact,
since the compliant Robotiq position controller loses to contact torque).
v2.4 added FINGER_MISUSE_COST for touching a non-target button and
CONTACT_SPEED_COST for fast contact with any button. Fine-tune result:
success dropped 85%->70%->43% with more training, while contact-quality
metrics stayed unchanged. ROOT CAUSE (found 2026-07-21): the target advances
on the same step a press succeeds, so the lingering finger contact with the
just-pressed cap during retreat counted as "wrong button" — measured ~93% of
all "wrong contact" steps were this artefact, meaning both penalties fired
right after every successful press and clawed back the press bonus. The
policy learned that pressing is unprofitable — same collapse mechanism as
v2.2. (The v2.3-era estimate "~25% of contact is bumping wrong buttons"
suffered the same measurement artefact; genuine bumping into unpressed
non-target buttons is ~2% of contact steps.)

Contact quality (reward v2.5, ALSO REJECTED 2026-07-21): both penalties fire
only on contact with a button that is neither the current target nor already
pressed. Lingering post-press contact and the correct final press are
penalty-free; only genuinely bumping a not-yet-pressed wrong button
(optionally fast) is penalized. Result: best 83/100 (below the 90/100 v2.3
baseline), still degrading with training; genuine bumping is only ~2% of
contact steps so the penalty had nothing to work on. Conclusion after three
failures (v2.2/v2.4/v2.5): contact grace belongs in the supervisor's
SafetyLayer (TCP speed limit near the board), not in the reward. The v2.5
penalties remain available as OPT-IN constructor params (finger_misuse_cost,
contact_speed_cost); the DEFAULT configuration is reward v2.3. The finger
press gate can be disabled (finger_press_gate=False) for curriculum stage 1
when training from scratch (e.g. for the 6DOF robot — see
envs/kinova3_6dof_robot.py).

Coordinate frame:
  - Robot base at (0, 0, KINOVA_BASE_Z), facing +X
  - Wall face at x = WALL_X; omniboard big (back) face mounted on it,
    small front face at x = WALL_X - BOARD_DEPTH
  - Each button's spring joint compresses along the inward normal
    of the face it is mounted on
"""

import numpy as np
import mujoco
import xml.etree.ElementTree as ET

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from objects.button_object import ButtonObject
from objects.custom_objects import OmniBoardObject

_ROBOSUITE_PKG_PATH = os.path.split(robosuite.__file__)[0]

# Uklad przyciskow na NIEOBROCONEJ tablicy: (offset srodka scianki wzgledem
# srodka tylnej sciany, normalna scianki na zewnatrz) w ukladzie swiata.
# Scianki skosne maja 45 st., ich srodki leza 40 mm (pol glebokosci) przed
# sciana i 107 mm od srodka krzyza.
#
# 2026-09-01: promien 120 -> 107 mm. Zmierzony na modelu MASKOR (webots_ros2_spot,
# galaz kinova_gen3, wezel DEF estop w worlds/spot.wbt), robionym przez zespol
# majacy fizyczna arene z omniboardem. Gniazda leza tam na 0.106 / 0.110 / 0.110
# (czwarte 0.092 odstaje - niedokladnosc modelu, odrzucone). Mediana 0.107.
# Kat ukladu potwierdzony niezaleznie: +45.0 / +135.4 / -45.0 / -135.0 stopnia.
_S = 0.7071067811865476  # sqrt(2)/2
_BUTTON_FACES_FLAT = [
    ((-0.08, 0.00, 0.00), (-1.0, 0.0, 0.0)),    # 0: srodek czolowy
    ((-0.04, 0.00, 0.107), (-_S, 0.0, _S)),      # 1: sciecie gorne
    ((-0.04, 0.107, 0.00), (-_S, _S, 0.0)),      # 2: sciecie prawe
    ((-0.04, 0.00, -0.107), (-_S, 0.0, -_S)),    # 3: sciecie dolne
    ((-0.04, -0.107, 0.00), (-_S, -_S, 0.0)),    # 4: sciecie lewe
]


def _roll_x(vec, ang):
    """Obrot wektora wokol osi X swiata (normalnej tablicy) o ang [rad]."""
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = vec
    return (x, y * c - z * s, y * s + z * c)


def _rolled_button_faces(roll_deg):
    """Uklad przyciskow po obroceniu tablicy o roll_deg wokol jej normalnej.

    Przy 45 st. krzyz (gora/dol/lewo/prawo) przechodzi w uklad przekatniowy
    (gora-lewo / gora-prawo / dol-lewo / dol-prawo) — tak omniboard jest
    montowany w Rescue League. Przycisk 0 lezy w srodku, wiec obrot go nie
    rusza. Ta sama wartosc obrotu jest nakladana na quaternion bryly tablicy
    w _load_model, wiec przyciski zawsze leza na swoich sciankach.
    """
    a = np.radians(roll_deg)
    return [(_roll_x(off, a), _roll_x(nrm, a)) for off, nrm in _BUTTON_FACES_FLAT]


class ButtonBoardEnv(ManipulationEnv):
    """
    5 Emergency Stop buttons in a cross pattern on the Omniboard, press in sequence.

    Args:
        robots:            "Kinova3"
        random_order:      if True, randomise press order each episode
        board_pos_noise:   +/- uniform noise [m] on board centre (Y, Z) per episode
        percep_pos_bias:   +/- uniform per-episode bias [m] on perceived target pos
                           (simulates board pose estimation error, e.g. ArUco)
        percep_pos_noise:  gaussian per-step noise std [m] on perceived target pos
        percep_axis_noise: gaussian per-episode perturbation of the press axis
        base_z_noise:      +/- uniform noise [m] on ARM BASE height per episode

                           2026-09-01, poziomy szumu podniesione pod realna
                           percepcje (ArUco teraz, YOLO pozniej) zamiast pod
                           idealny odczyt pozy z symulatora:
                             board_pos_noise    0.05  -> 0.06
                             percep_pos_bias    0.005 -> 0.015
                             percep_pos_noise   0.002 -> 0.005
                             percep_axis_noise  0.02  -> 0.05
                             base_z_noise       (nowy) 0.03
                           Bias 15 mm to rzad bledu pozy ze znacznika ArUco na
                           kamerze 640x480 z odleglosci roboczej; 5 mm to drzenie
                           miedzy klatkami. percep_axis_noise podniesiony
                           mocniej niz pozycyjny, bo blad ORIENTACJI z ArUco
                           jest znacznie gorszy od pozycyjnego (niejednoznacznosc
                           plaskiego znacznika, przeskoki normalnej). Szum jest
                           tylko w OBSERWACJI - nagroda i detekcja wcisniecia
                           licza sie na prawdzie z symulatora.
        reward_scale:      multiplier on total reward
        force_gripper_closed: if True, override the gripper action to
                           "close" every step (action_dim stays 7)
        finger_press_gate: if True (v2.2+), a press only counts when a
                           gripper finger touches the cap; False = any
                           contact that depresses the button counts
                           (curriculum stage 1 — the gate strangles
                           exploration when training from scratch)
        finger_misuse_cost: penalty per step of finger contact with a
                           non-target, not-yet-pressed button (v2.5;
                           default 0.0 = v2.3 semantics, v2.5 REJECTED)
        contact_speed_cost: penalty coef for EEF speed above
                           CONTACT_SPEED_THRESH during such contact
                           (v2.5; default 0.0 = v2.3 semantics)
        dense_align_coef:  weight of the axis-alignment dense term
                           (v2.3: 0.05; v2.4/v2.5 used 0.15)
        collision_cost:    penalty per step of arm/board contact
                           (v2: 0.5 — negligible; v2.1+: 5.0). For
                           FROM-SCRATCH training use the v2 value: with
                           5.0 a single collision step (-5/225) outweighs
                           the whole approach shaping (~10*0.7/225), so
                           exploration learns to avoid the board and
                           never discovers pressing (6DOF s1 failure,
                           2026-07-30)
        action_rate_cost:  penalty coef on ||a_t - a_{t-1}||^2 (v2.1+:
                           0.02; v2 scratch: 0.0)
        activation_depth:  cap depression [m] that counts as a press
                           (default 0.012 = historical 7DOF setting).
                           12 mm of the 15 mm travel is at the physical
                           margin of the Robotiq85+cap geometry: an
                           aligned centered pinch bottoms out on the
                           base ring at ~0.5 mm, a deliberate aligned
                           single-pad press reaches ~7 mm, the learned
                           oblique fingertip technique ~11 mm (6DOF,
                           2026-07-31). The real button's activation
                           depth is unknown and press detection on
                           hardware is electrical (supervisor) — the
                           sim threshold is a task-design parameter,
                           not ground truth. 6DOF uses 0.010.
        base_x_offset:     shifts the robot base toward the wall [m]
                           (default 0.0 = historical placement, base
                           axis 0.50 m from the wall face). Sim2real
                           knob: where Spot parks relative to the board.
                           The 6DOF needs to stand closer — at the
                           historical standoff its press poses require
                           folding into self-collision (base/pedestal)
                           or joint limits (docs 6.8).
        depress_shaping_k: potential-based shaping on the depression of
                           the CURRENT target: k * (depr_t - depr_{t-1}).
                           Pays for pushing the cap in, pays back on
                           release, zero for holding — same mechanism as
                           the distance shaping, on the press axis.
                           Discovery aid for 6DOF from-scratch training
                           (approach learned by 2.5M steps, dents up to
                           ~5 mm, but a full 12 mm press never sampled —
                           the 6DOF's tighter dexterous workspace makes
                           aligned pushes rare). Default 0.0 = v2.3.
        depress_clip_buffer: REWARD CLIPPING on the depression shaping —
                           depth is capped at activation_depth + buffer
                           before entering the shaping term, so pressing
                           deeper than "activated plus a safety margin"
                           earns nothing. Without it the shaping rewards
                           unbounded overtravel while collision_cost
                           penalises the resulting contact: the two pull
                           against each other and the policy collapses
                           after ~2.5M steps (docs 6.18). No-op when
                           depress_shaping_k = 0. Default 0.001 (1 mm).
    """

    N_BUTTONS = 5
    WALL_X = 0.50           # wall face; omniboard big (back) face sits here
    BOARD_DEPTH = 0.08      # omniboard depth (mesh: 80 mm)
    BOARD_Y = 0.0           # board lateral centre
    # 2026-09-01: 0.60 -> 0.64. Poprzednia wartosc brala sie z regulaminu
    # Rescue League ("Omni may be mounted on the wall at 60 cm or floor"),
    # ale to widelki, nie wymiar areny. Model MASKOR (webots_ros2_spot,
    # kinova_gen3) ma srodek tablicy na 0.639 przy bazie ramienia na 0.619 —
    # tablica lezy 2 cm POWYZEJ bazy, a nie 5 cm ponizej jak dotad. Roznica
    # 7 cm wykraczala poza board_pos_noise, wiec nie byla do nadrobienia
    # randomizacja. Zrodlo: zespol z fizyczna arena, patrz _BUTTON_FACES_FLAT.
    BOARD_Z = 0.64
    # WYSOKOSC BEZWZGLEDNA bazy ramienia nad podloga (Spot ~0.5 m + mocowanie
    # ~0.15 m). UWAGA: set_base_xpos() przyjmuje pozycje SPODU cokolu, wiec
    # w _load_model odejmowane jest robot_model.bottom_offset (RethinkMount:
    # 0.912 m). Wczesniej ta stala byla podawana wprost do set_base_xpos,
    # przez co ramie stalo na 1.562 m zamiast 0.65 m — patrz docs 6.15.
    # 2026-09-01: 0.65 -> 0.62, za pomiarem TF w Webots (base_link na 0.5073
    # wg /Spot/gps, base_link->gen3_base_link = [0.256, 0.000, 0.112]).
    # Wysokosc bazy jest na Spocie STEROWALNA (/Spot/set_height), wiec to
    # nominal, wokol ktorego randomizuje base_z_noise.
    KINOVA_BASE_Z = 0.62

    # Obrot tablicy wokol jej normalnej (osi X swiata). W Rescue League
    # omniboard jest zamontowany "na rogu": przyciski leza na przekatnych
    # (gora-lewo / gora-prawo / dol-lewo / dol-prawo), a nie w pionie i
    # poziomie. 45 st. przeprowadza uklad krzyzowy w przekatniowy.
    BOARD_ROLL_DEG = 45.0
    BUTTON_FACES = _rolled_button_faces(BOARD_ROLL_DEG)

    SPRING_STIFFNESS = 500.0  # N/m, matches button.xml
    ACTIVATION_DEPTH = 0.012  # domyslne activation_depth (7DOF/v2.3);
    # 12 mm z 15 mm skoku jest przy geometrii Robotiq85+grzybek na granicy
    # fizycznej wykonalnosci — patrz parametr activation_depth w __init__
    # Przykladowy profil zatrzaskowego E-stopu (docs 6.33). f_start/f_peak/x_peak wg
    # uzytkownika (15 -> 35 N przy 3-4 mm), x_stroke wg karty katalogowej (5 mm).
    # f_after i snap_width to ZALOZENIA, nie dane.
    ESTOP_PROFILE_EXAMPLE = dict(f_start=15.0, f_peak=35.0, x_peak=0.0035,
                                 f_after=5.0, snap_width=0.0003, x_stroke=0.005,
                                 latch=True)

    def __init__(
        self,
        robots="Kinova3",
        env_configuration="default",
        controller_configs=None,
        gripper_types="Robotiq85Gripper",
        initialization_noise=None,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        horizon=500,
        random_order=False,
        board_pos_noise=0.06,
        percep_pos_bias=0.015,
        percep_pos_noise=0.005,
        percep_axis_noise=0.05,
        base_z_noise=0.03,
        gain_noise_lin=None,
        gain_noise_rot=None,
        joint_vel_limits=None,
        joint_acc_limits=None,
        joint_acc_limit_ref="qvel",
        joint_acc_limit_mode="clip",
        joint_vel_limit_mode="clip",
        eef_twist_limits=None,
        init_qpos=None,
        spot_body=False,
        spot_front_cam=False,
        obs_acc_saturation=False,
        action_rate_limit=None,
        acc_demand_cost=0.0,
        acc_margin_bonus=0.0,
        joint_acc_limits_noise=None,
        approach_speed_cost=0.0,
        approach_brake_time=0.30,
        spot_collision_cost=0.0,
        button_stiffness=None,
        button_estop_profile=None,
        force_gripper_closed=True,
        finger_press_gate=True,
        finger_misuse_cost=0.0,
        contact_speed_cost=0.0,
        dense_align_coef=0.05,
        collision_cost=5.0,
        action_rate_cost=0.02,
        depress_shaping_k=0.0,
        depress_clip_buffer=0.001,
        # Odsuniecie Spota od sciany. Po sprowadzeniu tablicy na 0.60 m i bazy
        # na 0.65 m poza domowa ramienia (TCP na x~0.43) trafia dokladnie w
        # kapturki (x=0.423) — przy 0.0 epizod startuje z penetracja przyciskow
        # i polityka dostaje kontakt juz w kroku 0. -0.10 daje zero kontaktow
        # startowych dla 6DOF i 7DOF przy pelnej randomizacji (docs 6.15).
        base_x_offset=-0.10,
        activation_depth=None,
        **kwargs,
    ):
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.random_order = random_order
        self.board_pos_noise = board_pos_noise
        self.percep_pos_bias = percep_pos_bias
        self.percep_pos_noise = percep_pos_noise
        self.percep_axis_noise = percep_axis_noise
        self.base_z_noise = base_z_noise
        self.force_gripper_closed = force_gripper_closed
        self.finger_press_gate = finger_press_gate
        self.finger_misuse_cost = finger_misuse_cost
        self.contact_speed_cost = contact_speed_cost
        self.dense_align_coef = dense_align_coef
        self.collision_cost = collision_cost
        self.action_rate_cost = action_rate_cost
        self.depress_shaping_k = depress_shaping_k
        self.depress_clip_buffer = depress_clip_buffer
        self.base_x_offset = base_x_offset
        self.activation_depth = (activation_depth if activation_depth is not None
                                 else self.ACTIVATION_DEPTH)

        # Randomizacja WZMOCNIENIA mechanizmu — patrz docs 6.30.
        # (lo, hi) albo None. Mnoznik losowany NA EPIZOD, OSOBNO dla kazdej
        # z 3 osi liniowych i 3 obrotowych, bo zmierzona rozbieznosc
        # sim2sim byla anizotropowa.
        self.gain_noise_lin = gain_noise_lin
        self.gain_noise_rot = gain_noise_rot
        # Limity predkosci stawow [rad/s], 6 wartosci albo None.
        # NIE jest to randomizacja ani kara — to poprawka WIERNOSCI symulatora,
        # ta sama klasa co sprowadzenie geometrii z Webotsa (docs 6.24).
        # Firmware Gen3 (ros2_kortex PR #241): 1.3963 rad/s ramie, 1.2218 nadgarstek.
        # MuJoCo nie ma limitu predkosci. Limit egzekwowany jest PO calkowaniu
        # kazdego podkroku fizyki (hak na sim.step/step2, _install_joint_limit_hook):
        # v = clip(v_nowe), q = q_poprzednie + v*dt. Integrator to EULER, wiec to
        # dokladnie ruch z predkoscia nie wieksza niz limit — takze w POZYCJI.
        # Poprzednia wersja (saturacja qvel PRZED krokiem) byla NIESZCZELNA: lekki
        # nadgarstek (armature 0, damping 0.01) w jednym podkroku 2 ms przyspieszal
        # ponad limit i osiagal do 2.9x (docs 6.32).
        self.joint_vel_limits = (None if joint_vel_limits is None
                                 else np.asarray(joint_vel_limits, dtype=np.float64))
        # Limity PRZYSPIESZEN [rad/s^2]. Domyslnie WYLACZONE — w odroznieniu od
        # predkosci (0.8727 rad/s = 50 st./s, zgodne z karta katalogowa) wartosci
        # 1.0/10.0 z joint_limits.yaml moga byc limitami PLANERA MoveIt, a nie
        # firmware'u. Nie ograniczamy na podstawie liczby, ktorej pochodzenia
        # nie jestesmy pewni — patrz docs 6.31.
        self.joint_acc_limits = (None if joint_acc_limits is None
                                 else np.asarray(joint_acc_limits, dtype=np.float64))
        # Sposob egzekwowania limitu predkosci (docs 6.37, instrukcja Gen3 R07 s. 100):
        #  "clip"  — kazdy staw przycinany osobno (tryby katowe Kinovy; dotychczasowy
        #            model, zmienia KIERUNEK ruchu),
        #  "scale" — wszystkie stawy skalowane proporcjonalnie, az zaden nie przekracza
        #            limitu; kierunek zachowany (tryby kartezjanskie = twist).
        if joint_vel_limit_mode not in ("clip", "scale"):
            raise ValueError(f"joint_vel_limit_mode: {joint_vel_limit_mode}")
        self.joint_vel_limit_mode = joint_vel_limit_mode
        # Kapy predkosci narzedzia (lin [m/s], ang [rad/s]) albo None. Kinova w trybie
        # twist: |v| <= 0.5 m/s, |omega| <= 0.8727 rad/s, kazda czesc przeskalowana
        # proporcjonalnie. Kinova skaluje KOMENDE; tu skalujemy zrealizowana predkosc
        # stawow (twist = J v, powrot przez tlumione najmniejsze kwadraty) — przyblizenie.
        self.eef_twist_limits = (None if eef_twist_limits is None
                                 else (float(eef_twist_limits[0]), float(eef_twist_limits[1])))
        self._limit_debug = False      # testy: zapis ostatniego podkroku do _limit_last
        self._limit_last = None
        # Sztywnosc sprezyny grzybka [N/m] albo None (= wartosc z button.xml, 500).
        # Poprawka WIERNOSCI obiektu zadania, nie robota ani chwytaka (docs 6.33):
        # typowy E-stop wymaga 20-40 N, model 500 N/m x 12 mm daje ~6 N.
        # 1700 N/m -> 20.4 N, 3300 N/m -> 39.6 N przy progu 12 mm.
        # POZA STARTOWA. None = domyslna z kinova3_6dof_robot.py, czyli
        # [0, 0.65, -1.89, 0, -0.6, -pi/2] — chwytak PIONOWO W DOL (os narzedzia
        # [0.002, 0, -1]). W tej pozie kamera nadgarstkowa Gen3, ktora patrzy
        # wzdluz osi narzedzia, widzi PODLOGE: zmierzone katy do przyciskow to
        # 48.8-120.6 st. przy polowie FOV 32.5 st., czyli 0/5 przyciskow w kadrze,
        # a gorne przyciski sa DOSLOWNIE ZA KAMERA. Percepcja na sprzecie
        # (ArUco -> nadzorca -> target_relative_pos) nie ma z czego wystartowac;
        # w symulacji bylo to niewidoczne, bo env podaje cel z informacji
        # uprzywilejowanej. Lista 6 wartosci nadpisuje poze (docs 6.42).
        self.init_qpos = (None if init_qpos is None
                          else np.asarray(init_qpos, dtype=np.float64))
        # Bryla korpusu Spota pod ramieniem. ZMIERZONE z Webotsa
        # (protos/Spot.proto, galaz kinova_gen3, 2026-10-01): suma AABB
        # wszystkich 16 pudelek z `boundingObject DEF BODY_BO`, Z UWZGLEDNIENIEM
        # ICH OBROTOW (szesc jest obroconych; branie samego `size` zawyza jedne
        # osie i zanizalo inne — pierwsza wersja tej bryly byla przesunieta
        # o 3.6 cm za daleko w strone tablicy i obcieta z tylu o 23.6 cm).
        # Montaz z TF: base_link -> gen3_base_link = [0.256, 0, 0.112], rotacja
        # JEDNOSTKOWA, wiec ramka ramienia jest zorientowana jak ramka korpusu.
        # W RAMCE PODSTAWY RAMIENIA: X -0.729 .. +0.266, Y +-0.139,
        # Z -0.202 .. +0.002 (gorne 2 cm to podkladki pod sam montaz ramienia,
        # celowo POMINIETE, zeby nie wkleic bryly w podstawe robota).
        # W robosuite tam byla PUSTKA — polityka nie miala czego omijac, a na
        # sprzecie i w Webots chwytak ociera sie o Spota (obserwacja uzytkownika
        # 2026-10-01, potwierdzona pomiarem osi: w Webots ramie przejezdza nawet
        # przez tablice, wiec sama scena nie stawia zadnego oporu).
        # DOMYSLNIE WYLACZONE: wlaczenie zmienia zadanie i uniewaznia
        # porownywalnosc z biegami 6.1-6.41.
        self.spot_body = bool(spot_body)
        # Kamera z przodu Spota. W Webots to Kinect (`foot FALSE`, wiec geometria
        # kolizyjna = LISTWA SENSORA Box 0.28 x 0.038 x 0.064, a nie klocek
        # 0.084 ze stojaka — sprawdzone w szablonie protoa). Na sprzecie
        # w laboratorium jest mniejsza kamera, ale wymiarow nie znamy, a decyzja
        # uzytkownika (2026-10-01) jest brac obrys Kinecta jako BUFOR: lepiej
        # trenowac z wieksza przeszkoda niz z zadna.
        # W ramce podstawy ramienia: X +0.209..+0.273, Y +-0.140, Z -0.026..+0.012
        # — listwa 28 cm wszerz, 21-27 cm PRZED podstawa, na jej wysokosci,
        # czyli w korytarzu dojazdu do tablicy (nie pod ramieniem jak korpus).
        self.spot_front_cam = bool(spot_front_cam)
        # Sygnal nasycenia w OBSERWACJI (docs 6.47). DOMYSLNIE WYLACZONY, bo zmienia
        # wymiar obs 120 -> 132, a modele sprzed 2026-10-07 maja wejscie 120 i bez
        # tego nie wczytalyby sie wcale (ta sama niewymiennosc co 7DOF/6DOF).
        self.obs_acc_saturation = bool(obs_acc_saturation)
        # OGRANICZENIE TEMPA KOMENDY (docs 6.47) — OGRANICZENIE, nie kara.
        # |a_t - a_{t-1}| <= action_rate_limit na kazdej skladowej, nakladane ZANIM
        # akcja trafi do kontrolera. Powod: kara jest zacheta, a zachete mozna kupic —
        # wariant A kosztowal 16 % nagrody i polityka ZAPLACILA, zadajac dalej swojego
        # (zadania ponad limit 74.1 -> 70.8 %, czyli szum). Randomizacja wyszla gorzej
        # (78.1 %), bo oplacalo sie zadac WIECEJ na wypadek ciasnego limitu.
        # Ograniczenie jest NIEZBYWALNE — nie istnieje w przestrzeni akcji.
        # Fizycznie uzasadnione: ogranicznik tempa komendy to standard na realnych
        # ramionach, wiec to samo da sie nalozyc na moscie — i wtedy symulacja
        # i sprzet widza IDENTYCZNE ograniczenie, zamiast polegac na tym, ze firmware
        # przytnie tak samo jak nasz hak (przy 78 % krokow na nasyceniu to glowny
        # czynnik ksztaltujacy trajektorie, a nie szczegol).
        self.action_rate_limit = (None if action_rate_limit is None
                                  else float(action_rate_limit))
        self._prev_cmd = None
        # WARIANT A (docs 6.46): kara za ZADANE przyspieszenie ponad limit i premia
        # za zapas pod limitem. Powod: zrealizowane przyspieszenie jest PRZYCINANE,
        # wiec zawsze zgodne — czlon liczony z niego nagradzalby ogranicznik, nie
        # zachowanie. Zepsute jest ZADANIE: polityka prosi o ponad 5.2 w 74.8 %
        # krokow (pomiar 2026-10-06, model bez ogranicznika). Hak zapisuje predkosc
        # PRZED przycieciem, wiec zadanie jest dostepne.
        #   acc_demand_cost  — kara za nadmiar zadania (gradient tam, gdzie polityka lamie)
        #   acc_margin_bonus — premia za zapas (gradient TAM, GDZIE JEST ZGODNA; kara
        #                      zawiasowa daje tam zero, wiec polityka siedzi na scianie)
        # Nagroda i kara roznia sie o STALA, wiec sama zamiana znaku nic by nie dala —
        # dziala dopiero GESTOSC, czyli to, ze gradient istnieje po obu stronach progu.
        self.acc_demand_cost = float(acc_demand_cost)
        self.acc_margin_bonus = float(acc_margin_bonus)
        self._acc_demand = None      # [rad/s^2] zadanie z ostatniego kroku sterowania
        # SKALA, W KTOREJ DZIALA LIMIT PRZYSPIESZEN (docs 6.48)
        #   "qvel" — przycinanie wzgledem ZMIERZONEJ predkosci z poprzedniego podkroku.
        #            Mierzone 2026-10-07: przy a = 0 (ramie STOI, |qvel| 0.02-0.09 rad/s)
        #            |dv|/2 ms ma mediane 40 rad/s^2 na j3 i 236 na j5 — to tetnienie
        #            OSC/solvera. Budzet 5.2 rad/s^2 na podkrok zjada wiec szum,
        #            a na ruch makroskopowy zostaje resztka. Stad spowolnienie 3x (6.44).
        #   "cmd"  — przycinanie wzgledem WLASNEJ poprzedniej komendy. Ogranicza nachylenie
        #            zadanego profilu predkosci, czyli to, co ogranicza firmware Kinovy,
        #            i jest odporne na tetnienie. Ta sama wartosc limitu, inna wielkosc.
        # Domyslnie "qvel" — zeby nie unieważnić calej historii ewaluacji.
        self.joint_acc_limit_ref = str(joint_acc_limit_ref)
        # SEMANTYKA PRZYCINANIA PRZYSPIESZEN (docs 6.50)
        #   "clip"  — przyciecie NIEZALEZNIE na kazdym stawie. Zmienia kierunek ruchu:
        #             staw na limicie zostaje przyciety, pozostale nie, wiec wypadkowa
        #             trajektoria jest inna niz zadana.
        #   "scale" — przeskalowanie CALEGO przyrostu predkosci jednym wspolczynnikiem,
        #             kierunek zachowany. Tak wg instrukcji Gen3 R07 zachowuje sie
        #             firmware w trybie twist (i tak dziala nasz limit PREDKOSCI,
        #             joint_vel_limit_mode="scale").
        # To NIE jest kwestia ostrosci limitu — obie wersje trzymaja te sama obwiednie.
        # Roznica jest w KIERUNKU ruchu po przycieciu, czyli dokladnie w tym, czego
        # polityka pracujaca na nasyceniu moze nie wchlonac (ryzyko transferu z 6.45).
        self.joint_acc_limit_mode = str(joint_acc_limit_mode)
        self._v_cmd_prev = None      # referencja dla trybu "cmd"

        # WARIANT B (docs 6.46): losowanie limitu przyspieszen per epizod, [lo, hi]
        # jako mnoznik wartosci bazowej. Powod: polityka dostroila sie do JEDNEJ
        # implementacji nasycenia (zadania ponad limit 62 % -> 74.8 % po treningu
        # w obwiedni). Randomizacja odbiera eksploatacji sens, zamiast ja karac —
        # ta sama logika co base_z_noise i gain_noise, ktore w tym projekcie dzialaja.
        self.joint_acc_limits_noise = (None if joint_acc_limits_noise is None
                                       else (float(joint_acc_limits_noise[0]),
                                             float(joint_acc_limits_noise[1])))
        self._acc_limits_base = None
        # KARA ZA ZBYT SZYBKI DOJAZD DO CELU (docs 6.45). Powstala z pomiaru:
        # po nalozeniu twardego limitu przyspieszen Gen3 (5.2 rad/s^2, tab. 48)
        # kolizyjnosc skoczyla 0.4 -> 67.2 kroku/epizod, a 6.8M krokow treningu
        # zbilo ja tylko do 32.3 i krzywa sie WYPLASZCZYLA (67.2 -> 52.9 -> 36.0
        # -> 32.3). Mechanizm: polityka hamowala przed tablica gwaltownym
        # przeskokiem komendy, a limit ten hamulec obcina — ramie dojezdza
        # szybciej, niz zdoła wytracic predkosc.
        #
        # Istniejace czlony kary dzialaja ZA POZNO (collision_cost i
        # contact_speed_cost odpalaja PO kontakcie) albo za ogolnie
        # (action_rate_cost karze szarpanie wszedzie, nie tylko przy przeszkodzie).
        # Ten czlon dziala PRZED kontaktem i tylko tam, gdzie trzeba.
        #
        # Postac: kara za predkosc zblizania PONAD te, z ktorej da sie jeszcze
        # wyhamowac. Dopuszczalna predkosc rosnie liniowo z dystansem:
        #   v_dop = dystans / approach_brake_time
        # czyli "miej tyle predkosci, zeby zatrzymac sie w czasie tau". Karane
        # jest tylko przekroczenie i tylko skladowa W STRONE celu — cofanie sie
        # i ruch boczny sa wolne, zeby nie dlawic manewru obrotu nadgarstka,
        # ktorym ta polityka rozwiazuje zadanie.
        self.approach_speed_cost = float(approach_speed_cost)
        self.approach_brake_time = float(approach_brake_time)
        # Kara za kontakt z korpusem Spota, per krok, PRZED normalizacja (225).
        # NIE kopiowac collision_cost=5.0 z tablicy: przy zmierzonych setkach
        # krokow kontaktu na epizod dalaby karę wielokrotnie wieksza niz cala
        # nagroda zadania i polityka nauczylaby sie stac (tak wywrocily sie
        # proby 6.20). Wartosc skalowana z POMIARU czestosci kontaktu.
        self.spot_collision_cost = float(spot_collision_cost)
        self.button_stiffness = button_stiffness
        # Nieliniowy, zatrzaskowy model E-stopu albo None (= liniowa sprezyna).
        # Sila [N] przeciwna wciskaniu: od f_start rosnie liniowo do f_peak przy
        # x_peak, na odcinku snap_width spada do f_after (rozwarcie stykow), dalej
        # stala. Skok przyciety do x_stroke; przy latch=True grzybek zostaje na dole
        # do resetu. Wcisniecie liczy sie w chwili spadku sily (x_peak + snap_width),
        # chyba ze podano activation_depth jawnie.
        self.button_estop_profile = (None if button_estop_profile is None
                                     else dict(button_estop_profile))
        if self.button_estop_profile is not None:
            _p = self.button_estop_profile
            _miss = {"f_start", "f_peak", "x_peak", "f_after", "snap_width", "x_stroke"} - set(_p)
            if _miss:
                raise ValueError(f"button_estop_profile: brak kluczy {sorted(_miss)}")
            if self.button_stiffness is not None:
                raise ValueError("button_stiffness i button_estop_profile wykluczaja sie")
            if activation_depth is None:
                self.activation_depth = _p["x_peak"] + _p["snap_width"]
            if self.activation_depth > _p["x_stroke"]:
                raise ValueError("activation_depth wiekszy niz x_stroke")
        self._btn_latched = [False] * self.N_BUTTONS
        self._gain = np.ones(6)
        self._raw_action = None

        self._base_z = self.KINOVA_BASE_Z
        self._current_idx = 0
        self._pressed = [False] * self.N_BUTTONS
        # Per-episode perception errors (redrawn in _reset_internal)
        self._bias_pos = np.zeros(3)
        self._bias_axis = np.zeros(3)
        # Potential-based shaping state (reward v2)
        self._prev_dist = None
        self._prev_target = None
        self._prev_action = None
        self._prev_cmd = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            horizon=horizon,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # XML path fix (same workaround as other envs in this project)
    # ------------------------------------------------------------------

    def edit_model_xml(self, xml_str):
        path_split = _ROBOSUITE_PKG_PATH.split("/")
        tree = ET.fromstring(xml_str)
        asset = tree.find("asset")
        if asset is not None:
            for elem in asset.findall("mesh") + asset.findall("texture"):
                old_path = elem.get("file")
                if old_path is None or not old_path.startswith(_ROBOSUITE_PKG_PATH):
                    continue
                old_path_split = old_path.split("/")
                check_lst = [i for i, v in enumerate(old_path_split) if v == "robosuite"]
                if check_lst:
                    ind = max(check_lst)
                    new_path = "/".join(path_split + old_path_split[ind + 1:])
                    elem.set("file", new_path)
        return ET.tostring(tree, encoding="utf8").decode("utf8")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _btn_body_pos(self, i):
        """World position of button body origin (centre of its mounting face)."""
        dx, dy, dz = self.BUTTON_FACES[i][0]
        return (self.WALL_X + dx, self._board_y + dy, self._board_z + dz)

    @staticmethod
    def _face_quat(normal):
        """Quat (wxyz string) rotating button local +Z onto the face normal."""
        z = np.asarray(normal, dtype=float)
        z = z / np.linalg.norm(z)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up, z)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        q = T.mat2quat(np.column_stack([x, y, z]))  # xyzw
        return f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}"

    def _load_model(self):
        super()._load_model()

        # Per-episode domain randomisation: board centre shifts in Y/Z
        yz_noise = np.random.uniform(-self.board_pos_noise,
                                     self.board_pos_noise, size=2)
        self._board_y = self.BOARD_Y + yz_noise[0]
        self._board_z = self.BOARD_Z + yz_noise[1]

        # Kinova na wysokosci Spota; base_x_offset = Spot zaparkowany blizej
        # sciany (pokretlo sim2real, patrz docstring).
        # set_base_xpos() ustawia SPOD cokolu, a nie baze ramienia, wiec
        # KINOVA_BASE_Z (wysokosc bezwzgledna) trzeba skorygowac o wysokosc
        # cokolu; bottom_offset jest ujemny (RethinkMount: -0.912 m).
        # Randomizacja WYSOKOSCI BAZY. Ablacja 7DOF (docs 6.22) pokazala, ze
        # przeniesienie bazy jest jedynym ZMIERZONYM trybem awarii transferu:
        # wzgledna reprezentacja celu (target_relative_pos, target_press_axis)
        # daje odpornosc na poze TABLICY, ale nie na poze BAZY, bo obs zawiera
        # takze bezwzgledne katy stawow i pozycje EEF. Na Spocie wysokosc
        # korpusu jest sterowalna, a montaz nie bedzie trafiony co do
        # centymetra — wiec ta os musi byc w treningu, nie w kalibracji.
        self._gain = np.ones(6)
        if self.gain_noise_lin is not None:
            lo, hi = self.gain_noise_lin
            self._gain[0:3] = np.random.uniform(lo, hi, 3)
        if self.gain_noise_rot is not None:
            lo, hi = self.gain_noise_rot
            self._gain[3:6] = np.random.uniform(lo, hi, 3)

        if self.joint_acc_limits_noise is not None and self.joint_acc_limits is not None:
            if self._acc_limits_base is None:
                self._acc_limits_base = np.array(self.joint_acc_limits, dtype=np.float64)
            lo, hi = self.joint_acc_limits_noise
            self.joint_acc_limits = self._acc_limits_base * float(np.random.uniform(lo, hi))

        self._base_z = self.KINOVA_BASE_Z + np.random.uniform(
            -self.base_z_noise, self.base_z_noise)
        mount_z = float(self.robots[0].robot_model.bottom_offset[2])
        self.robots[0].robot_model.set_base_xpos(
            [self.base_x_offset, 0, self._base_z + mount_z])
        if self.init_qpos is not None:
            # robot_model.init_qpos to property BEZ settera, wiec ustawiamy
            # atrybut instancji. robosuite bierze wartosc z modelu tylko gdy
            # instancja ma None (robot.py:176), a stosuje ja w _reset_internal
            # (robot.py:245) — wiec to wystarcza i przezywa hard_reset.
            self.robots[0].init_qpos = self.init_qpos.copy()

        mujoco_arena = EmptyArena()

        # Camera: side-front view — robot centre-left, wall+buttons right
        # pos=(-0.2, -1.5, 1.5) looking toward (0.3, 0, 1.2)
        # view direction (0.31, 0.93, -0.19), xyaxes computed from that
        cam = ET.SubElement(mujoco_arena.worldbody, "camera")
        cam.set("name", "board_view")
        # Kadr dobrany pod geometrie z docs 6.15 (tablica 0.60 m, baza 0.65 m).
        # 2026-09-01 tablica poszla na 0.64 — kadr jest o ~4 cm nizej niz
        # srodek tablicy. Wplywa tylko na nagrania (watch_board --record),
        # nie na obserwacje ani nagrode; przestroic przy okazji.
        cam.set("pos", "-0.25 -0.75 0.95")
        cam.set("xyaxes", "0.7458 -0.6662 0 0.2190 0.2452 0.9445")

        # PHYSICAL wall panel behind the board. Until 2026-08-05 this geom was
        # visual only (contype/conaffinity = 0), so nothing stopped the arm
        # from reaching THROUGH the board and operating from behind it — a
        # pose that is impossible on hardware, where the omniboard is bolted
        # to a solid wall. Policies trained against the visual-only wall can
        # therefore learn approaches that do not transfer (docs 6.19).
        #
        # No spurious contact from the flush mounting: the board's back face
        # and the wall's front face both sit at WALL_X, but the board is a
        # static object (OmniBoardObject joints=None), so both geoms weld to
        # the world and MuJoCo filters the pair. Buttons DO carry slide
        # joints, but they sit on the front face (x ~ 0.42), 8 cm clear of
        # the wall.
        wall = ET.SubElement(mujoco_arena.worldbody, "geom")
        wall.set("name", "wall_panel")
        wall.set("type", "box")
        # half-sizes: 10 cm deep, 60 cm wide, 50 cm tall; face flush with board back
        wall.set("size", "0.10 0.60 0.50")
        wall.set("pos", f"{self.WALL_X + 0.10:.4f} "
                        f"{self._board_y:.4f} {self._board_z:.4f}")
        wall.set("rgba", "0.55 0.55 0.55 1.0")
        # group 1 = grupa wizualna. Bez tego geom siedzi w grupie 0 (kolizyjnej),
        # ktorej renderer robosuite NIE rysuje — sciana byla FIZYCZNIE obecna od
        # 2026-08-05 (docs 6.19), ale NIEWIDOCZNA na wszystkich nagraniach od tamtej
        # pory. Czysto wizualne; na fizyke i obserwacje nie wplywa.
        wall.set("group", "1")
        wall.set("contype", "1")
        wall.set("conaffinity", "1")

        if self.spot_body:
            # Jedna bryla obejmujaca korpus + przednia sekcje (konserwatywnie,
            # czyli nieco wieksza niz suma bryl Webotsa). Srodek i polowki
            # wyliczone z zakresow w __init__.
            spot = ET.SubElement(mujoco_arena.worldbody, "geom")
            spot.set("name", "spot_body_collision")
            spot.set("type", "box")
            # polowki: (0.266+0.729)/2, 0.139, (0.002+0.202)/2
            spot.set("size", "0.4975 0.1390 0.1020")
            spot.set("pos", f"{self.base_x_offset - 0.2315:.4f} 0 "
                            f"{self._base_z - 0.1000:.4f}")
            spot.set("rgba", "0.85 0.45 0.10 1.0")   # pomaranczowy: grafit zlewal sie z cokolem
            spot.set("group", "1")   # bez tego geom jest w grupie kolizyjnej i NIE JEST RYSOWANY
            spot.set("contype", "1")
            spot.set("conaffinity", "1")

        if self.spot_front_cam:
            cam_g = ET.SubElement(mujoco_arena.worldbody, "geom")
            cam_g.set("name", "spot_frontcam_collision")
            cam_g.set("type", "box")
            cam_g.set("size", "0.0320 0.1400 0.0190")
            cam_g.set("pos", f"{self.base_x_offset + 0.2410:.4f} 0 "
                             f"{self._base_z - 0.0070:.4f}")
            cam_g.set("rgba", "0.10 0.55 0.85 1.0")  # niebieski, zeby odrozniac listwe od korpusu
            cam_g.set("group", "1")
            cam_g.set("contype", "1")
            cam_g.set("conaffinity", "1")

        # Omniboard: big (back) face on the wall at WALL_X, block tapers toward
        # the robot. Body local +Z (depth axis) must map to world -X:
        # rotate -90 deg around Y.
        self._board_obj = OmniBoardObject(name="omniboard")
        self._board_obj.get_obj().set(
            "pos", f"{self.WALL_X:.4f} {self._board_y:.4f} {self._board_z:.4f}")
        # Bryla tablicy dostaje ten sam obrot wokol normalnej co BUTTON_FACES
        # (patrz _rolled_button_faces), inaczej przyciski zwisalyby obok
        # swoich scianek. R = Rx(roll) @ Ry(-90) — Ry(-90) mapuje lokalne +Z
        # bryly na -X swiata, Rx(roll) obraca uklad scianek w plaszczyznie YZ.
        a = np.radians(self.BOARD_ROLL_DEG)
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(a), -np.sin(a)],
                       [0.0, np.sin(a), np.cos(a)]])
        ry = np.array([[0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0]])
        bq = T.mat2quat(rx @ ry)  # xyzw
        self._board_obj.get_obj().set(
            "quat", f"{bq[3]:.6f} {bq[0]:.6f} {bq[1]:.6f} {bq[2]:.6f}")

        # 5 buttons: centre on the front face, 4 on the 45-deg bevels
        self._button_objs = []
        for i in range(self.N_BUTTONS):
            btn = ButtonObject(name=f"btn{i}")
            bx, by, bz = self._btn_body_pos(i)
            btn.get_obj().set("pos", f"{bx:.4f} {by:.4f} {bz:.4f}")
            btn.get_obj().set("quat", self._face_quat(self.BUTTON_FACES[i][1]))
            self._button_objs.append(btn)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self._board_obj] + self._button_objs,
        )

    # ------------------------------------------------------------------
    # References (IDs into sim arrays, set up after model compile)
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()

        self._btn_top_ids = [
            self.sim.model.site_name2id(f"btn{i}_button_top")
            for i in range(self.N_BUTTONS)
        ]

        jnt_ids = [
            self.sim.model.joint_name2id(f"btn{i}_button_slide")
            for i in range(self.N_BUTTONS)
        ]
        self._btn_qposadr = [self.sim.model.jnt_qposadr[j] for j in jnt_ids]
        self._btn_dofadr  = [self.sim.model.jnt_dofadr[j]  for j in jnt_ids]

        # Geom IDs for the v2.2 press gate / button-misuse penalty
        arm = self.robots[0].arms[0]
        gripper = self.robots[0].gripper[arm]
        finger_names = (gripper.important_geoms["left_finger"]
                        + gripper.important_geoms["right_finger"])
        self._finger_geom_ids = {self.sim.model.geom_name2id(n)
                                 for n in finger_names}
        self._cap_geom_ids = [
            self.sim.model.geom_name2id(f"btn{i}_cap_collision")
            for i in range(self.N_BUTTONS)
        ]
        self._btn_geom_ids = set(self._cap_geom_ids) | {
            self.sim.model.geom_name2id(f"btn{i}_base_collision")
            for i in range(self.N_BUTTONS)
        }
        # v2.3: misuse = only ARM links (robot0_*) on a button; gripper
        # housing/knuckles must stay free to touch during fingertip presses
        self._arm_geom_ids = {
            gid for gid in range(self.sim.model.ngeom)
            if (self.sim.model.geom_id2name(gid) or "").startswith("robot")
        }
        # v2.4: geom -> button index, to tell WHICH button a finger touches
        self._geom_to_btn = {}
        for i in range(self.N_BUTTONS):
            self._geom_to_btn[self._cap_geom_ids[i]] = i
            self._geom_to_btn[self.sim.model.geom_name2id(
                f"btn{i}_base_collision")] = i
        eef_site_id = self.robots[0].eef_site_id[arm]
        self._eef_site_name = self.sim.model.site_id2name(eef_site_id)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()

        for i in range(self.N_BUTTONS):
            self.sim.data.qpos[self._btn_qposadr[i]] = 0.0
            self.sim.data.qvel[self._btn_dofadr[i]]  = 0.0
            self.sim.data.qfrc_applied[self._btn_dofadr[i]] = 0.0
        self._btn_latched = [False] * self.N_BUTTONS
        self._v_cmd_prev = None

        self._current_idx = 0
        self._pressed = [False] * self.N_BUTTONS

        if self.random_order:
            self._order = np.random.permutation(self.N_BUTTONS).tolist()
        else:
            self._order = list(range(self.N_BUTTONS))

        # Per-episode perception error: constant board-pose estimation bias
        self._bias_pos = np.random.uniform(
            -self.percep_pos_bias, self.percep_pos_bias, size=3)
        self._bias_axis = np.random.normal(0.0, self.percep_axis_noise, size=3)

        # Reset shaping state (reward v2)
        self._prev_dist = None
        self._prev_target = None
        self._prev_action = None
        self._prev_depress = None
        self._prev_depress_target = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_btn(self):
        """Index of the button the robot should press right now."""
        if self._current_idx >= self.N_BUTTONS:
            return self.N_BUTTONS - 1  # all done, avoid out-of-bounds
        return self._order[self._current_idx]

    def _depression(self, i):
        return float(self.sim.data.qpos[self._btn_qposadr[i]])

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def target_relative_pos(obs_cache):
            """EEF->target vector as the PERCEPTION would see it:
            true vector + per-episode board-pose bias + per-step noise."""
            eef = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            tgt = self.sim.data.site_xpos[self._btn_top_ids[self._target_btn()]]
            noise = np.random.normal(0.0, self.percep_pos_noise, size=3)
            return (tgt - eef) + self._bias_pos + noise

        @sensor(modality="object")
        def target_distance(obs_cache):
            """Norm of the perceived (noisy) relative vector — consistent
            with target_relative_pos computed earlier this step."""
            rel = obs_cache.get("target_relative_pos")
            if rel is None:
                eef = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
                tgt = self.sim.data.site_xpos[self._btn_top_ids[self._target_btn()]]
                rel = (tgt - eef) + self._bias_pos
            return np.array([np.linalg.norm(rel)])

        @sensor(modality="object")
        def target_onehot(obs_cache):
            """One-hot: which button is current target (5-dim)."""
            oh = np.zeros(self.N_BUTTONS, dtype=np.float32)
            oh[self._target_btn()] = 1.0
            return oh

        @sensor(modality="object")
        def target_press_axis(obs_cache):
            """Push direction of the current target as PERCEIVED
            (per-episode board orientation estimation error)."""
            n = np.asarray(self.BUTTON_FACES[self._target_btn()][1], dtype=np.float64)
            axis = -n + self._bias_axis
            return (axis / np.linalg.norm(axis)).astype(np.float32)

        @sensor(modality="object")
        def eef_align_with_target(obs_cache):
            """Angle [rad] between EEF +Z axis and the perceived press axis."""
            eef_quat = obs_cache.get(f"{pf}eef_quat", np.array([0, 0, 0, 1]))
            eef_z = T.quat2mat(eef_quat)[:, 2]
            axis = obs_cache.get("target_press_axis")
            if axis is None:
                axis = -np.asarray(self.BUTTON_FACES[self._target_btn()][1])
            cos_a = np.clip(np.dot(eef_z, axis), -1.0, 1.0)
            return np.array([np.arccos(np.abs(cos_a))])

        @sensor(modality="object")
        def joint_acc_saturation(obs_cache):
            """Ile komenda zostala PRZYCIETA przez limit przyspieszen, per staw.

            0.0 = polityka miesci sie w obwiedni; rosnie wraz z nadmiarem zadania,
            1.0 = zadanie dwukrotnie ponad limit (przyciete do 1.0 powyzej).

            POWOD (docs 6.47). Zadanie ponad limit jest dla polityki NIEOBSERWOWALNE:
            ruch po przycieciu jest identyczny niezaleznie od tego, o ile poprosila.
            Przez to nie ma czego powiazac z kara i trzy proby jej nalozenia (6.43,
            6.45, 6.46) nie zbily zadan. Ten sygnal daje bezposrednia przeslanke
            "twoja komenda zostala przycieta", ktora polityka moze powiazac
            z wlasnym wyjsciem.

            ZEROWY gdy limit przyspieszen wylaczony — wtedy nie ma przycinania.
            """
            if self.joint_acc_limits is None or self._acc_demand is None:
                return np.zeros(6, dtype=np.float32)
            lim = np.asarray(self.joint_acc_limits, dtype=np.float64)
            nad = np.clip((self._acc_demand - lim) / lim, 0.0, 1.0)
            return nad.astype(np.float32)

        new_sensors = ([joint_acc_saturation] if self.obs_acc_saturation else []) + [
            target_relative_pos,
            target_distance,
            target_onehot,
            target_press_axis,
            eef_align_with_target,
        ]
        new_names = (["joint_acc_saturation"] if self.obs_acc_saturation else []) + [
            "target_relative_pos",
            "target_distance",
            "target_onehot",
            "target_press_axis",
            "eef_align_with_target",
        ]

        for name, s in zip(new_names, new_sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Limity napedu stawow (docs 6.31, 6.32)
    # ------------------------------------------------------------------

    def _initialize_sim(self, xml_string=None):
        super()._initialize_sim(xml_string)
        # hard_reset tworzy NOWY obiekt sim przy kazdym resecie — hak trzeba
        # instalowac za kazdym razem, inaczej dziala tylko do pierwszego resetu.
        self._install_joint_limit_hook()
        self._apply_button_model()

    def _apply_button_model(self):
        # Nadpisanie po kompilacji modelu; hard_reset tworzy nowy sim, wiec wolane
        # przy kazdej inicjalizacji. getattr: pierwsza inicjalizacja moze nastapic
        # w trakcie super().__init__, zanim atrybuty istnieja — kazdy reset go poprawia.
        k = getattr(self, "button_stiffness", None)
        prof = getattr(self, "button_estop_profile", None)
        if k is None and prof is None:
            return
        M = self.sim.model._model
        for i in range(self.N_BUTTONS):
            jid = self.sim.model.joint_name2id(f"btn{i}_button_slide")
            if prof is not None:
                M.jnt_stiffness[jid] = 0.0          # sile daje profil, nie sprezyna
                M.jnt_range[jid, 1] = float(prof["x_stroke"])
            else:
                M.jnt_stiffness[jid] = float(k)

    def _estop_force(self, x):
        """Sila [N] przeciwna wciskaniu dla glebokosci x [m] wg button_estop_profile."""
        p = self.button_estop_profile
        if x <= 0.0:
            return p["f_start"]
        if x < p["x_peak"]:
            return p["f_start"] + (p["f_peak"] - p["f_start"]) * x / p["x_peak"]
        x1 = p["x_peak"] + p["snap_width"]
        if x < x1:
            return p["f_peak"] + (p["f_after"] - p["f_peak"]) * (x - p["x_peak"]) / p["snap_width"]
        return p["f_after"]

    def _install_joint_limit_hook(self):
        # Oba punkty calkowania: robosuite przy lite_physics wola step1/step2
        # zamiast step. Hak na samo step bylby wtedy martwy.
        #
        # WAZNE (docs 6.32, OOM 2026-09-14): NIE przechwytywac metody zwiazanej
        # self.sim.<name>. Domkniecie trzymajace ja tworzy cykl
        # sim -> domkniecie -> metoda zwiazana -> sim, wiec po twardym resecie
        # stary MjSim z cala pamiecia C MuJoCo czeka na cykliczny GC zamiast
        # zwolnic sie od razu (test: RSS pilokształtnie 524 -> 957 MB w 120 resetach).
        # Wolamy implementacje z KLASY na biezacym self.sim — domkniecie nie ma
        # referencji do obiektu sim.
        cls = type(self.sim)
        for name in ("step", "step2"):
            fn = getattr(cls, name, None)
            if fn is not None:
                setattr(self.sim, name, self._limited_integration(fn))

    def _limited_integration(self, fn):
        def integrate(*args, **kwargs):
            d = self.sim.data
            prof = getattr(self, "button_estop_profile", None)
            btn_ok = prof is not None and getattr(self, "_btn_qposadr", None) is not None
            if btn_ok:
                if getattr(self, "_btn_latched", None) is None:
                    self._btn_latched = [False] * self.N_BUTTONS
                for i in range(self.N_BUTTONS):
                    da = self._btn_dofadr[i]
                    if self._btn_latched[i]:
                        d.qfrc_applied[da] = 0.0
                    else:
                        d.qfrc_applied[da] = -self._estop_force(float(d.qpos[self._btn_qposadr[i]]))
            lim_v, lim_a = self.joint_vel_limits, self.joint_acc_limits
            tw = getattr(self, "eef_twist_limits", None)
            robots = getattr(self, "robots", None)
            arm_ok = not ((lim_v is None and lim_a is None and tw is None) or not robots or
                          getattr(robots[0], "_ref_joint_vel_indexes", None) is None)
            if arm_ok:
                vidx = robots[0]._ref_joint_vel_indexes
                qidx = robots[0]._ref_joint_pos_indexes
                q_prev = np.array(d.qpos[qidx], dtype=np.float64)
                v_prev = np.array(d.qvel[vidx], dtype=np.float64)
            out = fn(self.sim, *args, **kwargs)
            if arm_ok:
                v = np.array(d.qvel[vidx], dtype=np.float64)
                v_in = v.copy()
                if tw is not None:
                    M = self.sim.model._model
                    site = robots[0].eef_site_id[robots[0].arms[0]]
                    jacp = np.zeros((3, M.nv)); jacr = np.zeros((3, M.nv))
                    mujoco.mj_jacSite(M, self.sim.data._data, jacp, jacr, site)
                    J = np.vstack([jacp[:, vidx], jacr[:, vidx]])
                    t = J @ v
                    nl, na = np.linalg.norm(t[:3]), np.linalg.norm(t[3:])
                    if nl > tw[0] or na > tw[1]:
                        t2 = np.concatenate([t[:3] * min(1.0, tw[0] / max(nl, 1e-12)),
                                             t[3:] * min(1.0, tw[1] / max(na, 1e-12))])
                        lam2 = 1e-6
                        v = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), t2)
                v_tw = v.copy()
                if lim_v is not None:
                    if getattr(self, "joint_vel_limit_mode", "clip") == "scale":
                        ratio = np.max(np.abs(v) / lim_v)
                        if ratio > 1.0:
                            v = v / ratio
                    else:
                        v = np.clip(v, -lim_v, lim_v)
                # PRZYSPIESZENIA NA KONCU, nie na poczatku. Do 2026-10-05 ten clip
                # byl PIERWSZY, a kapy twist i limit predkosci zmienialy v PO nim —
                # i potrafily zmienic ja o wiecej, niz przyspieszenie dopuszcza
                # (zmierzone: j1 33.9 rad/s^2 przy limicie 5.2, szesciokrotnie).
                # Na koncu jest bezpiecznie takze dla pozostalych limitow: v_prev
                # i v sa oba w obwiedni predkosci, a clip ku v_prev daje punkt
                # POMIEDZY nimi, wiec obwiedni nie narusza (zbior jest wypukly).
                if lim_a is not None:
                    # Referencja przycinania: zmierzona predkosc albo wlasna poprzednia
                    # komenda (patrz komentarz przy self.joint_acc_limit_ref).
                    if self.joint_acc_limit_ref == "cmd":
                        if self._v_cmd_prev is None:
                            self._v_cmd_prev = v_prev.copy()
                        v_ref = self._v_cmd_prev
                    else:
                        v_ref = v_prev
                    # ZADANIE przed przycieciem — jedyne miejsce, gdzie je widac
                    zad = np.abs(v - v_ref) / max(self.model_timestep, 1e-12)
                    self._acc_demand = (zad if self._acc_demand is None
                                        else np.maximum(self._acc_demand, zad))
                    dv = lim_a * self.model_timestep
                    if self.joint_acc_limit_mode == "scale":
                        # Jeden wspolczynnik na caly przyrost — kierunek zachowany
                        d_req = v - v_ref
                        ratio = np.max(np.abs(d_req) / dv)
                        if ratio > 1.0:
                            v = v_ref + d_req / ratio
                    else:
                        v = np.clip(v, v_ref - dv, v_ref + dv)
                    self._v_cmd_prev = v.copy()
                if getattr(self, "_limit_debug", False):
                    self._limit_last = dict(v_in=v_in, v_tw=v_tw, v=v.copy(), q_prev=q_prev,
                                            J=(J if tw is not None else None))
                d.qvel[vidx] = v
                d.qpos[qidx] = q_prev + v * self.model_timestep
            if btn_ok:
                # Twarde ograniczniki kinematyczne, jak zatrzask. Miekki limit stawu
                # MuJoCo (solref 0.02) przy 36 g grzybka ustepuje 8 mm pod sila
                # wstepna 15 N (docs 6.34) — stad clamp zamiast jnt_range.
                xs = float(prof["x_stroke"])
                latch = prof.get("latch", True)
                for i in range(self.N_BUTTONS):
                    qa, da = self._btn_qposadr[i], self._btn_dofadr[i]
                    if latch and (self._btn_latched[i] or d.qpos[qa] >= xs - 1e-4):
                        self._btn_latched[i] = True
                        d.qpos[qa] = xs
                        d.qvel[da] = 0.0
                    elif d.qpos[qa] <= 0.0:
                        d.qpos[qa] = 0.0
                        d.qvel[da] = max(float(d.qvel[da]), 0.0)
                    elif d.qpos[qa] >= xs:
                        d.qpos[qa] = xs
                        d.qvel[da] = min(float(d.qvel[da]), 0.0)
            return out
        return integrate

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        if self.force_gripper_closed:
            action[-1] = 1.0  # Robotiq85: 1 = close
        # Kara za akcje liczy sie na tym, co WYSTAWILA POLITYKA, a nie na
        # akcji po przeskalowaniu — inaczej randomizacja wzmocnienia
        # zmienialaby reward i mieszala dwie zmienne (docs 6.30).
        if self.action_rate_limit is not None:
            if self._prev_cmd is not None:
                d = self.action_rate_limit
                action[0:6] = np.clip(action[0:6], self._prev_cmd[0:6] - d,
                                      self._prev_cmd[0:6] + d)
            self._prev_cmd = action.copy()
        # Kara za akcje liczy sie na tym, co WYSTAWILA POLITYKA — ale po ograniczeniu
        # tempa, bo to ono jest faktyczna komenda trafiajaca do kontrolera.
        self._raw_action = action.copy()
        if self.gain_noise_lin is not None or self.gain_noise_rot is not None:
            action[0:6] = np.clip(action[0:6] * self._gain, -1.0, 1.0)
        return super().step(action)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    # Reward — wagi stale. Domyslna konfiguracja parametrow konstruktora
    # (finger_misuse_cost=0, contact_speed_cost=0, dense_align_coef=0.05,
    # finger_press_gate=True) = reward v2.3 (ostateczny; v2.5 ODRZUCONY,
    # jego kary sa dostepne jako opt-in przez parametry konstruktora).
    SHAPING_K = 10.0        # nagroda za kazdy metr zblizenia do celu
    PRESS_BONUS = 25.0      # za wcisniecie biezacego celu
    COMPLETE_BONUS = 100.0  # za komplet 5 przyciskow
    DENSE_COEF = 0.05       # sladowy skladnik odleglosci (gradient na starcie nauki)
    ACTION_COST = 0.01      # kara za wielkosc akcji
    ACTION_RATE_COST = 0.02  # domyslne action_rate_cost (v2.1+; v2 scratch: 0)
    COLLISION_COST = 5.0    # domyslne collision_cost (v2.1+; v2 scratch: 0.5)
    CONTACT_SPEED_THRESH = 0.15  # m/s; prog dla contact_speed_cost (v2.5)
    REWARD_NORM = 225.0     # ~max epizodu: 5*25 + 100

    def reward(self, action=None):
        if self._raw_action is not None:
            action = self._raw_action        # patrz step(): kara na akcji polityki
        tgt = self._target_btn()
        depress = self._depression(tgt)

        # Potential-based shaping on the target's depression (see docstring).
        # Uses the PRE-advance target; after a successful press the target
        # switches, the guard skips one step, so the released (pressed)
        # button never pays the shaping back — same guard as dist shaping.
        #
        # REWARD CLIPPING: the depth entering the shaping is capped at
        # activation_depth + depress_clip_buffer. Past that point the button
        # is already registered as pressed, so extra travel buys nothing on
        # the task but does buy contact force — and the collision penalty
        # charges for exactly that. Uncapped, the two terms pull in opposite
        # directions and the run collapses late (docs 6.18). Clipping the
        # potential keeps the term telescoping: the capped depth is still a
        # function of state alone, so the optimal policy is unchanged.
        depress_shaped = min(depress,
                             self.activation_depth + self.depress_clip_buffer)
        r_depress = 0.0
        if (self.depress_shaping_k
                and self._prev_depress_target == tgt
                and self._prev_depress is not None):
            r_depress = self.depress_shaping_k * (depress_shaped
                                                  - self._prev_depress)
        self._prev_depress_target = tgt
        self._prev_depress = depress_shaped

        # Advance sequence when current button is activated. v2.2 gate:
        # the press must come from a gripper finger on the cap — the
        # hardware supervisor detects presses at the TCP and would never
        # register one made with the wrist or arm. The gate can be
        # disabled (finger_press_gate=False) for curriculum stage 1:
        # from-scratch training with the gate collapses to "do nothing".
        press_bonus = 0.0
        if (
            self._current_idx < self.N_BUTTONS
            and depress >= self.activation_depth
            and not self._pressed[tgt]
            and (not self.finger_press_gate or self._finger_on_cap(tgt))
        ):
            self._pressed[tgt] = True
            self._current_idx += 1
            press_bonus = self.PRESS_BONUS

        # Reward uses TRUE state (sim-only signal, no perception noise)
        arm = self.robots[0].arms[0]
        eef_site = self.robots[0].eef_site_id[arm]
        eef_pos = self.sim.data.site_xpos[eef_site]
        tgt = self._target_btn()  # may have advanced above
        tgt_pos = self.sim.data.site_xpos[self._btn_top_ids[tgt]]
        dist = float(np.linalg.norm(tgt_pos - eef_pos))

        eef_z = self.sim.data.site_xmat[eef_site].reshape(3, 3)[:, 2]
        axis = -np.asarray(self.BUTTON_FACES[tgt][1])
        cos_a = np.clip(np.dot(eef_z, axis), -1.0, 1.0)
        axis_err = float(np.arccos(np.abs(cos_a)))

        # Potential-based shaping: pays only for PROGRESS toward the target;
        # hovering earns 0, moving away pays back. No term on target switch
        # (a switched target must not look like a shaping penalty).
        if self._prev_target == tgt and self._prev_dist is not None:
            r_shape = self.SHAPING_K * (self._prev_dist - dist)
        else:
            r_shape = 0.0
        self._prev_target = tgt
        self._prev_dist = dist

        r_dense = (self.DENSE_COEF * np.exp(-8.0 * dist)
                   + self.dense_align_coef * np.exp(-3.0 * axis_err))
        r_complete = self.COMPLETE_BONUS if all(self._pressed) else 0.0
        r_collision = (-self.collision_cost
                       if (self._check_board_collision()
                           or self._check_button_misuse()) else 0.0)
        # Korpus Spota: osobny wspolczynnik, bo czestosc kontaktu jest o rzedy
        # wielkosci wieksza niz z tablica, a na sprzecie to uderzanie we wlasna
        # platforme (ryzyko dla kamery nadgarstkowej i samego montazu).
        if self.spot_collision_cost and self._check_spot_collision():
            r_collision -= self.spot_collision_cost

        # v2.5 (fixes v2.4): penalize clumsy gripper contact — bumping a
        # wrong button, and doing so fast. Both penalties fire ONLY on
        # contact with a button that is neither the current target NOR
        # already pressed. The pressed-exclusion is critical: the target
        # advances on the very step a press succeeds, while the finger
        # keeps touching the just-pressed cap for several retreat steps —
        # in v2.4 those steps counted as "wrong contact" and (with fast
        # retreat) triggered the speed penalty too, so every successful
        # press was immediately punished (~93% of all "wrong contact"
        # steps were this artefact). Target contact stays penalty-free:
        # taxing correct presses collapsed the policy twice (v2.2, v2.4).
        r_finger_misuse = 0.0
        r_contact_speed = 0.0
        if self.finger_misuse_cost or self.contact_speed_cost:
            touched = self._finger_touched_buttons()
            wrong_touched = any(i != tgt and not self._pressed[i]
                                for i in touched)
            if wrong_touched:
                r_finger_misuse = -self.finger_misuse_cost
                if self.contact_speed_cost:
                    speed = float(np.linalg.norm(
                        self.sim.data.get_site_xvelp(self._eef_site_name)))
                    if speed > self.CONTACT_SPEED_THRESH:
                        r_contact_speed = -self.contact_speed_cost * (
                            speed - self.CONTACT_SPEED_THRESH)

        r_acc = 0.0
        if (self.acc_demand_cost or self.acc_margin_bonus) and self._acc_demand is not None \
                and self.joint_acc_limits is not None:
            lim = self.joint_acc_limits
            # LINIOWY i OGRANICZONY do 3x limitu. Wersja kwadratowa bez ograniczenia
            # dawala kare -274 na epizod (274x nagroda zadania), bo `_acc_demand` to
            # MAKSIMUM po podkrokach fizyki, a w podkrokach z kontaktem impuls
            # zderzenia daje ogromne chwilowe przyspieszenie. Czlon mierzyl wtedy
            # uderzenia w tablice, nie zadanie polityki. Ograniczenie przycina wplyw
            # impulsow, liniowosc nie wzmacnia ogona rozkladu.
            nad = np.clip((self._acc_demand - lim) / lim, 0.0, 3.0)
            zapas = np.maximum(lim - self._acc_demand, 0.0) / lim
            r_acc = (-self.acc_demand_cost * float(np.mean(nad))
                     + self.acc_margin_bonus * float(np.mean(zapas)))
        self._acc_demand = None     # zerowanie na kolejny krok sterowania

        r_approach = 0.0
        if self.approach_speed_cost:
            v_eef = np.asarray(self.sim.data.get_site_xvelp(self._eef_site_name),
                               dtype=np.float64)
            # PRAWDZIWA geometria (tgt_pos/eef_pos z site_xpos), NIE obserwacja:
            # `target_relative_pos` niesie bias percepcji, a nagroda nie moze
            # zalezec od szumu czujnika. Zmienne policzone wyzej w tej funkcji.
            kier = np.asarray(tgt_pos - eef_pos, dtype=np.float64)
            d = dist
            if d > 1e-6:
                # skladowa predkosci W STRONE celu (dodatnia = zbliza sie)
                v_do = float(np.dot(v_eef, kier / d))
                v_dop = d / max(self.approach_brake_time, 1e-3)
                nadmiar = v_do - v_dop
                if nadmiar > 0.0:
                    r_approach = -self.approach_speed_cost * nadmiar * nadmiar

        r_action = 0.0
        if action is not None:
            action = np.asarray(action, dtype=np.float64)
            r_action = -self.ACTION_COST * float(np.sum(np.square(action)))
            if self._prev_action is not None:
                delta = action - self._prev_action
                r_action -= self.action_rate_cost * float(np.sum(np.square(delta)))
            self._prev_action = action.copy()

        total = (r_shape + r_dense + r_depress + press_bonus + r_complete
                 + r_collision + r_action + r_finger_misuse + r_contact_speed
                 + r_approach + r_acc)

        return total * self.reward_scale / self.REWARD_NORM

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_success(self):
        return all(self._pressed)

    def _post_action(self, action):
        """Terminate the episode once all buttons are pressed (reward v2):
        the completion bonus is a terminal reward, no post-success loitering."""
        reward, done, info = super()._post_action(action)
        if self._check_success():
            done = True
        return reward, done, info

    def _check_spot_collision(self):
        """True, gdy jakikolwiek geom robota dotyka bryly Spota.

        Licznik OSOBNY od _check_board_collision: tablica i sciana to jedna
        przeszkoda zadaniowa, a korpus Spota to wlasna platforma. Mieszanie ich
        zepsulo by porownywalnosc metryki kolizji z biegami 6.1-6.41.
        """
        if not (self.spot_body or self.spot_front_cam):
            return False
        wlasne = ("spot_body_collision", "spot_frontcam_collision")
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1 = self.sim.model.geom_id2name(contact.geom1) or ""
            g2 = self.sim.model.geom_id2name(contact.geom2) or ""
            if g1 in wlasne or g2 in wlasne:
                other = g2 if g1 in wlasne else g1
                if other.startswith("robot0") or "gripper0" in other:
                    return True
        return False

    def _check_board_collision(self):
        """True if any robot geom touches the board plate OR the wall behind
        it (buttons excluded — their geoms carry no "board" in the name).

        The wall counts exactly like the plate: on hardware both are the same
        rigid obstacle, and hitting the wall next to the board is no safer
        than hitting the board. Before 2026-08-05 the wall was a visual-only
        geom and this check ignored it (docs 6.19)."""
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom1 = self.sim.model.geom_id2name(contact.geom1) or ""
            geom2 = self.sim.model.geom_id2name(contact.geom2) or ""
            robot_geom = any(g.startswith(("robot", "gripper"))
                             for g in (geom1, geom2))
            board_geom = any("board" in g or g == "wall_panel"
                             for g in (geom1, geom2))
            if robot_geom and board_geom:
                return True
        return False

    def _finger_on_cap(self, i):
        """True if any gripper finger geom touches button i's cap."""
        cap_id = self._cap_geom_ids[i]
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == cap_id and g2 in self._finger_geom_ids) or \
               (g2 == cap_id and g1 in self._finger_geom_ids):
                return True
        return False

    def _check_button_misuse(self):
        """True if an ARM link geom (robot0_*) touches any button — pressing
        with the wrist/arm risks the Gen3 wrist camera on hardware and
        invalidates TCP-based force/depth estimates. Gripper geoms are
        exempt (v2.3): knuckle/housing contact inevitably accompanies
        legitimate fingertip presses, and penalising it (v2.2) taught the
        policy to avoid buttons altogether."""
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            for r, b in ((g1, g2), (g2, g1)):
                if b in self._btn_geom_ids and r in self._arm_geom_ids:
                    return True
        return False

    def _finger_touched_buttons(self):
        """Set of button indices touched by ANY gripper finger geom this
        step. The reward (v2.5) penalizes contact with buttons that are
        neither the current target nor already pressed — genuine bumping
        on the way to the target, not post-press lingering contact."""
        touched = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in self._finger_geom_ids and b in self._geom_to_btn:
                    touched.add(self._geom_to_btn[b])
        return touched
