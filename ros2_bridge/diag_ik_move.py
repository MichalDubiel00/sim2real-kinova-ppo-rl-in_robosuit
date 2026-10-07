#!/usr/bin/env python3
"""
Warstwa A: jeden ruch przez /compute_ik. SZABLON DO WYPELNIENIA.

Cel: sprawdzic, czy lancuch
    poza TCP -> +delta -> IK -> katy stawow -> kontroler
faktycznie przesuwa ramie tam, gdzie kazano.

To skrypt JEDNORAZOWY (bez timerow/callbackow) - dlatego mozna tu bezpiecznie
uzyc spin_until_future_complete. W wersji z petla to sie zakleszczy przy
domyslnym jednowatkowym executorze; wtedy trzeba MultiThreadedExecutor
+ ReentrantCallbackGroup albo obsluga future asynchronicznie.

URUCHOMIENIE (bez condy! ROS wymaga systemowego Pythona 3.10):claude
    source /opt/ros/humble/setup.bash
    source ~/Batchelor/webots_ros2_spot/install/setup.bash
    python3 ros2_bridge/diag_ik_move.py

ODNIESIENIE - robosuite OSC_POSE przy 20 Hz, akcja 1.0:
    os X: 0.01126 m/krok, 90% po 12 krokach
    os Y: 0.00722 m/krok, 90% po  3 krokach
    os Z: 0.01317 m/krok, 90% po  2 krokach
    (output_max = 0.05 m, czyli OSC realizuje 14-27 % komendy)
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# --- konfiguracja ---------------------------------------------------------
BASE_FRAME = "gen3_base_link"      # ramka komend; = input_ref_frame "base" z OSC
EE_FRAME   = "end_effector_link"   # ee_frame_name z moveit_servo_config.yaml
GROUP      = "manipulator"
JOINTS     = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
CMD_TOPIC  = "/kinova_joint_trajectory_controller/joint_trajectory"

DELTA      = np.array([0.0, 0.0, 0.05])   # TODO: eksperymentuj (os, znak, wielkosc)
TIME_FROM_START = 0.05                     # TODO: 0.05 = 1 krok przy 20 Hz. Za krotko
                                           # -> szarpanie/odrzucenie; za dlugo -> wolniej
                                           # niz zaklada polityka. Sprawdz empirycznie.


class IkMove(Node):
    def __init__(self):
        super().__init__("diag_ik_move")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Klient tworzony RAZ. W gen3_rl_moveit_ik.py jest tworzony w kazdym
        # kroku - to wyciek zasobow przy 20 Hz.
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        self.pub = self.create_publisher(JointTrajectory, CMD_TOPIC, 10)
        self.gripper_orientation = None
        self.gripper_position = None

        

    # ---------------------------------------------------------------- TF --
    def read_tcp(self, timeout=5.0):
        """Zwroc (pozycja xyz jako np.array, kwaternion xyzw jako tuple).

        TODO 1: odczytaj transformacje BASE_FRAME -> EE_FRAME.
          - self.tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time())
          - TF NIE jest gotowy od razu po starcie wezla: spinuj w petli
            (rclpy.spin_once(self, timeout_sec=0.1)) i lap wyjatek, az sie uda
          - wzorzec: rl_bridge/gen3_rl_moveit_ik.py, metoda update_transforms()
        """
        start_time = time.time()
        
        # Loop until we hit the timeout
        while (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                ee_transform = self.tf_buffer.lookup_transform('gen3_base_link', 'end_effector_link', rclpy.time.Time())
                self.gripper_position = np.array([
                    ee_transform.transform.translation.x,
                    ee_transform.transform.translation.y,
                    ee_transform.transform.translation.z
                ])
                self.gripper_orientation = np.array([
                    ee_transform.transform.rotation.x,
                    ee_transform.transform.rotation.y,
                    ee_transform.transform.rotation.z,
                    ee_transform.transform.rotation.w
                ])
            except Exception as e:
                self.get_logger().error(str(e))
            else:
                break
        return (self.gripper_position,self.gripper_orientation)

    # ---------------------------------------------------------------- IK --
    def solve_ik(self, pos, quat):
        """Zwroc liste 6 katow albo None."""
        
        # 1. Przygotowanie requestu
        req = GetPositionIK.Request()
        
        # Wszystkie parametry IK wpadaja do zagniezdzonego obiektu ik_request
        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = EE_FRAME
        req.ik_request.timeout.sec = 1
        
        # 2. Budowanie celu (PoseStamped)
        target_pose = PoseStamped()
        target_pose.header.frame_id = BASE_FRAME
        
        # ROS 2 wymaga jawnego rozpisania parametrow geometrii, nie przyjmuje np.array
        target_pose.pose.position.x = float(pos[0])
        target_pose.pose.position.y = float(pos[1])
        target_pose.pose.position.z = float(pos[2])
        
        target_pose.pose.orientation.x = float(quat[0])
        target_pose.pose.orientation.y = float(quat[1])
        target_pose.pose.orientation.z = float(quat[2])
        target_pose.pose.orientation.w = float(quat[3])
        
        req.ik_request.pose_stamped = target_pose
        
        # 3. Wywołanie asynchroniczne z odpowiednim argumentem
        future = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        # Pobranie wyniku z future
        res = future.result()
        
        # 4. Sprawdzenie sukcesu ZANIM dotkniemy wyniku
        if res is None or res.error_code.val != 1:
            self.get_logger().warn(f"IK failed with error code: {res.error_code.val if res else 'Timeout'}")
            return None
            
        # 5. Filtrowanie PO NAZWIE (Ekstrakcja odpowiednich 6 stawow)
        js = res.solution.joint_state
        solved_joints = []
        
        try:

            for j_name in JOINTS:
                idx = js.name.index(j_name)
                solved_joints.append(js.position[idx])
            return solved_joints
            
        except ValueError as e:
            self.get_logger().error(f"Nie znaleziono wymaganego stawu w odpowiedzi IK: {e}")
            return None

        
    # --------------------------------------------------------- publikacja --
    def send(self, joints):
        """Opublikuj trajektorie z jednym punktem.

        TODO 4: zbuduj JointTrajectory i opublikuj.
          - msg.joint_names = JOINTS
          - JointTrajectoryPoint: positions = joints,
            time_from_start.sec / .nanosec z TIME_FROM_START
          - wzorzec: rl_bridge/gen3_rl_jointPos.py, send_joint_positions()
          - UWAGA: kontroler musi miec subskrypcje. Sprawdz
            self.pub.get_subscription_count() > 0, inaczej wiadomosc przepadnie
            bez sladu
        """
        traj_msg = JointTrajectory()

        traj_msg.joint_names = JOINTS
        joint_trajectory_point = JointTrajectoryPoint()
        joint_trajectory_point.positions = joints
        joint_trajectory_point.time_from_start.sec = int(TIME_FROM_START)
        joint_trajectory_point.time_from_start.nanosec = int((TIME_FROM_START % 1) * 1e9)

        traj_msg.points.append(joint_trajectory_point)
        if self.pub.get_subscription_count() > 0 :
            self.pub.publish(traj_msg)
    


def main():
    rclpy.init()
    node = IkMove()

    if not node.ik_cli.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("/compute_ik niedostepne — czy dziala move_group?")
        return

    pos0, quat0 = node.read_tcp()
    target = pos0 + DELTA
    node.get_logger().info(f"teraz  {np.round(pos0, 4)}")
    node.get_logger().info(f"cel    {np.round(target, 4)}  (delta {DELTA})")

    joints = node.solve_ik(target, quat0)          # orientacja BEZ ZMIAN
    if joints is None:
        node.get_logger().error("IK nie znalazlo rozwiazania")
        return
    node.get_logger().info(f"katy   {np.round(joints, 3)}")

    node.send(joints)

    # TODO 5: odczekaj i zmierz ponownie, zeby zobaczyc, ILE ramie faktycznie
    # przejechalo. Wypisz: zadana delta vs osiagnieta delta.
    # Rozbieznosc powie Ci, czy TIME_FROM_START jest dobrze dobrany i czy
    # kontroler nadaza. To jest wlasciwy WYNIK tego skryptu.
    start_time = time.time()

    while (time.time() - start_time) < 1:
        rclpy.spin_once(node, timeout_sec=0.5)
        pos1, _ = node.read_tcp()
        node.get_logger().info(f"po ruchu {np.round(pos1, 4)}   "
                               f"osiagnieta delta {np.round(pos1 - pos0, 4)}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
