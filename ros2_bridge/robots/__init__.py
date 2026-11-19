"""
Robot backends. Each one supplies the robot's state and takes the grip-point twist:
  dry     compute obs and actions, send nothing (Webots topics)
  servo   Webots: MoveIt Servo
  ik      Webots: Jacobian steps streamed to the JTC
  kortex  hardware: ros2_kortex twist_controller
  labpy   hardware: lab driver kortex_ros2_python (TwistStamped, /goto_position); --no-send = dry
"""

BACKENDS = ("dry", "servo", "ik", "kortex", "labpy")


def make_robot(args, core):
    if args.backend == "dry":
        from robots.base import RobotBase
        return RobotBase(args)
    if args.backend == "servo":
        from robots.servo import ServoRobot
        return ServoRobot(args)
    if args.backend == "ik":
        from robots.ik import IkRobot
        return IkRobot(args, core)
    if args.backend == "kortex":
        from robots.kortex import KortexRobot
        return KortexRobot(args)
    if args.backend == "labpy":
        from robots.labpy import LabPyRobot
        return LabPyRobot(args, core)
    raise ValueError(args.backend)
