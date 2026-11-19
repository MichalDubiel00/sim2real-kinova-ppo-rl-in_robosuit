"""
Read-only check of the Gen3 speed limits through the Kortex API (no motion commands).

Prints the hard limits, the soft limits of every control mode (what the Web App shows) and,
with --watch, the active control mode and the measured tool speed while someone else moves
the arm (e.g. the lab Steam Deck). If the mode during a twist is CARTESIAN_JOYSTICK and the
speed plateaus at its soft limit, that limit applies to the bridge's twist commands.

Needs the kortex_api wheel (installed on the lab PC with kortex_ros2_python). Opens its own
session next to the running driver; run only with the lab's consent.

    python3 kortex_limits.py --ip 192.168.50.9
    python3 kortex_limits.py --ip 192.168.50.9 --watch 30
"""

import argparse
import math
import time

from kortex_api.RouterClient import RouterClient
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2, ControlConfig_pb2, Base_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient


def fmt_limits(lim):
    # field names differ between Kortex API versions: print the message as is
    return " ".join(str(lim).split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.50.9")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--watch", type=float, default=0.0, help="[s] poll mode and tool speed")
    ap.add_argument("--actions", action="store_true",
                    help="only list saved joint-angle actions (Home, nuc, ...) and exit")
    args = ap.parse_args()

    tcp, udp = TCPTransport(), UDPTransport()
    r_tcp = RouterClient(tcp, RouterClient.basicErrorCallback)
    r_udp = RouterClient(udp, RouterClient.basicErrorCallback)
    tcp.connect(args.ip, 10000)
    udp.connect(args.ip, 10001)
    info = Session_pb2.CreateSessionInfo()
    info.username, info.password = args.user, args.password
    info.session_inactivity_timeout = 60000
    info.connection_inactivity_timeout = 2000
    s_tcp, s_udp = SessionManager(r_tcp), SessionManager(r_udp)
    s_tcp.CreateSession(info)
    s_udp.CreateSession(info)
    try:
        if args.actions:
            base = BaseClient(r_tcp)
            req = Base_pb2.RequestedActionType()
            req.action_type = Base_pb2.REACH_JOINT_ANGLES
            for a in base.ReadAllActions(req).action_list:
                deg = [j.value for j in a.reach_joint_angles.joint_angles.joint_angles]
                print(f"{a.name:24s} deg [{', '.join(f'{v:.1f}' for v in deg)}]")
            return

        # limits and control mode live in the ControlConfig service, not Base
        cfg, cyclic = ControlConfigClient(r_tcp), BaseCyclicClient(r_udp)
        mode_name = ControlConfig_pb2.ControlMode.Name

        print("HARD:", fmt_limits(cfg.GetKinematicHardLimits()))
        for lim in cfg.GetAllKinematicSoftLimits().kinematic_limits_list:
            print("SOFT", fmt_limits(lim))

        if args.watch > 0:
            print("\nmode, |v| [m/s], |w| [deg/s]  (max so far)")
            t_end, v_max, w_max = time.time() + args.watch, 0.0, 0.0
            while time.time() < t_end:
                b = cyclic.RefreshFeedback().base
                v = math.sqrt(b.tool_twist_linear_x ** 2 + b.tool_twist_linear_y ** 2
                              + b.tool_twist_linear_z ** 2)
                w = math.sqrt(b.tool_twist_angular_x ** 2 + b.tool_twist_angular_y ** 2
                              + b.tool_twist_angular_z ** 2)
                v_max, w_max = max(v_max, v), max(w_max, w)
                mode = mode_name(cfg.GetControlMode().control_mode)
                print(f"{mode:28s} {v:.3f} ({v_max:.3f})  {w:6.1f} ({w_max:6.1f})", flush=True)
                time.sleep(0.2)
    finally:
        s_tcp.CloseSession()
        s_udp.CloseSession()
        tcp.disconnect()
        udp.disconnect()


if __name__ == "__main__":
    main()
