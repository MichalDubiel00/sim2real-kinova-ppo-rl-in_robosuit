"""Clearance of the lab saved poses (Web App degrees) to Spot, cameras and wall at stand-off -0.43 (docs 6.74)."""
import sys, os
sys.modules["tensorflow"] = None
sys.path.insert(0, os.getcwd())
import numpy as np, mujoco
import robosuite as suite
from envs.button_board import ButtonBoardEnv
POSES = {"rl_start": [0, 84.2, 231.6, 179.9, 244.3, 180], "Retract": [0, 340, 214, 0, 310, 90],
         "Home": [0, 15, 230, 0, 55, 90], "Packaging": [0, 330, 214, 0, 115, 270],
         "nuc": [0, 254.5, 212.2, 0, 298.8, 90], "Alien": [0, 260, 355, 0, 0, 90],
         "Crate": [0, 0, 212.2, 0, 7, 90], "Zero": [0, 0, 0, 0, 0, 0]}
for name, deg in POSES.items():
    q = [((d + 180) % 360 - 180) * np.pi / 180 for d in deg]
    ctrl = suite.load_composite_controller_config(controller="BASIC", robot="Kinova3_6DOF")
    env = ButtonBoardEnv(robots="Kinova3_6DOF", controller_configs=ctrl, has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False, base_x_offset=-0.43,
                         init_qpos=q, spot_body=True, spot_front_cam=True,
                         lab_cam_box=[0.06, 0.115, -0.1, 0.1, -0.03, 0.06],
                         gripper_types="Robotiq140Gripper", button_xml="button_lab.xml",
                         base_z_noise=0.0)
    env.reset()
    m, d = env.sim.model._model, env.sim.data._data
    d.qpos[env.robots[0]._ref_joint_pos_indexes] = q; mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    rob = [i for i, n in enumerate(names) if (n.startswith("robot0") or "gripper0" in n)
           and (m.geom_contype[i] or m.geom_conaffinity[i]) and n != "robot0_base_collision"]
    out = []
    for sn in ("spot_body_collision", "spot_frontcam_collision", "spot_labcam_collision", "wall_panel"):
        sid = names.index(sn); best = (9, "")
        for g in rob:
            dist = mujoco.mj_geomDistance(m, d, g, sid, 1.0, None)
            if dist != 0.0 and dist < best[0]: best = (dist, names[g])
        out.append(f"{sn.replace('_collision','').replace('spot_','')}={100*best[0]:5.1f}cm({best[1].replace('robot0_','').replace('_collision','')[:14]})")
    base = np.array([env._base_x, 0, env._base_z])
    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]] - base
    print(f"{name:9s} eef(base frame) {np.round(eef,3)}  " + "  ".join(out))
    env.close()
