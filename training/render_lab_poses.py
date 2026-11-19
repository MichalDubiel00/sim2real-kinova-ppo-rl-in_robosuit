"""Side-view render of the lab saved poses next to the RL start pose (docs 6.74). Usage: MUJOCO_GL=egl python training/render_lab_poses.py out.png"""
import sys, os
sys.modules["tensorflow"] = None
sys.path.insert(0, os.getcwd())
import numpy as np, mujoco
import robosuite as suite
from PIL import Image, ImageDraw
from envs.button_board import ButtonBoardEnv
P = {"Home [0,15,230,0,55,90]": [0, 15, 230, 0, 55, 90],
     "start RL [0,84.2,231.6,179.9,244.3,180]": [0, 84.2, 231.6, 179.9, 244.3, 180],
     "spoczynek (12.11)": [0, 254.4, 204.0, 358.7, 314.5, 90.3],
     "nuc [0,254.5,212.2,0,298.8,90]": [0, 254.5, 212.2, 0, 298.8, 90]}
w = lambda deg: [((d + 180) % 360 - 180) * np.pi / 180 for d in deg]
ctrl = suite.load_composite_controller_config(controller="BASIC", robot="Kinova3_6DOF")
env = ButtonBoardEnv(robots="Kinova3_6DOF", controller_configs=ctrl, has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=False, base_x_offset=-0.43,
                     spot_body=True, spot_front_cam=True, lab_cam_box=[0.06, 0.115, -0.1, 0.1, -0.03, 0.06],
                     gripper_types="Robotiq140Gripper", button_xml="button_lab.xml", base_z_noise=0.0)
env.reset()
m, d = env.sim.model._model, env.sim.data._data
# side view camera, perpendicular to the wall
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [env._base_x + 0.35, 0, env._base_z + 0.2]; cam.distance = 2.0; cam.azimuth = 90; cam.elevation = -10
tiles = []
for name, deg in P.items():
    d.qpos[env.robots[0]._ref_joint_pos_indexes] = w(deg); mujoco.mj_forward(m, d)
    r.update_scene(d, cam); img = Image.fromarray(r.render())
    ImageDraw.Draw(img).text((10, 10), name, fill=(0, 0, 0))
    tiles.append(img)
out = Image.new("RGB", (1280, 960))
for i, t in enumerate(tiles): out.paste(t, ((i % 2) * 640, (i // 2) * 480))
out.save(sys.argv[1]); print("saved")
