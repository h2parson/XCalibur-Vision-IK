import roboticstoolbox as rtb
import swift
from math import pi
from spatialgeometry import Cylinder, Sphere
import spatialmath as sm
from roboticstoolbox import ET
import numpy as np
from spatialgeometry import Mesh
from time import sleep

RADIUS = 0.004  # 4mm in metres

# ─── Robot Definition ────────────────────────────────────────────────────────
robot = rtb.Robot(
    rtb.ETS([
        ET.tz(qlim=[0.01937, 0.265-0.02737 ]),

        ET.tx(0.02963),
        ET.Rx(-pi/2),
        ET.tz(-0.01724),
        ET.Rz(qlim=[-pi, pi]),

        ET.Rx(pi/2),
        ET.tx(0.0275),
        ET.tz(0.00649),
        ET.Rz(qlim=[-pi, pi]),

        ET.Rx(pi/2),
        ET.tz(0.04673),
        ET.Rz(qlim=[-pi, pi]),

        ET.tx(0.03074),
        ET.tz(qlim=[0, 0.115]),
    ]),
    name="XCalibur"
)

ORANGE = (1.0, 0.4, 0.1, 1.0)
BLUE   = (0.2, 0.6, 1.0, 1.0)
RED    = (1.0, 0.0, 0.0, 1.0)
GREEN  = (0.0, 1.0, 0.0, 1.0)

def cyl_x(length, frame=sm.SE3(), color=BLUE):
    pose = frame * sm.SE3.Tx(length / 2) * sm.SE3.Ry(pi / 2)
    return Cylinder(radius=RADIUS, length=abs(length), pose=pose.A, color=color)

def cyl_z(length, frame=sm.SE3(), color=BLUE):
    pose = frame * sm.SE3.Tz(length / 2)
    return Cylinder(radius=RADIUS, length=abs(length), pose=pose.A, color=color)

def sphere(frame=sm.SE3(), color=ORANGE):
    return Sphere(radius=RADIUS * 1.5, pose=frame.A, color=color)

def make_normal_arrow(r, n, length=0.05, color=RED):
    n = np.array(n) / np.linalg.norm(n)
    center = np.array(r) + n * length / 2
    z = np.array([0, 0, 1])
    axis = np.cross(z, n)
    if np.linalg.norm(axis) < 1e-6:
        T = sm.SE3(center)
    else:
        angle = np.arccos(np.clip(np.dot(z, n), -1, 1))
        axis = axis / np.linalg.norm(axis)
        T = sm.SE3(center) * sm.SE3.AngleAxis(angle, axis)
    return Cylinder(radius=RADIUS, length=length, pose=T.A, color=color)

def _normal_arrow_pose(r, n, length=0.05):
    n = np.array(n) / np.linalg.norm(n)
    center = np.array(r) + n * length / 2
    z = np.array([0, 0, 1])
    axis = np.cross(z, n)
    if np.linalg.norm(axis) < 1e-6:
        return sm.SE3(center)
    else:
        angle = np.arccos(np.clip(np.dot(z, n), -1, 1))
        axis = axis / np.linalg.norm(axis)
        return sm.SE3(center) * sm.SE3.AngleAxis(angle, axis)

def build_shapes(q, r, robot):
    O = sm.SE3()

    T0      = sm.SE3(robot.fkine(q, end=robot.links[0]).A)
    T0_tx   = T0 * sm.SE3.Tx(0.02963)
    T0_rx   = T0_tx * sm.SE3.Rx(-pi/2)

    T1      = sm.SE3(robot.fkine(q, end=robot.links[1]).A)
    T1_rx   = T1 * sm.SE3.Rx(pi/2)
    T1_tx   = T1_rx * sm.SE3.Tx(0.0275)

    T2      = sm.SE3(robot.fkine(q, end=robot.links[2]).A)
    T2_rx   = T2 * sm.SE3.Rx(pi/2)

    T3      = sm.SE3(robot.fkine(q, end=robot.links[3]).A)
    T3_tx   = T3 * sm.SE3.Tx(0.03074)
    T3_roll = T3 * sm.SE3.Rz(-pi/2) * sm.SE3.Ty(-0.02) * sm.SE3.Tx(-0.029) * sm.SE3.Tz(-0.008)

    T4      = sm.SE3(robot.fkine(q, end=robot.links[4]).A)

    Tr = O * sm.SE3.Tx(r[0]) * sm.SE3.Ty(r[1]) * sm.SE3.Tz(r[2])

    return [
        sphere(O),                                                      # 0
        cyl_z(q[0], O),                                                 # 1
        sphere(T0),                                                     # 2
        cyl_x(0.02963, T0),                                             # 3
        sphere(T0_tx, color=BLUE),                                      # 4
        cyl_z(-0.01724, T0_rx),                                         # 5
        sphere(T1),                                                     # 6
        cyl_x(0.0275, T1_rx),                                           # 7
        cyl_z(0.00649, T1_tx),                                          # 8
        sphere(T2),                                                     # 9
        cyl_z(0.04673, T2_rx),                                          # 10
        sphere(T3),                                                     # 11
        Mesh(                                                           # 12
            filename=r"C:\Vision\CAD\ROLL_MECHANISM.STL",
            scale=(0.001, 0.001, 0.001),
            pose=(T3_roll).A,
            color=(0.2, 0.6, 1.0, 1.0)
        ),
        cyl_x(0.03074, T3),                                             # 13
        sphere(T3_tx, color=BLUE),                                      # 14
        cyl_z(q[4], T3_tx),                                             # 15
        sphere(T4),                                                     # 16
        sphere(Tr, color=RED),                                          # 17
    ]

def update_shapes(shapes, q, r):
    O = sm.SE3()

    T0      = sm.SE3(robot.fkine(q, end=robot.links[0]).A)
    T0_tx   = T0 * sm.SE3.Tx(0.02963)
    T0_rx   = T0_tx * sm.SE3.Rx(-pi/2)

    T1      = sm.SE3(robot.fkine(q, end=robot.links[1]).A)
    T1_rx   = T1 * sm.SE3.Rx(pi/2)
    T1_tx   = T1_rx * sm.SE3.Tx(0.0275)

    T2      = sm.SE3(robot.fkine(q, end=robot.links[2]).A)
    T2_rx   = T2 * sm.SE3.Rx(pi/2)

    T3      = sm.SE3(robot.fkine(q, end=robot.links[3]).A)
    T3_tx   = T3 * sm.SE3.Tx(0.03074)
    T3_roll = T3 * sm.SE3.Rz(-pi/2) * sm.SE3.Ty(-0.02) * sm.SE3.Tx(-0.029) * sm.SE3.Tz(-0.008)

    T4      = sm.SE3(robot.fkine(q, end=robot.links[4]).A)

    Tr = O * sm.SE3.Tx(r[0]) * sm.SE3.Ty(r[1]) * sm.SE3.Tz(r[2])

    new_poses = [
        O,                                                          # 0  sphere(O)
        O * sm.SE3.Tz(q[0] / 2),                                   # 1  cyl_z(q[0], O)
        T0,                                                         # 2  sphere(T0)
        T0 * sm.SE3.Tx(0.02963 / 2) * sm.SE3.Ry(pi / 2),          # 3  cyl_x(0.02963, T0)
        T0_tx,                                                      # 4  sphere(T0_tx)
        T0_rx * sm.SE3.Tz(-0.01724 / 2),                           # 5  cyl_z(-0.01724, T0_rx)
        T1,                                                         # 6  sphere(T1)
        T1_rx * sm.SE3.Tx(0.0275 / 2) * sm.SE3.Ry(pi / 2),        # 7  cyl_x(0.0275, T1_rx)
        T1_tx * sm.SE3.Tz(0.00649 / 2),                            # 8  cyl_z(0.00649, T1_tx)
        T2,                                                         # 9  sphere(T2)
        T2_rx * sm.SE3.Tz(0.04673 / 2),                            # 10 cyl_z(0.04673, T2_rx)
        T3,                                                         # 11 sphere(T3)
        T3_roll,                                                    # 12 Mesh roll_mechanism
        T3 * sm.SE3.Tx(0.03074 / 2) * sm.SE3.Ry(pi / 2),          # 13 cyl_x(0.03074, T3)
        T3_tx,                                                      # 14 sphere(T3_tx)
        T3_tx * sm.SE3.Tz(q[4] / 2),                               # 15 cyl_z(q[4], T3_tx)
        T4,                                                         # 16 sphere(T4)
        Tr,                                                         # 17 sphere(Tr) red target
    ]

    for shape, pose in zip(shapes, new_poses):
        shape.T = pose.A

def mm_to_m_vec(v):
    return [v[i] / 1000.0 for i in range(3)]

def convert_q(q):
    result = q.copy()
    result[0] = result[0] / 1000
    result[4] = result[4] / 1000
    return result

def skew(v):
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ])

def autozoom(env, robot, q, scale=1.5):
    T_ee  = robot.fkine(q)
    ee_pos = T_ee.t
    dist  = float(np.linalg.norm(ee_pos)) * scale
    cam_pos = [dist, dist, dist]
    try:
        env.set_camera_pose(cam_pos, [0, 0, 0])
    except AttributeError:
        print(f"Note: set_camera_pose not available. Manual zoom distance: {dist:.3f}m")

def joint_v(robot, r, n, q, lam=0.5, mu=1e-3):
    n = n / np.linalg.norm(n)
    q = q.copy()

    T      = robot.fkine(q)
    p      = T.t
    R      = T.R
    x_axis = R[:, 0]

    pos_error    = p - r
    orient_error = np.cross(x_axis, n)
    e = np.hstack((pos_error, orient_error))

    J      = robot.jacob0(q)
    Jv     = J[0:3, :]
    Jw     = J[3:6, :]
    Jo     = skew(n) @ skew(x_axis) @ Jw
    J_task = np.vstack((Jv, Jo))

    JJt    = J_task @ J_task.T
    J_pinv = J_task.T @ np.linalg.inv(JJt + mu**2 * np.eye(6))
    qd     = -lam * J_pinv @ e

    return qd, e

def go_dest(robot, q_dest, r, steps=50, dt=0.01):
    q_start = robot.q.copy()
    q_dest  = np.array(q_dest)
    for i in range(steps + 1):
        alpha   = i / steps
        robot.q = q_start + alpha * (q_dest - q_start)
        update_shapes(shapes, robot.q, r)
        env.step(dt)

# knife = Mesh(
#     filename=r"C:\Vision\CAD\8IN_GERMAN_CHEF_KNIFE.STL",
#     scale=(0.001, 0.001, 0.001),
#     pose=(sm.SE3.Ty(0.15) * sm.SE3.Tx(0.085) * sm.SE3.Tz(0.121) * sm.SE3.Rz(-pi/2)).A,
#     color=(1.0, 1.0, 0.0, 1.0)
# )

knife = Mesh(
    filename=r"C:\Vision\CAD\4IN PAIRING KNIFEChungus.STL",
    scale=(0.001, 0.001, 0.001),
    pose=(sm.SE3.Ty(0.04461) * sm.SE3.Tx(0.086) * sm.SE3.Tz(0.123) * sm.SE3.Rz(-pi/2)).A,
    color=(1.0, 1.0, 0.0, 1.0)
)

if __name__ == "__main__":

    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)
    env.add(knife)

    q0 = [0, 0, pi/2, pi/2, 0]
    robot.q = q0

    data        = np.load("knife_data.npz")
    tip_q1      = data['tip_q1']
    tip_q2      = data['tip_q2']
    yaw_indices = data['yaw_indices']
    ratios1     = data['ratios1']
    ratios2     = data['ratios2']

    print(robot.fkine(tip_q1).t)

    q0 = tip_q2
    r  = robot.fkine(q0).t

    shapes = build_shapes(q0, r, robot)
    for s in shapes:
        env.add(s)

    def apply_ratio(yaw_v, ratio_slice):
        velocity = yaw_v * ratio_slice
        velocity = [*velocity[:2], yaw_v, *velocity[2:]]
        return np.array(velocity)

    yaw_v = -pi / 4
    dt    = 0.001

    while True:
        robot.q = q0
        yaw_idx = 0

        while robot.q[2] > yaw_indices[-1]:
            if robot.q[2] <= yaw_indices[yaw_idx + 1]:
                yaw_idx += 1

            velocity = apply_ratio(yaw_v, ratios2[yaw_idx])
            robot.q  = robot.q + dt * velocity

            pose = robot.fkine(robot.q)
            update_shapes(shapes, robot.q, pose.t)
            env.step(dt)

        sleep(0.5)

    env.hold()