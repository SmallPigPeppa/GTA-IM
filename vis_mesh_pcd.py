"""
GTA-IM Dataset
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from open3d import (LineSet, PinholeCameraIntrinsic, Vector2iVector,
                    Vector3dVector, draw_geometries)

from gta_utils import LIMBS, read_depthmap

sys.path.append('./')

from typing import Tuple, Union, List
import pyrender
from pyrender import Node, DirectionalLight
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
def load_mesh(mesh_location: str) -> pyrender.Mesh:
    """Loads a mesh with applied material from the specified file."""

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    body_mesh = trimesh.load(mesh_location)
    mesh = pyrender.Mesh.from_trimesh(
        body_mesh,
        material=material)

    return mesh
def _create_raymond_lights():
    """Taken from pyrender source code"""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(Node(
            light=DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

def get_scene_render(body_mesh: pyrender.Mesh,
                     pcd,
                     image_width: int,
                     image_height: int,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Renders the scene and returns the color and depth output"""

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    # scene.add(body_mesh, 'mesh')

    pcd_translate = np.array([[0, 0, 0]]).T
    # camera_translate=np.array([[0,0,1000]]).T
    # pcd_rotate = get_rotation_matrix_from_xyz((0, 0, 0))
    pcd_rotate=np.identity(3)
    pcd_pose = np.hstack([pcd_rotate, pcd_translate])
    pcd_pose = np.vstack([pcd_pose, [0, 0, 0, 1]])
    tmp_points = np.array(pcd.points)
    # tmp_points[:,2]*=0.5
    # tmp_points *= 0.00063
    mesh_pcd = pyrender.Mesh.from_points(points=np.array(pcd.points), colors=np.array(pcd.colors), poses=pcd_pose)
    scene.add(mesh_pcd, 'mesh-pcd')

    light_nodes = _create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)
    pyrender.Viewer(scene)
    camera_rotate=np.identity(3)
    camera_translate = np.array([[0,0,0]]).T

    camera_pose = np.hstack([camera_rotate, camera_translate])
    camera_pose = np.vstack([camera_pose, [0, 0, 0, 1]])
    # camera_translation[0] *= -1.0
    # camera_pose = np.eye(4)
    # camera_pose[:3, 3] = camera_translation

    # camera_pose = np.eye(4)
    # # camera_pose = RT
    # camera_pose[1, :] = - camera_pose[1, :]
    # camera_pose[2, :] = - camera_pose[2, :]
    #
    # # camera_rotate = pcd.get_rotation_matrix_from_xyz((0, np.pi*1.0, np.pi*1.0))
    # # camera_translate = np.array([[0, 0, 0]]).T
    # # camera_rotate = pcd.get_rotation_matrix_from_xyz((0, np.pi*-0.7, np.pi*0.9))
    # # camera_translate = np.array([[-3.5, -0.3, 0.8]]).T
    # # camera_rotate = pcd.get_rotation_matrix_from_xyz((0, np.pi*-0.9, np.pi*1.0))
    # # camera_translate = np.array([[-1.5, -0.15, -0.5]]).T
    # # camera_rotate = pcd.get_rotation_matrix_from_xyz((np.pi*-0.3, np.pi*1.0, np.pi*1.0))
    # camera_rotate=np.identity(3)
    # camera_translate = np.array([[0,0,0]]).T
    #
    # camera_pose = np.hstack([camera_rotate, camera_translate])
    # camera_pose = np.vstack([camera_pose, [0, 0, 0, 1]])
    #
    #
    #
    # '''
    # 1.08137000e+03
    # 0.
    # 9.59500000e+02
    # 0.
    # 1.08137000e+03
    # 5.39500000e+02
    # 0.
    # 0.
    # 1.
    # focal_length_x = intrinsics_mat[0]
    # focal_length_y = intrinsics_mat[4]
    # center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    # '''
    #
    # # camera = pyrender.camera.IntrinsicsCamera(
    # #    fx=camera_focal_length, fy=camera_focal_length,
    # #    cx=camera_center[0], cy=camera_center[1]
    # # )
    # camera = pyrender.camera.IntrinsicsCamera(
    #     fx=1.08137000e+03, fy=1.08137000e+03,
    #     cx=9.59500000e+02, cy=5.39500000e+02, zfar=10e20
    # )
    # scene.add(camera, pose=camera_pose)
    # pyrender.Viewer(scene)
    # light_nodes = _create_raymond_lights()
    # for node in light_nodes:
    #     scene.add_node(node)
    #
    # # light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    # # scene.add(light, pose=camera_pose)
    #
    # r = pyrender.OffscreenRenderer(
    #     viewport_width=image_width,
    #     viewport_height=image_height,
    #     point_size=1.0
    # )
    # color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # plt.imshow(color)
    # plt.show()
    # return color, depth

def create_skeleton_viz_data(nskeletons, njoints):
    lines = []
    colors = []
    for i in range(nskeletons):
        cur_lines = np.asarray(LIMBS)
        cur_lines += i * njoints
        lines.append(cur_lines)

        single_color = np.zeros([njoints, 3])
        single_color[:] = [0.0, float(i) / nskeletons, 1.0]
        colors.append(single_color[1:])

    lines = np.concatenate(lines, axis=0)
    colors = np.asarray(colors).reshape(-1, 3)
    return lines, colors


def vis_skeleton_pcd(rec_idx, f_id, fusion_window=20):
    info = pickle.load(open(rec_idx + '/info_frames.pickle', 'rb'))
    info_npz = np.load(rec_idx + '/info_frames.npz')

    pcd = o3d.geometry.PointCloud()
    global_pcd = o3d.geometry.PointCloud()
    # use nearby RGBD frames to create the environment point cloud
    # for i in range(f_id - fusion_window // 2, f_id + fusion_window // 2, 10):
    for i in range(0, 1):
        fname = rec_idx + '/' + '{:05d}'.format(i) + '.png'
        if os.path.exists(fname):
            infot = info[i]
            cam_near_clip = infot['cam_near_clip']
            if 'cam_far_clip' in infot.keys():
                cam_far_clip = infot['cam_far_clip']
            else:
                cam_far_clip = 800. 
            depth = read_depthmap(fname, cam_near_clip, cam_far_clip)
            # delete points that are more than 20 meters away
            depth[depth > 20.0] = 0

            # obtain the human mask
            p = info_npz['joints_2d'][i, 0]
            fname = rec_idx + '/' + '{:05d}'.format(i) + '_id.png'
            id_map = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
            human_id = id_map[
                np.clip(int(p[1]), 0, 1079), np.clip(int(p[0]), 0, 1919)
            ]

            mask = id_map == human_id
            kernel = np.ones((3, 3), np.uint8)
            mask_dilation = cv2.dilate(
                mask.astype(np.uint8), kernel, iterations=1
            )
            depth = depth * (1 - mask_dilation[..., None])
            depth = o3d.geometry.Image(depth.astype(np.float32))
            # cv2.imshow('tt', mask.astype(np.uint8)*255)
            # cv2.waitKey(0)

            fname = rec_idx + '/' + '{:05d}'.format(i) + '.jpg'
            color_raw = o3d.io.read_image(fname)

            focal_length = info_npz['intrinsics'][f_id, 0, 0]
            rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
                color_raw,
                depth,
                depth_scale=1.0,
                depth_trunc=15.0,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.create_point_cloud_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    PinholeCameraIntrinsic(
                        1920, 1080, focal_length, focal_length, 960.0, 540.0
                    )
                ),
            )
            depth_pts = np.asarray(pcd.points)

            depth_pts_aug = np.hstack(
                [depth_pts, np.ones([depth_pts.shape[0], 1])]
            )
            cam_extr_ref = np.linalg.inv(info_npz['world2cam_trans'][i])
            depth_pts = depth_pts_aug.dot(cam_extr_ref)[:, :3]
            pcd.points = Vector3dVector(depth_pts)

            global_pcd.points.extend(pcd.points)
            global_pcd.colors.extend(pcd.colors)


    # body_mesh = load_mesh("000.obj_cam_CS.obj")
    mesh_dir='FPS-5-2020-06-11-10-06-48/meshes'
    body_mesh_name='._001'
    # body_mesh= load_mesh(f"{os.path.join(mesh_dir, body_mesh_name)}.obj")
    scene_rgba, _ = get_scene_render(
        body_mesh=None,
        pcd=global_pcd,
        image_width=1920,
        image_height=1080,
    )

    #
    # pcd_name='001'
    # plt.imshow(scene_rgba)
    # plt.gcf().canvas.manager.set_window_title(f"{pcd_name}")
    # plt.axis('off')
    # plt.show()
    #
    # # scene_rgba.save(os.path.join(args.output, f"{pcd_name}.png"))
    # im = Image.fromarray(scene_rgba)
    # im.save(os.path.join( f"{body_mesh_name}.png"))

    draw_geometries([global_pcd])



    # # read gt pose in world coordinate, visualize nearby frame as well
    # joints = info_npz['joints_3d_world'][(f_id - 30) : (f_id + 30) : 10]
    # tl, jn, _ = joints.shape
    # joints = joints.reshape(-1, 3)
    #
    # # create skeletons in open3d
    # nskeletons = tl
    # lines, colors = create_skeleton_viz_data(nskeletons, jn)
    # line_set = LineSet()
    # line_set.points = Vector3dVector(joints)
    # line_set.lines = Vector2iVector(lines)
    # line_set.colors = Vector3dVector(colors)
    #
    # vis_list = [global_pcd, line_set]
    # for j in range(joints.shape[0]):
    #     # spine joints
    #     if j % jn == 11 or j % jn == 12 or j % jn == 13:
    #         continue
    #     transformation = np.identity(4)
    #     transformation[:3, 3] = joints[j]
    #     # head joint
    #     if j % jn == 0:
    #         r = 0.07
    #     else:
    #         r = 0.03
    #
    #     sphere = o3d.geometry.create_mesh_sphere(radius=r)
    #     sphere.paint_uniform_color([0.0, float(j // jn) / nskeletons, 1.0])
    #     vis_list.append(sphere.transform(transformation))
    #
    # draw_geometries(vis_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-pa', '--path', default='demo')
    parser.add_argument(
        '-f', '--frame', default=100, type=int, help='frame to visualize'
    )
    parser.add_argument(
        '-fw',
        '--fusion-window',
        default=200,
        type=int,
        help='timesteps of RGB frames for fusing',
    )
    args = parser.parse_args()
    current_dir=os.getcwd()
    data_path=os.path.join(os.path.dirname(current_dir),'datasets','GTA_IM','FPS-5','2020-06-11-10-06-48')
    data_path='2020-06-11-10-06-48'
    # data_path='C:\Users\lwz\Desktop\code\datasets\GTA-IM\FPS-5\2020-06-11-10-06-48'
    # vis_skeleton_pcd(args.path + '/', args.frame, args.fusion_window)
    vis_skeleton_pcd(data_path+ '/', args.frame, args.fusion_window)

