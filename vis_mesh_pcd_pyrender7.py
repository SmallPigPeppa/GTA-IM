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
# from open3d import (LineSet, PinholeCameraIntrinsic, Vector2iVector,
#                     Vector3dVector, draw_geometries)

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
    print(body_mesh)
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


def get_scene_render(info_npz, body_mesh,
                     pcd,
                     image_width: int,
                     image_height: int,
                     idx=0,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Renders the scene and returns the color and depth output"""

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    pcd_translate = np.array([[0, 0, 0]]).T
    pcd_rotate = np.identity(3)
    pcd_pose = np.hstack([pcd_rotate, pcd_translate])
    pcd_pose = np.vstack([pcd_pose, [0, 0, 0, 1]])
    mesh_pcd = pyrender.Mesh.from_points(points=np.array(pcd.points), colors=np.array(pcd.colors), poses=pcd_pose)
    scene.add(mesh_pcd, 'mesh-pcd', pose=pcd_pose)

    light_nodes = _create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    # add mesh
    mesh_dir = 'FPS-5-2020-06-11-10-06-48/meshes'
    mesh_dir='C:/Users/90532/Desktop/Datasets/GTA-IM/meshes'
    body_mesh_name = '{:03d}'.format(idx + 1)
    mesh_name = (f"{os.path.join(mesh_dir, body_mesh_name)}.obj")
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    trimesh_human = trimesh.load(mesh_name)
    mesh_human = pyrender.Mesh.from_trimesh(trimesh_human, material=material)

    t = -1 * np.array(info_npz['world2cam_trans'][idx][3, :3]).reshape([3, 1])
    r = np.array(info_npz['world2cam_trans'][idx][:3, :3]).T
    rinv = np.linalg.inv(r)
    t2 = np.matmul(rinv, t)
    mesh_pose = np.hstack([rinv, t2])


    mesh_pose = np.vstack([mesh_pose, [0, 0, 0, 1]])
    scene.add(mesh_human, 'mesh-human', pose=mesh_pose)

    # pyrender.Viewer(scene)

    world2cam = np.array(info_npz['world2cam_trans'][idx])
    cam2world=np.linalg.inv(world2cam)

    # t = -1 * np.array(info_npz['world2cam_trans'][0][3, :3]).reshape([3, 1])
    # r = np.array(info_npz['world2cam_trans'][0][:3, :3]).T
    # rinv = np.linalg.inv(r)
    # t2 = np.matmul(rinv, t)

    t3=t2+np.array([-4,65,-50]).reshape([3,1])

    t3=t2+np.array([3.0,8,0.2]).reshape([3,1])
    t3=t2+np.array([3.0,12,0.2]).reshape([3,1])
    # t3=t2+np.array([0,0,0]).reshape([3,1])
    # t3=t2+np.array([0,0,0]).reshape([3,1])

    cam_rotate = pcd.get_rotation_matrix_from_xyz((-1*np.pi*0.45, np.pi*0,-1*np.pi))
    cam_rotate = pcd.get_rotation_matrix_from_xyz((-1*np.pi*0.5, np.pi*0,-1*np.pi))
    camera_pose = np.hstack([cam_rotate, t3])
    camera_pose = np.vstack([camera_pose, [0, 0, 0, 1]])
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1.08137000e+03, fy=1.08137000e+03,
        cx=9.59500000e+02, cy=5.39500000e+02, zfar=10e20
    )
    scene.add(camera, pose=camera_pose)
    print(cam2world)
    # scene.add(camera, pose=np.vstack([cam2world[:3,:], [0, 0, 0, 1]]))
    #
    pyrender.Viewer(scene)
    # light_nodes = _create_raymond_lights()
    # for node in light_nodes:
    #     scene.add_node(node)
    #
    # # light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    # # scene.add(light, pose=camera_pose)
    #
    r = pyrender.OffscreenRenderer(
        viewport_width=image_width,
        viewport_height=image_height,
        point_size=1.0
    )
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # plt.imshow(color)
    # plt.show()
    im = Image.fromarray(color)
    im.save(os.path.join(args.outpath, f"{body_mesh_name}.png"))
    return color,depth

def read_pcd(id1,id2):
    global_pcd = o3d.geometry.PointCloud()
    rec_idx='2020-06-11-10-06-48/'
    rec_idx = 'C:/Users/90532/Desktop/Datasets/GTA-IM/FPS-5/2020-06-11-10-06-48'
    info = pickle.load(open(rec_idx + '/info_frames.pickle', 'rb'))
    info_npz = np.load(rec_idx + '/info_frames.npz')
    for i in range(id1, id2):
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

            focal_length = info_npz['intrinsics'][i, 0, 0]
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw,
                depth,
                depth_scale=1.0,
                depth_trunc=15.0,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsic(
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
            pcd.points = o3d.utility.Vector3dVector(depth_pts)

            global_pcd.points.extend(pcd.points)
            global_pcd.colors.extend(pcd.colors)

    return global_pcd

def vis_skeleton_pcd(rec_idx, f_id, fusion_window=1):
    info_npz = np.load(rec_idx + '/info_frames.npz')
    for i in range(0,1):
        global_pcd = read_pcd(id1=max(0,i-fusion_window), id2=min(i+fusion_window,955))
        scene_rgba, _ = get_scene_render(
            info_npz=info_npz,
            body_mesh=None,
            pcd=global_pcd,
            image_width=1920,
            image_height=1080,
            idx=i
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-pa', '--path', default='demo')
    parser.add_argument(
        '-f', '--frame', default=100, type=int, help='frame to visualize'
    )
    parser.add_argument(
        '-fw',
        '--fusion-window',
        default=10,
        type=int,
        help='timesteps of RGB frames for fusing',
    )
    parser.add_argument('--outpath', default='test-demo6')
    args = parser.parse_args()
    current_dir = os.getcwd()
    data_path = os.path.join(os.path.dirname(current_dir), 'datasets', 'GTA_IM', 'FPS-5', '2020-06-11-10-06-48')
    # data_path = '2020-06-11-10-06-48'
    data_path = 'C:/Users/90532/Desktop/Datasets/GTA-IM/FPS-5/2020-06-11-10-06-48'
    # data_path='C:\Users\lwz\Desktop\code\datasets\GTA-IM\FPS-5\2020-06-11-10-06-48'
    # vis_skeleton_pcd(args.path + '/', args.frame, args.fusion_window)
    vis_skeleton_pcd(data_path, args.frame, args.fusion_window)
