"""
GTA-IM Dataset
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import numpy.linalg
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
    for i in range(0, 955):
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
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw,
                depth,
                depth_scale=1.0,
                depth_trunc=15.0,
                convert_rgb_to_intensity=False,
            )
            pcd =o3d.geometry.PointCloud.create_from_rgbd_image(
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

    mesh_dir='FPS-5-2020-06-11-10-06-48/meshes'
    body_mesh_name='001'
    mesh_name=(f"{os.path.join(mesh_dir, body_mesh_name)}.obj")
    mesh=o3d.io.read_triangle_mesh(mesh_name)
    np.asarray(mesh.vertices)
    np.asarray(mesh.triangles)
    mesh.compute_vertex_normals()
    t=-1*info_npz['world2cam_trans'][0][3,:3]
    r=info_npz['world2cam_trans'][0][:3,:3].T
    rinv=numpy.linalg.inv(r)
    print(t)
    import copy
    mesh_t=copy.deepcopy(mesh).translate(t,relative=True)
    mesh_r=copy.deepcopy(mesh_t).rotate(rinv,center=(0,0,0))
    # # o3d.visualization.draw_geometries([global_pcd,mesh_world])
    # o3d.visualization.draw_geometries([mesh,mesh_r])
    # o3d.visualization.draw_geometries([mesh_r,global_pcd])
    vis=o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_r)
    vis.add_geometry(global_pcd)
    ctr=vis.get_view_control()
    # ctr.change_field_of_view(step=-15.0)
    ctr.rotate(45,45)
    vis.run()
    # image=vis.capture_screen_image(False)
    image=vis.capture_screen_float_buffer(False)
    plt.imsave(f'test.png',np.asarray(image),dpi=128)
    vis.destroy_window()



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

