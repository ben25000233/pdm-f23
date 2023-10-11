import numpy as np
import open3d as o3d
import argparse
import os
import cv2
from matplotlib import pyplot as plt
import copy


def depth_image_to_point_cloud(rgb, depth):
    # TODO: Get point cloud from rgb and depth image 
    height, width = depth.shape
    rgb_image = o3d.geometry.Image(rgb)
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, depth_scale=1.0, convert_rgb_to_intensity = False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=512.0,  
        fy=512.0,
        cx=width / 2,
        cy=height / 2
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    points = np.asarray(pcd.points)*10
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


'''
    points = points[points[:, 2] < 1.5]
    print(points)
    

    colors = np.asarray(pcd.colors)
    colors = colors[colors[0:len(points)]]
    pcd.colors = o3d.utility.Vector3dVector(colors)
'''

    


#    print(np.unique(np.asarray(pcd.points)[:, 2]))
#    exit()
    # valid = points[:, 2] < upper_boud
    # points = points[valid]
    # colors = colors[valid]
    
    # print(np.unique(points[:, 2]))
    # o3d.visualization.draw_geometries([pcd])

    

def preprocess_point_cloud(pcd, voxel_size = 0.000005):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                                                                pcd_down,
                                                                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
                                                                )
    return pcd_down, pcd_fpfh



def execute_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size = 0.000005):
    distance_threshold = voxel_size * 0.5
    print("in")
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    trans = result.transformation
    return trans


def local_icp_algorithm(source_down, target_down, global_trans , voxel_size = 2/255*10):
    # TODO: Use Open3D ICP function to implement
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, global_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
   #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    trans = result.transformation
    return trans


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result

def reconstruct(args):
    # TODO: Return results
    pcd_list = []
    pred_cam_pos = []
    estimated_poses = []

    for i in range(len(rgb_file_list)):
        bgr_img = cv2.imread(rgb_file_list[i])
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file_list[i], -1)
        pcd = depth_image_to_point_cloud(rgb_img, depth)
        pcd_list.append(pcd)
    

    source_pcd = pcd_list[0]

    aligned_pcd_list = [source_pcd]  

    trans = np.identity(4)
    for i in range(1, 5):
        target = copy.deepcopy(pcd_list[i])
        source_down, source_fpth = preprocess_point_cloud(source_pcd)
        target_down, target_fpfh = preprocess_point_cloud(target)
         # Update the source point cloud for the next iteration
        source_pcd = copy.deepcopy(pcd_list[i])

        # global registration (optional)
        global_trans = execute_global_registration(
            target_down, source_down,  target_fpfh, source_fpth)
        # local ICP registration
        local_trans = local_icp_algorithm(
            target_down, source_down, global_trans,
        )

        trans = trans@global_trans
        target.transform(trans)
       

        aligned_pcd_list.append(target)
        
        #estimated_poses.append(np.asarray(target.transformation))
    print(len(aligned_pcd_list))

    #draw_registration_result(source_down, target_down, trans)
    return aligned_pcd_list, estimated_poses



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    ground_truth_poses = np.load("data_collection/first_floor/GT_pose.npy")
    #get image

    rgb_folder_path = os.path.join(os.getcwd(), args.data_root, 'rgb')
    depth_folder_path = os.path.join(os.getcwd(), args.data_root, 'depth')
    rgb_file_list = []
    depth_file_list = []
    for i in range(len(os.listdir(rgb_folder_path))):
        rgb_file_list.append(os.listdir(rgb_folder_path)[i])
        depth_file_list.append(os.listdir(depth_folder_path)[i])
    
    rgb_file_list = [os.path.join(args.data_root,'rgb/{}.png').format(i+1) for i in range(len(rgb_file_list))]
    depth_file_list = [os.path.join(args.data_root,'depth/{}.png').format(i+1) for i in range(len(depth_file_list))]
  
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pos = reconstruct(args)

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    
    l2_distances = [np.linalg.norm(gt - est) for gt, est in zip(ground_truth_poses, pred_cam_pos)]
    mean_l2_distance = np.mean(l2_distances)
    print("Mean L2 distance:", mean_l2_distance)

    '''
    # TODO: Visualize result
    
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    
        # Create estimated camera pose geometries (red lines)
    for pose in pred_cam_pos:
        estimated_pose_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        estimated_pose_geom.transform(pose)
        estimated_pose_geom.paint_uniform_color([1, 0, 0])  # Red
        o3d.visualization.Visualizer().add_geometry(estimated_pose_geom)

    # Create ground truth camera pose geometries (black lines)
    for pose in ground_truth_poses:
        ground_truth_pose_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ground_truth_pose_geom.transform(pose)
        ground_truth_pose_geom.paint_uniform_color([0, 0, 0])  # Black
        o3d.visualization.Visualizer().add_geometry(ground_truth_pose_geom)
    '''
    
    o3d.visualization.draw_geometries(result_pcd)
    
