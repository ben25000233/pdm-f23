import numpy as np
import open3d as o3d
import argparse
import os
import cv2
from matplotlib import pyplot as plt

def depth_image_to_point_cloud(rgb, depth):
    # TODO: Get point cloud from rgb and depth image 
    height, width = depth.shape
    rgb_image = o3d.geometry.Image(rgb)
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, depth_scale=1000.0,
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
    #o3d.visualization.draw_geometries([pcd])
    return pcd


def preprocess_point_cloud(pcd, voxel_size = 0.009):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                                                                pcd_down,
                                                                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50)
                                                                )
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size = 0.009):
    distance_threshold = voxel_size * 1.25
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                                                                                    source_down, target_down, source_fpfh, target_fpfh, True,                                                                                        distance_threshold,
                                                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
                                                                                    [
                                                                                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                                                                                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                                                                                    ], 
                                                                                    o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
                                                                                    )
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
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
    threshold = 0.02
    pred_cam_pos = []
    for i in range(len(rgb_file_list)):
        bgr_img = cv2.imread(rgb_file_list[i])
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file_list[i], -1)
        pcd = depth_image_to_point_cloud(rgb_img, depth)
        pcd_list.append(pcd)
    

    source_pcd = pcd_list[0]

    aligned_pcd_list = [source_pcd]  # Store the aligned point clouds in this list

    for i in range(1, len(pcd_list)):
        target = pcd_list[i]
        source_down, source_fpth = preprocess_point_cloud(source_pcd)
        target_down, target_fpfh = preprocess_point_cloud(target)

        # Perform global registration (optional)
        global_registration = execute_global_registration(
            source_down, target_down, source_fpth, target_fpfh )
        
        # Perform local ICP registration
        trans = local_icp_algorithm(
            target_down, source_down, np.identity(4), threshold
        )
        target.transform(global_registration.transformation)
        # Apply the transformation to align source with reference
        target.transform(trans)

        # Update the reference point cloud for the next iteration
        reference_pcd = target

        # Append the aligned point cloud to the list
        aligned_pcd_list.append(target)

    return aligned_pcd_list, pred_cam_pos



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
    print("Mean L2 distance: ", )

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    o3d.visualization.draw_geometries(result_pcd)
