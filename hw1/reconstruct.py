import numpy as np
import open3d as o3d
import argparse
import os
import copy
from tqdm import tqdm


def depth_image_to_point_cloud(rgb, depth):
    # TODO: Get point cloud from rgb and depth image 
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=1000, convert_rgb_to_intensity = False
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=512,
        height=512,
        fx=256.0,  
        fy=256.0,
        cx=512 / 2,
        cy=512 / 2
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd = no_ceiling(pcd)
    return pcd


    

def preprocess_point_cloud(pcd, voxel_size ):
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
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    global_trans = result.transformation
    return global_trans


def local_icp_algorithm(source_down, target_down, global_trans , voxel_size ):
    # TODO: Use Open3D ICP function to implement
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, global_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    trans = result.transformation
    return trans

from scipy.spatial import cKDTree
def my_local_icp_algorithm(source_down, target_down, global_trans, voxel_size):

    source = np.asarray(source_down.points)
    target = np.asarray(target_down.points)

    # Create KD-tree for the target point cloud
    tree = cKDTree(target)

    # Find nearest neighbors and distances from source to target
    distances, correspondences = tree.query(source, k=1)

    mean_distance = np.mean(distances)

    # Filter correspondences based on the mean distance
    cut_correspondences = [correspondences[i] for i in range(len(correspondences)) if distances[i] <= mean_distance/2]
    filtered_source = [source[i] for i in range(len(correspondences)) if distances[i] <= mean_distance/2]
    filtered_source = np.array(filtered_source)
    
    target_correspondences = target[cut_correspondences]
 
    H = np.dot(filtered_source.T, target_correspondences)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = np.mean(target_correspondences, axis=0) - np.dot(R, np.mean(filtered_source, axis=0))

    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    trans = np.dot(transformation, global_trans)
    return trans

def no_ceiling(pcd):
    ceiling_height = 0.00003
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    selected_indices = np.where(points[:, 1] < ceiling_height)[0]
    result_pcd_no_ceiling = o3d.geometry.PointCloud()
    result_pcd_no_ceiling.points = o3d.utility.Vector3dVector(points[selected_indices])
    result_pcd_no_ceiling.colors = o3d.utility.Vector3dVector(colors[selected_indices])
    return result_pcd_no_ceiling

def reconstruct(args,rgb_file_list, depth_file_list):
    # TODO: Return results

    pcd_list = []

    voxel_size = 0.000005
    for i in range(len(rgb_file_list)):
        bgr_img = o3d.io.read_image(rgb_file_list[i])
        #rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        depth = o3d.io.read_image(depth_file_list[i])
        pcd = depth_image_to_point_cloud(bgr_img, depth)
        pcd_list.append(pcd)
    
    source_pcd = pcd_list[0]
    no_ceiling_source_pcd = no_ceiling(source_pcd)
    aligned_pcd_list = [no_ceiling_source_pcd]  

    pred_cam_poses = []
    init_cam_pose = [0,0,0]
    pred_cam_poses.append(init_cam_pose)
    

    trans = np.identity(4)
    for i in tqdm(range(1, 220)):
        target = copy.deepcopy(pcd_list[i])
        source_down, source_fpth = preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        source_pcd = copy.deepcopy(pcd_list[i])

        # global registration (optional)
        global_trans = execute_global_registration(
            target_down, source_down,  target_fpfh, source_fpth, voxel_size)
        # local ICP registration
        if args.version == "my_icp":
            local_trans = my_local_icp_algorithm(
                target_down, source_down, global_trans, voxel_size
            )
        elif args.version == "open3d":
            local_trans = local_icp_algorithm(
                target_down, source_down, global_trans,voxel_size
            )
        else:
            local_trans = global_trans

        trans = trans@local_trans
        target.transform(trans)
        aligned_pcd_list.append(target)

        #predict the camera pose
        pre_cam_pos = np.array(trans[:3, 3])
        pred_cam_poses.append(pre_cam_pos)

 
    return aligned_pcd_list, pred_cam_poses



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=2)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
        ground_truth_poses = np.load("data_collection/first_floor/GT_pose.npy")
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
        ground_truth_poses = np.load("data_collection/second_floor/GT_pose.npy")

    
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
    result_pcd, pred_cam_pos = reconstruct(args, rgb_file_list, depth_file_list)
    

    camera_pos = []
    gt_camera_pos_value = []
    gt_lines = o3d.geometry.LineSet()
    gt_lines_points = []
    gt_lines_colors = []
    init_cam_pose = ground_truth_poses[0][0:3]
    for i in range(len(result_pcd)):
        temp = [x * 0.0001 for x in ground_truth_poses[i][0:3]]
        temp -= init_cam_pose/10000

        gt_camera_pos_value.append(np.array(temp))
        test = o3d.geometry.PointCloud()
        test_points = np.array([temp])  
        test.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))  
        test.points = o3d.utility.Vector3dVector(test_points)
        camera_pos.append(test)

        # Add the current camera position to the trajectory line
        gt_lines_points.append(temp)
        gt_lines_colors.append([0, 0, 0])  
    gt_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(gt_lines_points) - 1)])
    gt_lines.points = o3d.utility.Vector3dVector(gt_lines_points)
    gt_lines.colors = o3d.utility.Vector3dVector(gt_lines_colors)




    pre_cam_pos = []
    pre_lines = o3d.geometry.LineSet()
    pre_lines_points = []
    pre_lines_colors = []    
    for cam_pos in pred_cam_pos:

        pre_point = o3d.geometry.PointCloud()
        pre_points = np.array([cam_pos])  
        pre_point.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))  
        pre_point.points = o3d.utility.Vector3dVector(pre_points)
        pre_cam_pos.append(pre_point)
        # Add the current camera position to the trajectory line
        pre_lines_points.append(cam_pos)
        pre_lines_colors.append(np.array([1, 0, 0]))

    pre_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pre_lines_points) - 1)])
    pre_lines.points = o3d.utility.Vector3dVector(pre_lines_points)
    pre_lines.colors = o3d.utility.Vector3dVector(pre_lines_colors[:-1])



    # TODO: Calculate and print L2 distance
    #Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    l2_distances = np.mean([np.linalg.norm(gt - est) for gt, est in zip(gt_camera_pos_value, pred_cam_pos)])
    print("Mean L2 distance:", l2_distances)

    
    '''
    # TODO: Visualize result
    
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    
    '''

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(gt_lines)
    vis.add_geometry(pre_lines)
    for i in range(len(result_pcd)):
        vis.add_geometry(result_pcd[i])
#        vis.add_geometry(camera_pos[i])
#        vis.add_geometry(pre_cam_pos[i])

    vis.run()




