import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import time
import os

# --- RealSense 配置 (沿用您的设置) ---
WIDTH = 640
HEIGHT = 480
FPS = 30
# 累积点云的容器
global_combined_pcd = o3d.geometry.PointCloud()
INITIAL_FRAME = True

# --- Open3D 注册和配准辅助函数 ---

def rs_frames_to_open3d(aligned_depth_frame, color_frame, pipeline_profile):
    """将 pyrealsense2 的帧转换为 Open3D 的 Image 和 PointCloud 对象"""
    
    # 1. 获取内参 (用于从深度图生成点云的关键)
    intrinsics = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    
    # 转换为 Open3D 的 PinholeCameraIntrinsic 对象
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics.width, height=intrinsics.height,
        fx=intrinsics.fx, fy=intrinsics.fy,
        cx=intrinsics.ppx, cy=intrinsics.ppy
    )
    
    # 2. 转换为 NumPy 数组
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # 3. 创建 Open3D 图像对象
    # RealSense 深度是 uint16，彩色是 BGR8 (需要转 RGB)
    o3d_depth = o3d.geometry.Image(depth_image)
    o3d_color = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    
    # 4. 创建 RGBD 图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, 
        depth_scale=1.0 / pipeline_profile.get_device().first_depth_sensor().get_depth_scale(),
        depth_trunc=3.0, # 截断深度，超过 3.0 米的点被忽略
        convert_rgb_to_intensity=False
    )
    
    # 5. 从 RGBD 和内参创建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, o3d_intrinsic
    )
    
    # 6. 对点云进行坐标变换（将 Z 轴指向前方，符合 Open3D 习惯）
    # Realsense 默认 Z+ 轴指向观察者，但我们希望 Z+ 轴远离观察者（Open3D/计算机图形学标准）
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd

def register_and_fuse_pcd(new_pcd, initial_pcd=None):
    """
    使用 Open3D 的配准功能将新点云与现有场景对齐。
    这里使用简化的 ICP 配准（适用于相邻帧移动较小的情况）。
    """
    global global_combined_pcd
    global INITIAL_FRAME

    if INITIAL_FRAME:
        # 第一帧，直接作为全局场景
        global_combined_pcd = new_pcd
        INITIAL_FRAME = False
        print("✅ 已初始化场景。")
        return

    # --- 1. 配准 (Alignment) ---
    print("...正在进行配准...")
    
    # 使用 Point-to-Plane ICP 算法进行粗略配准
    # 注意: 这个简化示例没有提供很好的初始变换矩阵，可能需要一个外部跟踪器 (如 T265) 来提供。
    # 假设：相邻帧之间的移动很小 (即您缓慢移动相机)
    
    # 估算配准参数
    voxel_size = 0.005 # 体素大小，用于下采样
    
    # 必须先计算法线，ICP才能使用 Point-to-Plane 目标函数
    new_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    global_combined_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    
    # 运行 ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=new_pcd, 
        target=global_combined_pcd, 
        max_correspondence_distance=0.02, # 最大对应距离 (2cm)
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    # 获取估计的变换矩阵
    transformation = reg_p2p.transformation
    print(f"配准得分 (越低越好): {reg_p2p.inlier_rmse:.4f}")

    # --- 2. 融合 (Fusion) ---
    
    # 应用变换到新点云
    new_pcd.transform(transformation)
    
    # 合并点云 (可以使用 Open3D 的体素下采样进行融合优化)
    global_combined_pcd += new_pcd
    
    # 3. 下采样和清理，以防止点云无限增大
    global_combined_pcd = global_combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"场景点数已更新至: {len(global_combined_pcd.points)} 点。")
    print("✅ 点云已配准并融合到场景中。")


def setup_realsense_pipeline():
    """初始化RealSense管线并返回所需对象"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, align, profile

# --- 主程序逻辑 ---

def reconstruction_main():
    pipeline, align, profile = setup_realsense_pipeline()
    global global_combined_pcd

    # 启动 Open3D 可视化器（在新线程中）
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D 3D Reconstruction Viewer', width=1000, height=700)
    
    # 添加一个用于显示累积点云的几何体
    vis.add_geometry(global_combined_pcd)
    
    # 额外的相机配置，用于 Open3D 转换
    profile_for_pcd = pipeline.get_active_profile()

    try:
        print("\n--- 相机已启动。按 'r' 键进行配准和融合，按 'q' 键退出 ---\n")
        
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue

            # 获取彩色图像用于 OpenCV 显示
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Color Stream (Press "r" to Register, "q" to quit)', color_image)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                # 1. 将 RealSense 帧转换为 Open3D 点云
                current_pcd = rs_frames_to_open3d(aligned_depth_frame, color_frame, profile_for_pcd)
                
                # 2. 配准并融合到全局场景
                register_and_fuse_pcd(current_pcd)
                
                # 3. 更新 Open3D 可视化器
                vis.update_geometry(global_combined_pcd)
                vis.poll_events()
                vis.update_renderer()
                
            if key == ord('q'):
                break

    except Exception as e:
        print(f"运行过程中发生错误: {e}")
    finally:
        # 清理
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        
        # 最终保存合并的点云
        if len(global_combined_pcd.points) > 0:
            final_filename = os.path.join("../save_results/realsense_reconstruction", "final_reconstruction.ply")
            os.makedirs(os.path.dirname(final_filename), exist_ok=True)
            o3d.io.write_point_cloud(final_filename, global_combined_pcd)
            print(f"\n✨ 最终 3D 模型已保存至: {final_filename} ✨")
        
        print("相机流已停止，程序退出。")

if __name__ == "__main__":
    reconstruction_main()