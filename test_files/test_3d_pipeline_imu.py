import pyrealsense2 as rs
import numpy as np
import torch
import open3d as o3d
import os
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R  # 用于旋转矩阵处理


class RealSensePointCloudAccumulator:
    """
    带IMU位姿跟踪的RealSense点云累加器
    """
    def __init__(self, save_dir="accumulated_pcs", device='cpu'):
        # 初始化全局点云
        self.full_pc = torch.empty((0, 3), device=device)  # 世界坐标系下的点云
        self.full_pc_colors = torch.empty((0, 3), device=device)
        self.device = device
        
        # 保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"{save_dir}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # RealSense配置（包含IMU）
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 启用深度、彩色和IMU流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)  # 加速度计
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)   # 陀螺仪
        self.align = rs.align(rs.stream.color)

        # 位姿跟踪相关变量
        self.prev_time = None  # 上一帧时间戳
        self.rotation = R.from_matrix(np.eye(3))  # 初始旋转（单位矩阵）
        self.translation = np.array([0.0, 0.0, 0.0])  # 初始平移（原点）
        self.gyro_bias = np.zeros(3)  # 陀螺仪偏置（简单校准）
        self.accel_bias = np.zeros(3)  # 加速度计偏置

    def start(self):
        """启动相机流并初始化IMU校准"""
        self.profile = self.pipeline.start(self.config)
        print("相机启动成功，开始IMU校准（保持相机静止3秒）...")
        
        # 简单校准：前3秒静止，计算IMU偏置
        calib_start = time.time()
        gyro_samples = []
        accel_samples = []
        
        while time.time() - calib_start < 3:
            frames = self.pipeline.wait_for_frames()
            gyro = frames.first_or_default(rs.stream.gyro)
            accel = frames.first_or_default(rs.stream.accel)
            if gyro and accel:
                gyro_data = gyro.as_motion_frame().get_motion_data()
                accel_data = accel.as_motion_frame().get_motion_data()
                gyro_samples.append([gyro_data.x, gyro_data.y, gyro_data.z])
                accel_samples.append([accel_data.x, accel_data.y, accel_data.z])

        # 转换为 NumPy 数组再计算均值
        if gyro_samples:
            self.gyro_bias = np.mean(np.array(gyro_samples), axis=0)
        else:
            self.gyro_bias = np.zeros(3)

        if accel_samples:
            self.accel_bias = np.mean(np.array(accel_samples), axis=0)
        else:
            self.accel_bias = np.zeros(3)
            
        gyro_str = np.array2string(self.gyro_bias, formatter={'float_kind': lambda x: f"{x:.4f}"})
        accel_str = np.array2string(self.accel_bias, formatter={'float_kind': lambda x: f"{x:.4f}"})
        print(f"IMU校准完成 | 陀螺仪偏置: {gyro_str} | 加速度计偏置: {accel_str}")

    def update_pose_from_imu(self, gyro_data, accel_data, dt):
        """
        根据IMU数据更新相机位姿（简化版）
        输入：陀螺仪数据(rad/s)、加速度计数据(m/s²)、时间间隔(s)
        """
        # 1. 去除偏置
        gyro = gyro_data - self.gyro_bias
        accel = accel_data - self.accel_bias

        # 2. 陀螺仪积分更新旋转（欧拉角近似）
        #  gyro是角速度 [x, y, z]，积分得到角度变化
        delta_rot = R.from_rotvec(gyro * dt)  # 旋转向量转旋转矩阵
        self.rotation = self.rotation * delta_rot  # 累积旋转（右乘）

        # 3. 加速度计积分更新平移（简化：假设重力已过滤，仅保留线性加速度）
        #  先将加速度从相机坐标系转换到世界坐标系
        accel_world = self.rotation.apply(accel)
        #  移除重力（假设z轴向上）
        accel_world[2] -= 9.81  # 减去重力加速度
        #  积分求速度和位移（简化的欧拉积分）
        self.translation += 0.5 * accel_world * (dt **2)  # 位移 = 0.5*a*t²

    def get_partial_point_cloud(self):
        """获取当前帧点云，并返回相机位姿"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        # 获取深度、彩色和IMU帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        accel_frame = frames.first_or_default(rs.stream.accel)
        
        if not all([depth_frame, color_frame, gyro_frame, accel_frame]):
            return None, None, None

        # 计算时间间隔（用于IMU积分）
        current_time = depth_frame.get_timestamp() / 1000.0  # 转换为秒
        dt = 0.0
        if self.prev_time is not None:
            dt = current_time - self.prev_time
        self.prev_time = current_time

        # 更新位姿（仅当时间间隔有效时）
        if dt > 0 and dt < 0.1:  # 过滤异常时间间隔
            gyro_data = np.asarray(gyro_frame.get_motion_data())
            accel_data = np.asarray(accel_frame.get_motion_data())
            self.update_pose_from_imu(gyro_data, accel_data, dt)

        # 生成当前帧点云（相机坐标系）
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(color_frame.get_data())
        
        coordinates = []
        colors = []
        for i in range(len(vtx)):
            x = vtx[i][0]
            y = vtx[i][1]
            z = vtx[i][2]
            if z > 0:  # 过滤无效点
                coordinates.append([x, y, z])
                # 颜色映射
                u = int(np.clip(x * intr.fx / z + intr.ppx, 0, intr.width - 1))
                v = int(np.clip(y * intr.fy / z + intr.ppy, 0, intr.height - 1))
                colors.append(tex[v, u] / 255.0)
        
        if not coordinates:
            return None, None, None
        
        # 转换为torch张量
        part_pc = torch.tensor(coordinates, dtype=torch.float32, device=self.device)
        part_pc_colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        
        # 返回点云、颜色和当前位姿（旋转矩阵+平移向量）
        return part_pc, part_pc_colors, (self.rotation.as_matrix(), self.translation)

    def transform_to_world(self, points, rotation, translation):
        """
        将相机坐标系下的点转换到世界坐标系
        公式：世界坐标 = R * 相机坐标 + t
        """
        # 旋转：R是3x3矩阵，点是Nx3，转换为 R @ points.T → 3xN，再转置为Nx3
        rotated = torch.matmul(torch.tensor(rotation, device=self.device, dtype=torch.float32), 
                              points.T).T
        # 平移：加上平移向量
        transformed = rotated + torch.tensor(translation, device=self.device, dtype=torch.float32)
        return transformed

    def accumulate(self):
        """累积世界坐标系下的点云"""
        part_pc, part_pc_colors, pose = self.get_partial_point_cloud()
        if part_pc is None or pose is None:
            return False
        
        rotation, translation = pose
        # 转换到世界坐标系
        part_pc_world = self.transform_to_world(part_pc, rotation, translation)
        
        # 累积到全局点云
        self.full_pc = torch.vstack((self.full_pc, part_pc_world))
        self.full_pc_colors = torch.vstack((self.full_pc_colors, part_pc_colors))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 帧已累积（世界坐标系）。当前总点数: {len(self.full_pc)}")
        return True

    def save(self, timestamp):
        """保存世界坐标系下的点云"""
        if len(self.full_pc) == 0:
             print("全局点云为空，跳过保存。")
             return
             
        file_name = f"accumulated_pc_{timestamp}.ply"
        save_path = os.path.join(self.save_dir, file_name)

        # 转换为Open3D格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.full_pc.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.full_pc_colors.cpu().numpy())

        # 保存
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"\n✅ 成功保存世界坐标系点云至: {save_path} (点数: {len(self.full_pc)})")

        # 保存numpy数据
        numpy_save_path = save_path.rsplit('.', 1)[0] + "_data.npy"
        numpy_data = {
            "points": self.full_pc.cpu().numpy(),
            "colors": self.full_pc_colors.cpu().numpy()
        }
        np.save(numpy_save_path, numpy_data)
        print(f"✅ 成功保存numpy数据至: {numpy_save_path}")

    def stop(self):
        self.pipeline.stop()


if __name__ == "__main__":
    ACCUMULATE_INTERVAL = 0.5  # 0.5秒累积一帧
    SAVE_INTERVAL = 2.0        # 2.0秒保存一次
    COLLECTION_DURATION = 60 * 5   # 采集总时长（秒）

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    accumulator = RealSensePointCloudAccumulator(
        save_dir="../save_results/realsense_scans",
        device=device
    )
    
    accumulator.start()
    
    last_accumulate_time = time.time()
    last_save_time = time.time()
    start_time = time.time()
    
    print(f"--- 采集将在 {COLLECTION_DURATION} 秒后自动停止 ---")

    while time.time() - start_time < COLLECTION_DURATION:
        current_time = time.time()

        # 累积点云
        if current_time - last_accumulate_time >= ACCUMULATE_INTERVAL:
            accumulator.accumulate()
            last_accumulate_time = current_time

        # 保存点云
        if current_time - last_save_time >= SAVE_INTERVAL:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            accumulator.save(timestamp)
            last_save_time = current_time
            
        time.sleep(0.01)