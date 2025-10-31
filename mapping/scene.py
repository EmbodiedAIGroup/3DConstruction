import numpy as np
import open3d as o3d


class Scene:
    def __init__(self, voxel_size=0.05, sensor_range=5.0):
        self.voxel_size = voxel_size
        self.sensor_range = sensor_range  # 传感器最大有效范围（米）
        self.global_pcd = o3d.geometry.PointCloud()  # 全局点云
        self.voxel_down_pcd = o3d.geometry.PointCloud()  # 下采样点云
        self.explored_voxels = set()  # 已探索的体素坐标（去重）
        self.robot_trajectory = []  # 机器人轨迹（用于估算探索范围）

    def add_point_cloud(self, pcd, pose):
        """添加新的点云并更新探索体素"""
        # 点云坐标转换（相机坐标系→世界坐标系）
        pcd.transform(pose)
        self.global_pcd += pcd

        # 下采样优化
        self.voxel_down_pcd = self.global_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 记录机器人轨迹（用于估算探索范围）
        self.robot_trajectory.append(pose[:3, 3])  # 只保留位置信息

        # 更新已探索体素（去重）
        self._update_explored_voxels(pcd)

    def _update_explored_voxels(self, pcd):
        """将点云转换为体素坐标并记录（去重）"""
        points = np.asarray(pcd.points)
        # 过滤传感器有效范围内的点
        distances = np.linalg.norm(points, axis=1)
        valid_points = points[distances <= self.sensor_range]

        # 将点坐标转换为体素网格坐标（整数）
        voxel_coords = (valid_points / self.voxel_size).astype(int)
        # 转换为元组便于存入集合（去重）
        voxel_coords = [tuple(coord) for coord in voxel_coords]
        self.explored_voxels.update(voxel_coords)

    def get_covered_ratio(self):
        """
        基于体素的覆盖度计算：
        覆盖度 = 已探索体素数量 / 总可探索体素数量
        总可探索体素数量由机器人轨迹包围的空间范围估算
        """
        if len(self.robot_trajectory) < 2 or len(self.explored_voxels) == 0:
            return 0.0  # 初始阶段或无数据时返回0

        # 1. 计算机器人轨迹包围的空间范围（x,y,z轴的 min/max）
        traj_points = np.array(self.robot_trajectory)
        min_traj = np.min(traj_points, axis=0) - self.sensor_range  # 扩展传感器范围
        max_traj = np.max(traj_points, axis=0) + self.sensor_range

        # 2. 计算该范围内的总可探索体素数量（按传感器高度范围限制z轴）
        # 假设有效感知高度为0.1~2.0米（适应机器狗视角）
        z_min, z_max = 0.1, 2.0
        # 限制z轴范围在传感器有效高度内
        min_traj[2] = max(min_traj[2], z_min)
        max_traj[2] = min(max_traj[2], z_max)

        # 计算各轴体素数量
        voxel_count_x = int((max_traj[0] - min_traj[0]) / self.voxel_size)
        voxel_count_y = int((max_traj[1] - min_traj[1]) / self.voxel_size)
        voxel_count_z = int((max_traj[2] - min_traj[2]) / self.voxel_size)

        total_voxels = voxel_count_x * voxel_count_y * voxel_count_z
        if total_voxels <= 0:
            return 0.0  # 避免除以零

        # 3. 计算已探索体素占比（限制最大1.0）
        covered_ratio = min(1.0, len(self.explored_voxels) / total_voxels)
        return covered_ratio

    # def save_map(self, path):
    #     """保存点云地图"""
    #     o3d.io.write_point_cloud(path, self.voxel_down_pcd)