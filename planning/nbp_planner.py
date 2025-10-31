import numpy as np
import heapq
import open3d as o3d  # 引入 open3d

class NBPPlanner:
    def __init__(self, robot_radius=0.3):
        """
        初始化NBP规划器
        :param robot_radius: 机器人的半径（米），用作安全距离
        """
        self.robot_radius = robot_radius
        # self.map_resolution = map_resolution # 不再需要栅格地图分辨率
        # self.collision_map = None # 不再使用栅格地图
        
        # 用于碰撞检测的KDTree
        self.kdtree = None 
        # 存储从3D点云中提取的2D障碍物点（z=0），用于构建KDTree
        self.obstacle_pcd_2d = o3d.geometry.PointCloud()
        # 安全距离的平方，用于高效比较
        self.safety_threshold_sq = self.robot_radius * self.robot_radius

    def update_collision_map(self, voxel_down_pcd, robot_pose=None):
        """
        根据(下采样)点云更新碰撞检测器（KDTree）
        :param voxel_down_pcd: Scene提供的下采样点云 (o3d.geometry.PointCloud)
        :param robot_pose: (可选) 此参数保留以兼容旧接口，但在此实现中不再需要
        """
        points_3d = np.asarray(voxel_down_pcd.points)
        
        if len(points_3d) == 0:
            # 没有任何点云数据
            self.kdtree = None
            self.obstacle_pcd_2d = o3d.geometry.PointCloud()
            return

        # 1. 过滤出与2D导航相关的障碍物点
        #    我们假设机器人主要在地面移动，关心 (0.1m < z < 0.5m) 范围内的障碍物
        #    (0.1m是为了忽略纯粹的地面点，0.5m是机器人车身可能碰到的高度)
        obstacle_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 0.5)
        obstacle_points_3d = points_3d[obstacle_mask]

        if len(obstacle_points_3d) == 0:
            # 过滤后没有障碍物
            self.kdtree = None
            self.obstacle_pcd_2d = o3d.geometry.PointCloud()
            return

        # 2. 将3D障碍物点转换为2D点（x, y），并设置 z=0
        #    因为我们的规划是在2D平面进行的
        obstacle_points_xy = obstacle_points_3d[:, :2]
        obstacle_points_2d_as_3d = np.hstack(
            (obstacle_points_xy, np.zeros((obstacle_points_xy.shape[0], 1)))
        )

        # 3. 更新2D障碍物点云并构建KDTree
        self.obstacle_pcd_2d.points = o3d.utility.Vector3dVector(obstacle_points_2d_as_3d)
        self.kdtree = o3d.geometry.KDTreeFlann(self.obstacle_pcd_2d)

    def plan_next_move(self, current_pose, coverage):
        """规划下一步移动指令"""
        if self.kdtree is None:
            # 如果没有障碍物地图，默认前进
            return (0.2, 0, 0) 
        
        # 简化版NBP逻辑：优先向覆盖度低的方向移动
        # 实际应用中可替换为原仓库的价值地图计算
        directions = [
            (0.2, 0, 0),    # 前进
            (0, 0.1, 0),    # 右移
            (0, -0.1, 0),   # 左移
            (0, 0, 0.3),    # 右转 (假设转向不会立即碰撞)
            (0, 0, -0.3),   # 左转 (假设转向不会立即碰撞)
            (-0.2, 0, 0),   # 后退
        ]
        
        # 选择第一个无碰撞的方向
        for vx, vy, yaw in directions:
            # 检查线速度移动是否会导致碰撞
            if vx != 0 or vy != 0:
                if not self._check_collision(current_pose, vx, vy):
                    return (vx, vy, yaw)
            else:
                # 如果只是转向，我们假设它是安全的
                return (vx, vy, yaw)
        
        return (0, 0, 0.3)  # 无路可走时(被包围)，强制转向

    def _is_point_in_obstacle(self, x, y):
        """
        辅助函数：使用KDTree检查点(x,y)是否与障碍物过于接近（碰撞）
        """
        if self.kdtree is None:
            return False  # 没有障碍物，肯定不碰撞

        # 1. 创建查询点 (z=0，因为KDTree是2D的)
        query_point = [x, y, 0]
        
        # 2. 搜索最近的1个邻居
        #    [k, idx, dist_sq] = kdtree.search_knn_vector_3d(query_point, 1)
        #    k: 找到的邻居数 (0 或 1)
        #    idx: 邻居的索引
        #    dist_sq: 到邻居距离的 *平方*
        [k, idx, dist_sq] = self.kdtree.search_knn_vector_3d(query_point, 1)
        
        if k == 0:
            return False  # KDTree不为空，但没找到点（理论上不应发生，除非KDTree为空）

        # 3. 检查距离
        #    如果最近的障碍物点距离 < 机器人半径，则视为碰撞
        return dist_sq[0] < self.safety_threshold_sq

    def _check_collision(self, current_pose, vx, vy):
        """
        检查移动(vx, vy)是否会导致碰撞
        """
        # 预测移动后的 2D 位置
        # 注意：这里简化了模型，只检查目标点。
        # 更鲁棒的检查会检查从当前点到目标点的路径
        x = current_pose["position"][0] + vx
        y = current_pose["position"][1] + vy
        
        # 检查目标点是否离障碍物太近
        return self._is_point_in_obstacle(x, y)
    
    def check_obstacle_ahead(self, robot_pose, distance=0.5):
        """
        检测机器人前方指定距离内是否有障碍物
        """
        if self.kdtree is None:
            return False  # 无碰撞地图时视为无障碍物
        
        current_x, current_y = robot_pose["position"][0], robot_pose["position"][1]
        # 假设 yaw 存储在 "orientation" 的 z 分量 (欧拉角)
        yaw = robot_pose["orientation"][2]  
        
        target_x = current_x + distance * np.cos(yaw)
        target_y = current_y + distance * np.sin(yaw)
        
        # 检查目标点
        return self._is_point_in_obstacle(target_x, target_y)

    def check_obstacle_left(self, robot_pose, distance=0.5):
        """检测机器人左侧指定距离内是否有障碍物"""
        if self.kdtree is None:
            return False
        
        current_x, current_y = robot_pose["position"][0], robot_pose["position"][1]
        yaw = robot_pose["orientation"][2]
        
        # 左侧方向为航向角+90度（π/2弧度）
        left_angle = yaw + np.pi / 2
        target_x = current_x + distance * np.cos(left_angle)
        target_y = current_y + distance * np.sin(left_angle)
        
        return self._is_point_in_obstacle(target_x, target_y)

    def check_obstacle_right(self, robot_pose, distance=0.5):
        """检测机器人右侧指定距离内是否有障碍物"""
        if self.kdtree is None:
            return False
        
        current_x, current_y = robot_pose["position"][0], robot_pose["position"][1]
        yaw = robot_pose["orientation"][2]
        
        # 右侧方向为航向角-90度（π/2弧度）
        right_angle = yaw - np.pi / 2
        target_x = current_x + distance * np.cos(right_angle)
        target_y = current_y + distance * np.sin(right_angle)
        
        return self._is_point_in_obstacle(target_x, target_y)
