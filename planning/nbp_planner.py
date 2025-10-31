import numpy as np
import heapq

class NBPPlanner:
    def __init__(self, robot_radius=0.3, map_resolution=0.1):
        self.robot_radius = robot_radius
        self.map_resolution = map_resolution
        self.collision_map = None

    def update_collision_map(self, pcd, robot_pose=None):
        """
        根据点云更新碰撞地图，兼容无障碍物场景
        :param pcd: 全局点云（世界坐标系）
        :param robot_pose: 机器人当前位姿（用于无障碍物时初始化地图范围）
        """
        points = np.asarray(pcd.points)
        if len(points) == 0:
            # 无任何点云数据时，清空碰撞地图
            self.collision_map = None
            return

        # 取地面附近点作为障碍物（z在0~0.5m）
        ground_mask = (points[:, 2] > 0) & (points[:, 2] < 0.5)
        ground_points = points[ground_mask, :2]  # 只取x,y坐标

        # 处理无障碍物的情况
        if len(ground_points) == 0:
            if robot_pose is None:
                # 若无机器人位姿，无法初始化地图，暂时清空
                self.collision_map = None
                return

            # 以机器人当前位置为中心，生成一个默认范围的空地图（无障碍物）
            robot_pos = np.array(robot_pose["position"][:2])  # 取x,y坐标
            default_radius = 5.0  # 默认地图范围（机器人周围5米）
            min_coords = robot_pos - default_radius
            max_coords = robot_pos + default_radius
            grid_size = ((max_coords - min_coords) / self.map_resolution).astype(int)

            self.collision_map = {
                "min": min_coords,
                "max": max_coords,
                "grid": np.zeros(grid_size, dtype=bool),  # 全False（无障碍物）
                "resolution": self.map_resolution
            }
            return

        # 有障碍物时，正常构建碰撞地图
        min_coords = np.min(ground_points, axis=0) - 1.0  # 扩展1米边界
        max_coords = np.max(ground_points, axis=0) + 1.0
        grid_size = ((max_coords - min_coords) / self.map_resolution).astype(int)

        # 处理网格尺寸为0的极端情况（所有点坐标相同）
        grid_size = np.maximum(grid_size, 1)  # 确保至少1x1网格

        self.collision_map = {
            "min": min_coords,
            "max": max_coords,
            "grid": np.zeros(grid_size, dtype=bool),
            "resolution": self.map_resolution
        }

        # 标记障碍物网格
        for (x, y) in ground_points:
            grid_x = int((x - min_coords[0]) / self.map_resolution)
            grid_y = int((y - min_coords[1]) / self.map_resolution)
            # 确保网格索引在有效范围内
            if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                self.collision_map["grid"][grid_x, grid_y] = True

    def plan_next_move(self, current_pose, coverage):
        """规划下一步移动指令"""
        if self.collision_map is None:
            return (0.2, 0, 0)  # 初始移动
        
        # 简化版NBP逻辑：优先向覆盖度低的方向移动
        # 实际应用中可替换为原仓库的价值地图计算
        directions = [
            (0.2, 0, 0),    # 前进
            (-0.2, 0, 0),   # 后退
            (0, 0.1, 0),    # 右移
            (0, -0.1, 0),   # 左移
            (0, 0, 0.3),    # 右转
            (0, 0, -0.3)    # 左转
        ]
        
        # 选择第一个无碰撞的方向
        for vx, vy, yaw in directions:
            if not self._check_collision(current_pose, vx, vy):
                return (vx, vy, yaw)
        
        return (0, 0, 0.3)  # 无路可走时转向
    
    def _is_point_in_obstacle(self, x, y):
        """辅助函数：检查点(x,y)是否在障碍物网格中"""
        min_coords = self.collision_map["min"]
        grid_x = int((x - min_coords[0]) / self.map_resolution)
        grid_y = int((y - min_coords[1]) / self.map_resolution)
        
        grid = self.collision_map["grid"]
        # 检查网格索引是否有效且对应障碍物
        if 0 <= grid_x < grid.shape[0] and 0 <= grid_y < grid.shape[1]:
            return self.collision_map["grid"][grid_x, grid_y]
        return False  # 超出地图范围视为无障碍物
    
    def _check_collision(self, current_pose, vx, vy):
        """检查移动是否碰撞"""
        if self.collision_map is None:
            return False
        
        # 预测移动后的位置
        x = current_pose["position"][0] + vx
        y = current_pose["position"][1] + vy
        
        # 检查是否在障碍物网格中
        min_coords = self.collision_map["min"]
        grid_x = int((x - min_coords[0]) / self.map_resolution)
        grid_y = int((y - min_coords[1]) / self.map_resolution)
        
        grid = self.collision_map["grid"]
        if 0 <= grid_x < grid.shape[0] and 0 <= grid_y < grid.shape[1]:
            return self.collision_map["grid"][grid_x, grid_y]
        
        return False  # 超出地图范围视为安全
    
    def check_obstacle_ahead(self, robot_pose, distance=0.5):
        """检测    检测机器人前方指定距离内是否有障碍物
        :param robot_pose: 机器人当前位姿
        :param distance: 检测距离（米）
        :return: 有障碍物返回True，否则返回False
        """
        if self.collision_map is None:
            return False  # 无碰撞地图时视为无障碍物
        
        # 计算机器人前进方向上的目标点（沿当前朝向移动distance距离）
        current_x, current_y = robot_pose["position"][0], robot_pose["position"][1]
        yaw = robot_pose["orientation"][2]  # 获取机器人航向角（绕z轴旋转）
        
        # 计算目标点坐标（极坐标转直角坐标）
        target_x = current_x + distance * np.cos(yaw)
        target_y = current_y + distance * np.sin(yaw)
        
        # 检查目标点是否在障碍物网格中
        return self._is_point_in_obstacle(target_x, target_y)

    def check_obstacle_left(self, robot_pose, distance=0.5):
        """检测机器人左侧指定距离内是否有障碍物"""
        if self.collision_map is None:
            return False
        
        current_x, current_y = robot_pose["position"][0], robot_pose["position"][1]
        yaw = robot_pose["orientation"][2]  # 航向角
        
        # 左侧方向为航向角+90度（π/2弧度）
        left_angle = yaw + np.pi / 2
        target_x = current_x + distance * np.cos(left_angle)
        target_y = current_y + distance * np.sin(left_angle)
        
        return self._is_point_in_obstacle(target_x, target_y)

    def check_obstacle_right(self, robot_pose, distance=0.5):
        """检测机器人右侧指定距离内是否有障碍物"""
        if self.collision_map is None:
            return False
        
        current_x, current_y = robot_pose["position"][0], robot_pose["position"][1]
        yaw = robot_pose["orientation"][2]  # 航向角
        
        # 右侧方向为航向角-90度（π/2弧度）
        right_angle = yaw - np.pi / 2
        target_x = current_x + distance * np.cos(right_angle)
        target_y = current_y + distance * np.sin(right_angle)
        
        return self._is_point_in_obstacle(target_x, target_y)

    