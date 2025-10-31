class Config:
    # 探索参数
    MAX_EXPLORATION_STEPS = 500
    TARGET_COVERAGE = 0.8
    # TODO: debug 先每步保存一次地图
    SAVE_INTERVAL = 1
    SAVE_IMAGES = True  # 启用Realsense图像保存
    
    # 机器狗参数
    ROBOT_RADIUS = 1  # Go2机身半径
    MOVE_VELOCITY = 0.3  # 前进速度(m/s)
    MOVE_DURATION = 1  # 每次动作持续时间(s)
    
    # 地图参数
    MAP_RESOLUTION = 0.1  # 点云下采样分辨率(m)

    # 新增参数
    OBSTACLE_CHECK_DISTANCE = 0.2  # 障碍物检测距离（米）
    
    # 相机外参（相机→机器人基座坐标转换）
    # [x, y, z]偏移：相机在机器人前方0.2m，高度0.5m
    CAMERA_EXTRINSICS = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]