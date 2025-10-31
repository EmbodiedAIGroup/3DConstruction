import time
import numpy as np
import open3d as o3d
from hardware.dog_control import Go2Controller  # 改为Go2Controller
from hardware.realsense import RealsenseCamera
from mapping.scene import Scene
from planning.nbp_planner import NBPPlanner
from config import Config
from action import RobotAction

def main():
    # 初始化硬件（使用宇树Go2的网络接口）
    print("初始化硬件...")
    dog = Go2Controller(ifname="enx607d099f16d2")  # 替换为实际网卡名
    camera = RealsenseCamera(save_images=Config.SAVE_IMAGES)  # 启用图像保存
    scene = Scene(voxel_size=Config.MAP_RESOLUTION)
    planner = NBPPlanner(
        robot_radius=Config.ROBOT_RADIUS,
        # map_resolution=Config.MAP_RESOLUTION
    )
    
    try:
        print("开始探索...")
        step = 0
        while step < Config.MAX_EXPLORATION_STEPS:
            # 1. 获取传感器数据
            rgb_img, depth_meters = camera.get_frames()
            if rgb_img is None or depth_meters is None:
                time.sleep(0.1)
                continue
            
            # 2. 生成局部点云（相机坐标系）
            local_pcd = o3d.geometry.PointCloud()
            local_points = camera.depth_to_point_cloud(depth_meters)
            local_pcd.points = o3d.utility.Vector3dVector(local_points)
            
            # 3. 点云坐标转换（相机→机器人→世界）
            robot_pose = dog.get_pose()
            if robot_pose is None:
                continue  # 等待位姿数据
            
            # 相机到机器人的外参转换（根据实际安装位置调整）
            camera_to_robot = np.array(Config.CAMERA_EXTRINSICS)
            local_pcd.transform(camera_to_robot)
            
            # 机器人到世界坐标系（简化为机器人当前位置）
            robot_pos = robot_pose["position"]
            world_transform = np.eye(4)
            world_transform[:3, 3] = robot_pos
            local_pcd.transform(world_transform)
            
            # 4. 更新全局地图
            scene.add_point_cloud(local_pcd, np.eye(4))
            
            # 5. 更新碰撞地图并规划动作
            planner.update_collision_map(
                scene.voxel_down_pcd,
                robot_pose=dog.get_pose()  # 传入机器人当前位姿
            )
            coverage = scene.get_covered_ratio()
            
            # 转换规划器输出为宇树动作指令（1-前进,2-左转,3-右转）
            vx, vy, yaw = planner.plan_next_move(robot_pose, coverage)

            # 获取规划器中的障碍物信息（需在NBPPlanner中实现碰撞检测接口）
            is_forward_obstacle = planner.check_obstacle_ahead(robot_pose, distance=Config.OBSTACLE_CHECK_DISTANCE)
            is_left_obstacle = planner.check_obstacle_left(robot_pose, distance=Config.OBSTACLE_CHECK_DISTANCE)
            is_right_obstacle = planner.check_obstacle_right(robot_pose, distance=Config.OBSTACLE_CHECK_DISTANCE)

            # 优先选择无障碍物的方向
            if not is_forward_obstacle and vx > 0.1:
                action = RobotAction.MOVE_FORWARD  # 前进（无障碍物且有前进指令）
            elif not is_left_obstacle and yaw > 0.1:
                action = RobotAction.MOVE_LEFT  # 左转（无左侧障碍物且有左转指令）
            elif not is_right_obstacle and yaw < -0.1:
                action = RobotAction.MOVE_RIGHT  # 右转（无右侧障碍物且有右转指令）
            else:
                # 所有规划方向有障碍物时，优先尝试转向避开
                if not is_left_obstacle:
                    action = RobotAction.MOVE_LEFT  # 左转避开
                elif not is_right_obstacle:
                    action = RobotAction.MOVE_RIGHT  # 右转避开
                else:
                    action = RobotAction.STOP  # 无法移动，停止

            print('\n\n' + '#' * 20)
            print(f"动作: {action}")
            print('#' * 20 + '\n\n')

            # 6. 执行动作
            dog.set_action(
                action=action.value,
                velocity=Config.MOVE_VELOCITY,
                duration=Config.MOVE_DURATION
            )

            # input('Press Enter to continue...')
            time.sleep(1)
            
            # 状态输出与地图保存
            step += 1
            print(f"Step: {step}, 覆盖度: {coverage:.2f}, 动作: {action}")
            if step % Config.SAVE_INTERVAL == 0:
                scene.save_map(f"save_results/map_step_{step}.pcd")
            
            # 覆盖度达标退出
            if coverage >= Config.TARGET_COVERAGE:
                print("覆盖度达标，停止探索")
                break
            
            time.sleep(0.1)
        
        # 保存最终地图
        scene.save_map("save_results/final_map.pcd")
        print("探索完成，地图已保存")
    
    except KeyboardInterrupt:
        print("手动终止探索")
    finally:
        dog.stop()
        dog.disconnect()
        camera.release()

if __name__ == "__main__":
    main()