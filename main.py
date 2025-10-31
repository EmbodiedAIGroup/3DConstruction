import time
import numpy as np
import open3d as o3d
from hardware.dog_control import Go2Controller  # 改为Go2Controller
from hardware.realsense import RealsenseCamera
from mapping.scene import Scene
from planning.nbp_planner import NBPPlanner
from config import Config

def main():
    # 初始化硬件（使用宇树Go2的网络接口）
    print("初始化硬件...")
    dog = Go2Controller(ifname="enx607d099f16d2")  # 替换为实际网卡名
    camera = RealsenseCamera(save_images=Config.SAVE_IMAGES)  # 启用图像保存
    scene = Scene(voxel_size=Config.MAP_RESOLUTION)
    planner = NBPPlanner(
        robot_radius=Config.ROBOT_RADIUS,
        map_resolution=Config.MAP_RESOLUTION
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
            if yaw > 0.1:
                action = 2  # 左转
            elif yaw < -0.1:
                action = 3  # 右转
            elif vx > 0.1:
                action = 1  # 前进
            else:
                action = 0  # 停止
            
            # 6. 执行动作
            dog.set_action(
                action=action,
                velocity=Config.MOVE_VELOCITY,
                duration=Config.MOVE_DURATION
            )

            input('Press Enter to continue...')
            
            # 状态输出与地图保存
            step += 1
            print(f"Step: {step}, 覆盖度: {coverage:.2f}, 动作: {action}")
            if step % 10 == 0:
                print(f"Step: {step}, 覆盖度: {coverage:.2f}, 动作: {action}")
            if step % Config.SAVE_INTERVAL == 0:
                scene.save_map(f"map_step_{step}.pcd")
            
            # 覆盖度达标退出
            if coverage >= Config.TARGET_COVERAGE:
                print("覆盖度达标，停止探索")
                break
            
            time.sleep(0.1)
        
        # 保存最终地图
        scene.save_map("final_map.pcd")
        print("探索完成，地图已保存")
    
    except KeyboardInterrupt:
        print("手动终止探索")
    finally:
        dog.stop()
        dog.disconnect()
        camera.release()

if __name__ == "__main__":
    main()