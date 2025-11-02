import pyrealsense2 as rs
import numpy as np
import torch
import open3d as o3d
import os
import time  # 引入 time 模块用于时间控制
from datetime import datetime

class RealSensePointCloudAccumulator:
    """
    RealSense 点云累加器：从 RealSense 相机采集数据，转换为带颜色信息的点云，并累积。
    """
    def __init__(self, save_dir="accumulated_pcs", device='cpu'):
        # 初始化全局点云
        self.full_pc = torch.empty((0, 3), device=device)  # 全局点云坐标
        self.full_pc_colors = torch.empty((0, 3), device=device)  # 全局点云颜色
        self.device = device
        
        # 设置保存目录
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # RealSense 相机初始化和配置 (与原代码保持一致)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.align = rs.align(rs.stream.color)

    def start(self):
        """启动相机流"""
        self.profile = self.pipeline.start(self.config)
        print("相机启动成功，开始自动采集和保存...")

    def get_partial_point_cloud(self):
        """获取当前帧的局部点云（与原代码保持一致）"""
        # 等待一帧数据
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # 获取内参
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # 转换为点云
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(color_frame.get_data())
        
        # 提取坐标和颜色（过滤无效点，手动颜色查找）
        coordinates = []
        colors = []
        for i in range(len(vtx)):
            x = vtx[i].x
            y = vtx[i].y
            z = vtx[i].z
            if z > 0:  # 过滤深度为0的无效点
                coordinates.append([x, y, z])
                # 颜色查找（uv坐标转换）
                u = int(np.clip(x * intr.fx / z + intr.ppx, 0, intr.width - 1))
                v = int(np.clip(y * intr.fy / z + intr.ppy, 0, intr.height - 1))
                colors.append(tex[v, u] / 255.0)  # 归一化到[0,1]
        
        if not coordinates:
            return None, None
        
        # 转换为torch张量
        part_pc = torch.tensor(coordinates, dtype=torch.float32, device=self.device)
        part_pc_colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        
        return part_pc, part_pc_colors

    def accumulate(self):
        """累积局部点云到全局点云"""
        part_pc, part_pc_colors = self.get_partial_point_cloud()
        if part_pc is not None and part_pc_colors is not None:
            # 合并到全局点云
            self.full_pc = torch.vstack((self.full_pc, part_pc))
            self.full_pc_colors = torch.vstack((self.full_pc_colors, part_pc_colors))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 帧已累积。当前总点数: {len(self.full_pc)}")
            return True
        return False

    def save(self, timestamp):
        """保存全局点云为PLY文件，文件名包含时间戳"""
        if len(self.full_pc) == 0:
             print("全局点云为空，跳过保存。")
             return
             
        # 构建文件名
        file_name = f"accumulated_pc_{timestamp}.ply"
        save_path = os.path.join(self.save_dir, file_name)
        
        # 转换为Open3D格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.full_pc.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.full_pc_colors.cpu().numpy())
        
        # 保存文件
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"\n✅ 成功保存点云至: {save_path} (点数: {len(self.full_pc)})")
        
        # 清空全局点云，为下一轮累积做准备
        self.full_pc = torch.empty((0, 3), device=self.device)
        self.full_pc_colors = torch.empty((0, 3), device=self.device)
        print("🚀 已清空全局点云，开始下一轮 2 秒累积。\n")
        
        # O3D 可视化：移除，因为在自动循环中频繁弹出窗口会干扰操作，如果需要请手动调用。
        # o3d.visualization.draw_geometries([pcd], window_name="累积点云结果")

    def stop(self):
        """停止相机流"""
        self.pipeline.stop()

if __name__ == "__main__":
    # 配置
    ACCUMULATE_INTERVAL = 0.5  # 0.5秒累积一帧
    SAVE_INTERVAL = 2.0        # 2.0秒保存一次

    accumulator = RealSensePointCloudAccumulator(
        save_dir="realsense_scans", # 统一保存到 'realsense_scans' 目录下
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    try:
        accumulator.start()
        
        # 初始化计时器
        last_accumulate_time = time.time()
        last_save_time = time.time()
        
        # 设置采集持续时间（例如：采集 60 秒）
        COLLECTION_DURATION = 60 # 您可以根据实际需要调整采集时长（秒）
        start_time = time.time()
        
        print(f"--- 采集将在 {COLLECTION_DURATION} 秒后自动停止 ---")

        while time.time() - start_time < COLLECTION_DURATION:
            current_time = time.time()

            # 1. 检查是否达到累积时间间隔 (0.5s)
            if current_time - last_accumulate_time >= ACCUMULATE_INTERVAL:
                accumulator.accumulate()
                last_accumulate_time = current_time # 重置累积计时器

            # 2. 检查是否达到保存时间间隔 (2.0s)
            if current_time - last_save_time >= SAVE_INTERVAL:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                accumulator.save(timestamp)
                last_save_time = current_time # 重置保存计时器
                
            # 简单休眠以避免 CPU 占用过高
            time.sleep(0.01)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        
    finally:
        print("\n--- 采集结束，停止相机 ---")
        accumulator.stop()
        
        # 最后的检查：如果在循环结束时还有未保存的点云，则保存一次
        if len(accumulator.full_pc) > 0:
            final_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_FINAL')
            accumulator.save(final_timestamp)