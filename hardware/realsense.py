import numpy as np
import cv2
import pyrealsense2 as rs
import threading
from queue import Queue
import time
import os
from datetime import datetime

class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=15, save_images=False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 配置流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        
        # 对齐器（深度图对齐到彩色图）
        self.align = rs.align(rs.stream.color)
        
        # 深度尺度（用于转换为米）
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        
        # 内参缓存
        self.intrinsics = self._get_intrinsics()
        
        # 数据队列（保存最新帧）
        self.latest_frames = Queue(maxsize=1)
        self.shutdown_event = threading.Event()
        
        # 图像保存设置
        self.save_images = save_images
        self.frame_count = 0
        if self.save_images:
            self._init_save_dir()
        
        # 启动采集线程
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _get_intrinsics(self):
        """获取彩色相机内参"""
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return {
            "fx": intr.fx, "fy": intr.fy,
            "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height
        }

    def _init_save_dir(self):
        """初始化图像保存目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("saved_images", f"realsense_{timestamp}")
        os.makedirs(os.path.join(self.save_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "depth"), exist_ok=True)
        print(f"图像保存路径: {self.save_dir}")

    def _capture_loop(self):
        """后台采集图像帧"""
        while not self.shutdown_event.is_set():
            try:
                # 获取帧并对齐
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # 转换为numpy数组
                rgb_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())
                
                # 保存图像（如果启用）
                if self.save_images:
                    self._save_frame(rgb_img, depth_img)
                
                # 存入队列（只保留最新帧）
                if self.latest_frames.full():
                    self.latest_frames.get_nowait()
                self.latest_frames.put((rgb_img, depth_img))
                
                self.frame_count += 1
                time.sleep(0.01)  # 降低CPU占用
                
            except Exception as e:
                print(f"相机采集错误: {e}")
                time.sleep(0.1)

    def _save_frame(self, rgb_img, depth_img):
        """保存RGB和深度图像"""
        rgb_path = os.path.join(self.save_dir, "rgb", f"rgb_{self.frame_count:06d}.jpg")
        depth_path = os.path.join(self.save_dir, "depth", f"depth_{self.frame_count:06d}.png")
        
        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img.astype(np.uint16))  # 深度图保存为16位

    def get_frames(self):
        """获取最新的RGB图和深度图（米为单位）"""
        if not self.latest_frames.empty():
            rgb_img, depth_img = self.latest_frames.get()
            # 转换深度图单位为米
            depth_meters = depth_img.astype(np.float32) * self.depth_scale
            return rgb_img, depth_meters
        return None, None

    def depth_to_point_cloud(self, depth_meters):
        """将深度图转换为相机坐标系下的点云"""
        h, w = depth_meters.shape
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]
        
        # 生成像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        z = depth_meters.flatten()
        
        # 过滤无效深度
        mask = z > 0
        u, v, z = u[mask], v[mask], z[mask]
        
        # 转换为3D坐标（相机坐标系）
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return np.column_stack((x, y, z))

    def release(self):
        """释放资源"""
        self.shutdown_event.set()
        self.capture_thread.join()
        self.pipeline.stop()