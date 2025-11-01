import pyrealsense2 as rs
import numpy as np
import os
import time
import cv2 # 用于显示图像和按键检测

# --- 配置参数 ---
# 分辨率和帧率
WIDTH = 640
HEIGHT = 480
FPS = 30
# 保存目录
SAVE_DIR = "realsense_pointclouds"

def setup_realsense_pipeline():
    """初始化RealSense管线、配置流并返回所需对象"""
    
    # 1. 创建一个管线对象
    pipeline = rs.pipeline()
    # 2. 创建一个配置对象
    config = rs.config()

    # 启用深度和彩色流
    print(f"启用流: 深度 {WIDTH}x{HEIGHT}@{FPS}Hz, 彩色 {WIDTH}x{HEIGHT}@{FPS}Hz")
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # 3. 启动管线
    # 注意: 如果有多个相机，可以使用 config.enable_device(serial_number) 指定
    profile = pipeline.start(config)

    # 4. 获取深度传感器，用于深度单位 (Depth Scale)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度单位 (Depth Scale) 是: {depth_scale} 米/像素")

    # 5. 创建对齐对象 (将深度图像对齐到彩色图像)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 6. 创建点云对象
    pc = rs.pointcloud()

    return pipeline, align, pc, depth_scale

def save_point_cloud(pointcloud_data, color_frame, frame_count):
    """使用 pyrealsense2 内置的 PLY 导出功能保存点云"""
    
    # 确保保存目录存在
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 构造文件名
    filename = os.path.join(SAVE_DIR, f"pointcloud_{frame_count:06d}.ply")

    try:
        # 使用 pyrealsense2 的 save_to_ply 类
        # ply_saver = rs.save_to_ply(filename)
        # ply_saver.set_option(rs.save_to_ply.option_ply_binary, True) # 可选: 保存为二进制格式
        
        # 更好的方法: 使用 pointcloud 对象的 export_to_ply 方法 (更简洁)
        pointcloud_data.export_to_ply(filename, color_frame)

        print(f"\n✨ 成功保存点云至: {filename} (包含RGB信息) ✨")
    except Exception as e:
        print(f"保存点云时发生错误: {e}")

def main():
    """主函数，负责数据流处理和保存逻辑"""
    
    pipeline, align, pc, depth_scale = setup_realsense_pipeline()
    frame_count = 0

    try:
        print("\n--- 相机已启动。按 's' 键保存当前点云，按 'q' 键退出 ---\n")
        
        while True:
            # 1. 等待新帧集
            # wait_for_frames() 会阻塞直到接收到所有配置的流的帧
            frames = pipeline.wait_for_frames()

            # 2. 对齐帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # 检查是否有有效帧
            if not aligned_depth_frame or not color_frame:
                continue

            # 3. 生成点云
            # 将点云对象映射到彩色帧（以便获取RGB数据）
            pc.map_to(color_frame)
            # 计算点云
            points = pc.calculate(aligned_depth_frame)
            
            # 4. 可视化 (可选, 使用 OpenCV)
            # 将彩色帧转换为 numpy 数组以进行显示
            color_image = np.asanyarray(color_frame.get_data())
            
            # 显示彩色图像
            cv2.imshow('RealSense Color Stream (Press "s" to save PLY, "q" to quit)', color_image)
            
            key = cv2.waitKey(1) & 0xFF

            # 5. 保存逻辑
            if key == ord('s'):
                frame_count += 1
                save_point_cloud(points, color_frame, frame_count)
            
            # 6. 退出逻辑
            if key == ord('q'):
                break

    except Exception as e:
        print(f"运行过程中发生错误: {e}")
    finally:
        # 停止流，释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机流已停止，程序退出。")

if __name__ == "__main__":
    main()