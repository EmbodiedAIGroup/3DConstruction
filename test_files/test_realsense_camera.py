import pyrealsense2 as rs
import numpy as np
import cv2

def test_realsense_camera():
    # 初始化相机管道
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 获取设备信息并配置流
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    # 检查是否有深度传感器
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("该设备没有RGB传感器!")
        return
    
    # 配置流：彩色和深度
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 启动管道
    try:
        profile = pipeline.start(config)
        
        # 获取深度传感器的深度标尺
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"depth: {depth_scale} (1 point depth = {depth_scale} meters)")
        
        # 创建对齐对象（将深度框与彩色框对齐）
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        print("\n相机测试开始 - 按以下键操作:")
        print("  's' - 保存当前帧")
        print("  'm' - 测量鼠标点击点的距离")
        print("  'q' - 退出程序")
        
        save_count = 0
        measure_mode = False
        
        while True:
            # 等待获取帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
            
            # 转换为numpy数组
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 深度图着色以便可视化
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # 如果在测量模式，显示提示
            if measure_mode:
                cv2.putText(color_image, "Measurement Mode - Tap any point to measure distance", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('Color Image', color_image)
            cv2.imshow('Depth Image', depth_colormap)
            
            # 鼠标回调函数 - 用于测量距离
            def mouse_callback(event, x, y, flags, param):
                nonlocal measure_mode, depth_image, depth_scale, color_image
                if event == cv2.EVENT_LBUTTONDOWN and measure_mode:
                    # 获取点击点的深度值
                    depth = depth_image[y, x] * depth_scale

                    # === 新增：在命令行（控制台）显示距离 ===
                    print(f"Clicked at ({x}, {y}). Distance: {depth:.2f} meters")

                    # 在图像上显示距离
                    cv2.putText(color_image, f"Distance: {depth:.2f} meters", (x+10, y+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow('Color Image', color_image)

            # 设置鼠标回调
            cv2.setMouseCallback('Color Image', mouse_callback)

            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                cv2.imwrite(f"color_{save_count}.png", color_image)
                cv2.imwrite(f"depth_{save_count}.png", depth_colormap)
                print(f"Saved frame {save_count}")
                save_count += 1
            elif key == ord('m'):
                # Toggle measurement mode
                measure_mode = not measure_mode
                print(f"Measurement Mode {'Enabled' if measure_mode else 'Disabled'}")
    
    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        test_realsense_camera()
    except Exception as e:
        print(f"发生错误: {e}")
        print("请确保已正确安装pyrealsense2库和RealSense驱动")
        print("安装命令: pip install pyrealsense2")