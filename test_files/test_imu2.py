import pyrealsense2 as rs
import numpy as np
import time

# 创建管道
pipeline = rs.pipeline()
config = rs.config()

def print_config_details(config):
    print("config 类属性和方法：")
    # 列出所有可用属性/方法
    for attr in dir(config):
        # print(attr)
        if not attr.startswith('_'):  # 过滤私有属性
            # print(attr)
            value = getattr(config, attr)
            # 只打印非函数的属性值（避免刷屏）
            if not callable(value):
                print(f"  {attr}: {value}")


# 调用函数
print_config_details(config)

# 启用 IMU 流（加速度计 和 陀螺仪）
# D455 的 IMU 支持的格式和采样率如下：
# Accelerometer: RS2_FORMAT_MOTION_XYZ32F, 63Hz (or 250Hz)
# Gyroscope:     RS2_FORMAT_MOTION_XYZ32F, 200Hz (or 400Hz)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)

# print(config)
# 启动管道
pipeline.start(config)

print("开始读取 IMU 数据（按 Ctrl+C 停止）...")

try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()

        # 获取 IMU 帧
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if accel_frame and accel_frame.is_motion_frame():
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            print(f"[加速度] X: {accel_data.x: .3f}, Y: {accel_data.y: .3f}, Z: {accel_data.z: .3f} [m/s²]")

        if gyro_frame and gyro_frame.is_motion_frame():
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            print(f"[陀螺仪] X: {gyro_data.x: .3f}, Y: {gyro_data.y: .3f}, Z: {gyro_data.z: .3f} [rad/s]")

        print("-" * 50)
        time.sleep(0.1)  # 控制打印频率

except KeyboardInterrupt:
    print("\n停止读取...")

finally:
    pipeline.stop()