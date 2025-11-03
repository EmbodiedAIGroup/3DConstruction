import pyrealsense2 as rs

# 创建上下文并列举所有设备
ctx = rs.context()
devices = ctx.query_devices()

if not devices:
    print("未检测到 RealSense 设备。")
else:
    for dev in devices:
        print(f"\n设备序列号: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"设备名称: {dev.get_info(rs.camera_info.name)}")

        # 获取该设备的所有传感器
        sensors = dev.query_sensors()
        has_accel = False
        has_gyro = False

        for sensor in sensors:
            # 查询该传感器支持的流配置
            profiles = sensor.get_stream_profiles()
            for p in profiles:
                if p.stream_type() == rs.stream.accel:
                    has_accel = True
                if p.stream_type() == rs.stream.gyro:
                    has_gyro = True

        print(f"  支持加速度计: {has_accel}")
        print(f"  支持陀螺仪: {has_gyro}")

        if has_accel and has_gyro:
            print("  此设备包含 IMU（惯性测量单元）。")
        else:
            print("  此设备不包含完整的 IMU。")