import os
import numpy as np
import open3d as o3d
from datetime import datetime

def merge_point_clouds(npy_dir, output_dir=None, output_prefix="merged_pc"):
    """
    合并多个点云numpy文件并生成合并后的PLY文件
    
    参数:
        npy_dir: 存放accumulated_pc_{timestamp}_data.npy文件的目录
        output_dir: 输出目录，默认与npy_dir相同
        output_prefix: 输出文件前缀
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = npy_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有符合命名规则的npy文件
    npy_files = []
    for file in os.listdir(npy_dir):
        if file.startswith("accumulated_pc_") and file.endswith("_data.npy"):
            npy_files.append(os.path.join(npy_dir, file))
    
    if not npy_files:
        print("❌ 未找到符合条件的numpy文件")
        return
    
    print(f"📁 找到 {len(npy_files)} 个点云文件，开始合并...")
    
    # 初始化合并数组
    all_points = []
    all_colors = []
    
    # 逐个加载并合并文件
    for i, file_path in enumerate(npy_files, 1):
        try:
            # 加载numpy字典数据
            data = np.load(file_path, allow_pickle=True).item()
            points = data["points"]
            colors = data["colors"]
            
            # 检查数据形状是否匹配
            if len(points) != len(colors):
                print(f"⚠️ 警告: {os.path.basename(file_path)} 中点坐标与颜色数量不匹配，已跳过")
                continue
            
            all_points.append(points)
            all_colors.append(colors)
            print(f"✅ 已加载 {i}/{len(npy_files)}: {os.path.basename(file_path)} (点数: {len(points)})")
        
        except Exception as e:
            print(f"❌ 加载 {os.path.basename(file_path)} 失败: {str(e)}")
            continue
    
    if not all_points:
        print("❌ 没有可合并的有效点云数据")
        return
    
    # 合并所有点云和颜色
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    total_points = len(merged_points)
    print(f"\n📊 合并完成，总点数: {total_points}")
    
    # 生成输出文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_npy_path = os.path.join(output_dir, f"{output_prefix}_{timestamp}.npy")
    merged_ply_path = os.path.join(output_dir, f"{output_prefix}_{timestamp}.ply")
    
    # 保存合并后的numpy文件
    merged_data = {
        "points": merged_points,
        "colors": merged_colors,
        "source_files": [os.path.basename(f) for f in npy_files]  # 记录来源文件
    }
    np.save(merged_npy_path, merged_data)
    print(f"💾 已保存合并的numpy文件至: {merged_npy_path}")
    
    # 保存合并后的PLY文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    o3d.io.write_point_cloud(merged_ply_path, pcd)
    print(f"💾 已保存合并的PLY文件至: {merged_ply_path}")

if __name__ == "__main__":
    # 配置参数
    NPY_DIRECTORY = "./point_clouds"  # 替换为你的npy文件目录
    OUTPUT_DIRECTORY = "./merged_results"  # 输出目录，可设为None使用输入目录
    
    # 执行合并
    merge_point_clouds(
        npy_dir=NPY_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        output_prefix="merged_rooms"  # 输出文件前缀
    )