import os
import open3d as o3d
import numpy as np
from pathlib import Path


def find_latest_created_folder(current_dir):
    """
    找到当前工作目录下最新创建的文件夹
    
    返回:
        str: 最新创建的文件夹的完整路径，如果没有找到则返回None
    """
    
    # 存储所有文件夹及其创建时间
    folders = []
    
    # 遍历当前目录中的所有条目
    for entry in os.scandir(current_dir):
        # 只考虑文件夹，且排除特殊的.和..
        if entry.is_dir(follow_symlinks=False) and entry.name not in ('.', '..'):
            try:
                # 获取创建时间（Windows使用st_ctime，Unix-like系统也可用st_birthtime）
                if os.name == 'nt':  # Windows系统
                    create_time = entry.stat().st_ctime
                else:  # Unix/Linux/macOS系统
                    # 尝试获取创建时间，若不支持则 fallback 到修改时间
                    try:
                        create_time = entry.stat().st_birthtime
                    except AttributeError:
                        create_time = entry.stat().st_ctime
                
                fpath = entry.path.replace('\\', '/')
                folders.append((create_time, fpath))
            except OSError as e:
                print(f"无法访问文件夹 {entry.name}: {e}")
                continue
    
    if not folders:
        return None
    
    # 按创建时间排序（最新的在前）
    folders.sort(reverse=True, key=lambda x: x[0])
    return folders[0][1]


def find_latest_ply_file(folder_path):
    """
    找到指定文件夹中最新的.ply文件
    
    参数:
        folder_path (str): 文件夹路径
        
    返回:
        str: 最新.ply文件的完整路径，如果没有找到则返回None
    """
    # 确保路径存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"不是一个文件夹: {folder_path}")
    
    # 存储所有.ply文件及其修改时间
    ply_files = []
    
    # 遍历文件夹中的所有文件
    for entry in os.scandir(folder_path):
        if entry.is_file() and entry.name.lower().endswith('.ply'):
            # 获取文件的最后修改时间（时间戳）
            modified_time = entry.stat().st_mtime
            fpath = entry.path.replace('\\', '/')
            ply_files.append((modified_time, fpath))
    
    # 如果没有找到.ply文件，返回None
    if not ply_files:
        return None
    
    # 按修改时间排序，取最新的一个
    ply_files.sort(reverse=True, key=lambda x: x[0])
    return ply_files[0][1]


def visualize_ply(file_path):
    """
    读取 PLY 点云文件并使用 Open3D 进行可视化。

    Args:
        file_path (str): PLY 文件的路径。
    """
    print(f"正在读取点云文件: {file_path}")
    
    # 1. 读取点云文件
    # Open3D 的 io.read_point_cloud 函数会自动根据文件扩展名识别格式
    pcd = o3d.io.read_point_cloud(file_path)
    
    if not pcd.has_points():
        print("警告: 点云文件中没有点。")
        return

    # 打印一些基本信息
    print(pcd)
    print(f"点数量: {len(pcd.points)}")
    
    # 如果点云包含颜色信息，打印颜色信息
    if pcd.has_colors():
        print(f"包含颜色信息。部分点的颜色示例:\n{np.asarray(pcd.colors)[:5]}")

    # 2. 可视化点云
    # draw_geometries 函数会打开一个交互式窗口来显示 3D 几何体
    print("开始可视化点云。在弹出的窗口中使用鼠标/触摸板进行交互。")
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Open3D PLY Viewer",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50)

# --- 替换为你的 PLY 文件路径 ---

folder = "D:/X_Projects/YProjects/3DConstruction/test_files"  # 替换为实际文件夹路径
# 获取最新创建的文件夹路径
latest_folder = find_latest_created_folder(folder)
print(f"最新创建的文件夹: {latest_folder}")
# 获取该文件夹中最新的 PLY 文件路径
latest_ply = find_latest_ply_file(latest_folder)
print(f"最新的 PLY 文件: {latest_ply}")
# 执行函数
visualize_ply(latest_ply)