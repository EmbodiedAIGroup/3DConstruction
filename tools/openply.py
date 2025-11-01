import open3d as o3d
import numpy as np

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
your_ply_file = "path/to/your/point_cloud.ply"

# 执行函数
visualize_ply(your_ply_file)