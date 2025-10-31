import open3d as o3d
pcd = o3d.io.read_point_cloud("save_results/map_step_37.pcd")
o3d.visualization.draw_geometries([pcd])  # 显示点云