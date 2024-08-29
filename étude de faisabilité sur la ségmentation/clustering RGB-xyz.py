import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
from sklearn.cluster import KMeans, DBSCAN

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
time.sleep(1)

def capture_point_cloud(pipeline, max_distance=2.0):
    """Capture a point cloud from the RealSense camera and filter by distance."""
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None

    # Create a point cloud object
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    v = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    c = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    
    # Filter points beyond the maximum distance
    mask = np.linalg.norm(v, axis=1) <= max_distance
    v = v[mask]
    c = c[mask]
    
    return v, c, color_frame

def map_texture_to_color(color_frame, texture_coords):
    """Map texture coordinates to RGB colors."""
    color_image = np.asanyarray(color_frame.get_data())
    height, width, _ = color_image.shape
    colors = []
    for coord in texture_coords:
        u = min(max(int(coord[0] * width), 0), width - 1)
        v = min(max(int(coord[1] * height), 0), height - 1)
        colors.append(color_image[v, u] / 255.0)
    return np.array(colors)

def apply_clustering_rgb(points, colors, n_clusters):
    """Apply K-means clustering on the colors and return the clustered points."""
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(colors)
    
    clusters = []
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        clusters.append(cluster_points)
    
    return labels, clusters

def apply_clustering_xyz(points, eps, min_samples):
    """Apply DBSCAN clustering on the XYZ coordinates and return the clustered points."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    
    clusters = []
    for i in np.unique(labels):
        if i == -1:  # Noise points
            continue
        cluster_points = points[labels == i]
        clusters.append(cluster_points)
    
    return labels, clusters

def visualize_point_cloud_with_clusters(all_points, all_colors):
    """Visualize the point cloud with clusters using Open3D."""
    if all_points is not None and all_points.size > 0:
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        # Display the point cloud
        o3d.visualization.draw_geometries([pcd])
    else:
        print("No points to display.")

def downsample_point_cloud_random(points, colors, factor):
    """Downsample the point cloud by randomly selecting points."""
    total_points = points.shape[0]
    num_points_to_select = total_points // factor
    random_indices = np.random.choice(total_points, num_points_to_select, replace=False)
    return points[random_indices], colors[random_indices]

def count_points_within_radius(points, center, radius):
    """Count the number of points within a sphere of given radius centered on a point."""
    distances = np.linalg.norm(points - center, axis=1)
    count = np.sum(distances <= radius)
    return count

def filter_points_by_density(points, radius, threshold, colors):
    """Remove points with density below a threshold."""
    filtered_points = []
    filtered_colors = []
    for i, point in enumerate(points):
        density = count_points_within_radius(points, point, radius)
        if density >= threshold:
            filtered_points.append(point)
            filtered_colors.append(colors[i])
    return np.array(filtered_points), np.array(filtered_colors)

radius = 0.05
threshold = 30
n_clusters = 3
eps = 0.03
min_samples = 10

try:
    # Capture the point cloud
    point_cloud, texture_coords, color_frame = capture_point_cloud(pipeline, max_distance=2.0)
    if point_cloud is not None:
        # Map texture coordinates to RGB colors
        colors = map_texture_to_color(color_frame, texture_coords)
        
        # Downsample the point cloud
        point_cloud_ds, colors_ds = downsample_point_cloud_random(point_cloud, colors, 5)
        
        # Apply density filtering
        point_cloud_filtered, colors_filtered = filter_points_by_density(point_cloud_ds, radius, threshold, colors_ds)
        
        # Apply K-means clustering on the colors
        labels_rgb, clusters_rgb = apply_clustering_rgb(point_cloud_filtered, colors_filtered, n_clusters)
        
        all_points_rgb = []
        all_colors_rgb = []
        
        # Assign colors for the K-means clusters
        unique_labels_rgb = np.unique(labels_rgb)
        for i, cluster in enumerate(clusters_rgb):
            color = np.random.rand(3)  # Random color for each RGB cluster
            all_points_rgb.append(cluster)
            all_colors_rgb.append(np.tile(color, (cluster.shape[0], 1)))
        
        # Concatenate all points and colors
        all_points_rgb = np.concatenate(all_points_rgb, axis=0)
        all_colors_rgb = np.concatenate(all_colors_rgb, axis=0)
        
        # Visualize the point cloud with K-means clustering results
        visualize_point_cloud_with_clusters(all_points_rgb, all_colors_rgb)
        
        # Apply DBSCAN clustering on the XYZ coordinates for each RGB cluster
        all_points_xyz = []
        all_colors_xyz = []
        
        for i, cluster in enumerate(clusters_rgb):
            labels_xyz, clusters_xyz = apply_clustering_xyz(cluster, eps, min_samples)
            
            # Assign colors for the XYZ clusters
            unique_labels_xyz = np.unique(labels_xyz)
            for j, cluster_xyz in enumerate(clusters_xyz):
                color = np.random.rand(3)  # Random color for each XYZ cluster
                all_points_xyz.append(cluster_xyz)
                all_colors_xyz.append(np.tile(color, (cluster_xyz.shape[0], 1)))
        
        # Concatenate all points and colors
        all_points_xyz = np.concatenate(all_points_xyz, axis=0)
        all_colors_xyz = np.concatenate(all_colors_xyz, axis=0)
        
        # Visualize the point cloud with DBSCAN clustering results
        visualize_point_cloud_with_clusters(all_points_xyz, all_colors_xyz)
        
    else:
        print("Failed to capture point cloud.")

finally:
    # Stop streaming
    pipeline.stop()
