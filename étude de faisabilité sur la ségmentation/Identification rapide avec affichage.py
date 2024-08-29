import pyrealsense2 as rs
from ultralytics import FastSAM
import torch
import cv2
import os
import numpy as np
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_mask(mask_binary, idx, depth_image, camera_matrix, fixed_colors):
    color_unique = fixed_colors[idx % len(fixed_colors)]

    ys, xs = np.where(mask_binary == 1)
    zs = depth_image[ys, xs] * 0.001  # Conversion de millimètres à mètres
    valid = (zs > 0) & (zs <= 2.0)
    
    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]

    x_3d = (xs - camera_matrix[0, 2]) * zs / camera_matrix[0, 0]
    y_3d = (ys - camera_matrix[1, 2]) * zs / camera_matrix[1, 1]

    points = np.stack((x_3d, y_3d, zs), axis=-1)
    colors = np.tile(color_unique, (len(points), 1))

    if points.size > 0:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud
    return None

# Optional: Clear CUDA cache
torch.cuda.empty_cache()

# Configuration of PyTorch CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load the FastSAM model
model = FastSAM('FastSAM-s.pt')

# Configure depth and color streams from RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Aligner pour aligner les images de profondeur sur les images de couleur
align = rs.align(rs.stream.color)

# Matrice de la caméra et coefficients de distorsion (à ajuster selon la caméra)
camera_matrix = np.array([[615.0, 0.0, 320.0],
                          [0.0, 615.0, 240.0],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Couleurs fixes pour chaque objet
fixed_colors = [
    [1.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0],   
    [0.0, 0.0, 1.0],   
    [1.0, 1.0, 0.0],   
    [1.0, 0.0, 1.0],   
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 0.0],   
    [0.5, 1.0, 0.5],  
    [1.0, 1.0, 0.5],  
    [1.0, 0.5, 1.0],  
    [0.6, 1.0, 1.0]
]

# Start streaming
pipeline.start(config)
time.sleep(1)

# Initialize Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
point_clouds = []

# Add a custom view control
view_control = vis.get_view_control()

# Save the initial view parameters
initial_view_params = view_control.convert_to_pinhole_camera_parameters()

while True:
    t1= time.time()
    #------- Recup des donnees -----------------------------------------------------
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    #------- Segmentation ----------------------------------------------------------

    with torch.no_grad():

        result = model.track(source=color_image, imgsz=226, save=False, show=False) 

        all_msk = []

        #-------- Recup des masques binaires ----------------------------------------
        for obj in result:
            if obj.masks is not None:
                for mask in obj.masks:
                    mask_tensor = mask.data[0]  # Obtenir le premier masque (Tensor PyTorch)
                    mask_np = mask_tensor.cpu().numpy()  # Convertir en numpy array
                    mask_resized = cv2.resize(mask_np, (color_image.shape[1], color_image.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    all_msk.append(mask_binary)


        #------- Création des nuages de points pour chaque objet -------------------
        point_clouds.clear()

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_mask, mask, idx, depth_image, camera_matrix, fixed_colors): idx for idx, mask in enumerate(all_msk)}
            for future in as_completed(futures):
                point_cloud = future.result()
                if point_cloud:
                    point_clouds.append(point_cloud)


        # Update visualization
        vis.clear_geometries()
        for pc in point_clouds:
            vis.add_geometry(pc)

        # Apply the saved view parameters
        view_control.convert_from_pinhole_camera_parameters(initial_view_params)

        # Poll events and update renderer
        vis.poll_events()
        vis.update_renderer()

    # Exit condition
    if not vis.poll_events():
        break
    t2=time.time()
    print('Temps = ', t2-t1)
pipeline.stop()
vis.destroy_window()


