import pyrealsense2 as rs
from ultralytics import FastSAM
import torch
import cv2
import os
import numpy as np
import open3d as o3d
import time

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
    [1.0, 1.0,0.5],  
    [1.0,0.5, 1.0],  
    [0.6, 1.0, 1.0]
]

# Start streaming
pipeline.start(config)
time.sleep(1)

c = 1
while True:
    c += 1
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
        t1=time.time()

        result = model.track(source=color_image, imgsz=226, save=False, show=True)

        all_msk = []

        #-------- Recup des masques binaires ----------------------------------------
        for obj in result:
            if obj.masks is not None:
                for mask in obj.masks:
                    maskbin = mask.data
                    maskbin = maskbin[0].cpu().numpy()
                    mask_resized = cv2.resize(maskbin, (color_image.shape[1], color_image.shape[0]))

                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    all_msk.append(mask_binary)
        t3=time.time()
        print('Temps segmentation et recuperation des masques = ', t3-t1)
        #------- Création des nuages de points pour chaque objet -------------------
        point_clouds = []
        t4=time.time()
        for idx, mask in enumerate(all_msk):
            points = []
            colors = []

            # Utiliser une couleur fixe pour chaque objet
            color_unique = fixed_colors[idx % len(fixed_colors)]

            for y in range(depth_image.shape[0]):
                for x in range(depth_image.shape[1]):
                    if mask[y, x] == 1:
                        z = depth_image[y, x] * 0.001  # Conversion de millimètres à mètres
                        if z == 0 or z > 2.0:  # Ignorer les points sans profondeur ou au-delà de 2 mètres
                            continue

                        # Convertir les coordonnées (x, y, z) de l'image en coordonnées 3D
                        u, v = x, y
                        x_3d = (u - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
                        y_3d = (v - camera_matrix[1, 2]) * z / camera_matrix[1, 1]

                        points.append([x_3d, y_3d, z])
                        colors.append(color_unique)  # Utiliser la couleur fixe pour chaque objet
    
            # Convertir les listes en numpy arrays et vérifier la taille
            if points:
                points = np.array(points)
                colors = np.array(colors)

                # Vérifier que points a bien la forme (N, 3)
                if points.shape[1] == 3:
                    # Créer un nuage de points avec Open3D
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(points)
                    point_cloud.colors = o3d.utility.Vector3dVector(colors)

                    # Ajouter le nuage de points à la collection
                    point_clouds.append(point_cloud)


        # Visualiser tous les nuages de points dans une seule fenêtre
        if point_clouds:
            o3d.visualization.draw_geometries(point_clouds)



pipeline.stop()
