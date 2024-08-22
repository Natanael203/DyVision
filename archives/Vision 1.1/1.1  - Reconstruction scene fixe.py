'''
--------------------------------------------------------
Natanaël Jamet - contact : natanael.jamet@etu.unilim.fr |
Code datant de juillet 2024                             |
Dernière modification :30.07.2024                       |
Version 1.1 du pack "Vision"                            |
--------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


Le but de ce script est d'intentifier des obstacles dans une scène fixe scannée au préalable via le script '1.1  - Reconstruction scene fixe.py'
Celui-ci genére un fichier .ply qui est ouvert et lu par le code suivant, il doit donc être dans le même répertoire.

Une camera realsense (une D435 a été utilisée pour develloper ce script) est utilisée pour recuperer les nuages de points
qui seront traités.

Un marqueur Aruco doit etre placé sans avoir été déplacé depuis le scan de la scène fixe, il a besoins d'être visible uniquement pour
le calcul du placement de la caméra par rapport à celui-ci.

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


les versions des librairies utilisées au moment de l'écriture du script sont les suivantes :

sklearn    : 1.5.1
csv        : 1.0
open3d     : 0.18.0
matplotlib : 3.8.1

La version de open cv est importante : elle est necessaire pour la detection des Aruco, la derniere version en date ne fonctionnait pas

-------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------

En cas de changement de caméra ou de marqueur : verifier que les paramètres dist_coeffs, camera_matrix et taille_marqueur soient modifiés 

'''

import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import os

'''
---------------------------------------------------------------------------------------------------
                                - Defintion des parametres et variables - 
----------------------------------------------------------------------------------------------------
'''

# Initialisation de la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Chargement du dictionnaire de marqueurs ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

taille_marqueur = 0.105  # Taille du marqueur en m

# Matrice de la caméra et coefficients de distorsion (à ajuster selon la caméra)
camera_matrix = np.array([[615.0, 0.0, 320.0],
                          [0.0, 615.0, 240.0],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

'''
---------------------------------------------------------------------------------------------------
                                - Defintion des Fonctions  - 
----------------------------------------------------------------------------------------------------
'''

# Cette fonction prend en argument une depth frame, l'image rgb, un float et renvoie un nuage de point genere a partir de la depth frame dont les points au dela de max_distance ont ete retires 
def depth_to_point_cloud(depth_frame, color_frame, max_distance=2.0):
    """Convertir un frame de profondeur en nuage de points avec filtrage de distance."""
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Créer un objet point cloud
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    vertices = np.asanyarray(points.get_vertices())

    # Convertir en tableau NumPy classique
    vertices = np.array([list(vertex) for vertex in vertices])
    
    # Filtrer les points en fonction de la distance maximale
    distances = np.linalg.norm(vertices, axis=1)
    mask = distances <= max_distance
    filtered_vertices = vertices[mask]
    
    return filtered_vertices, color_image

# Cette fonction capture une image contenant un marqueur Aruco en demandant a l'utilisateur, et renvoie les image profondeur et rgb ainsi que les caracteristique du marqueur detecte 
def capture_image():
    """Capture une image jusqu'à ce qu'un marqueur ArUco soit détecté."""
    while True:
        print("Appuyez sur 'Enter' pour capturer une image.")
        input()  # Attendre l'entrée de l'utilisateur pour capturer une image
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Impossible de capturer l'image.")

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Détecter les marqueurs ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            print("Marqueur détecté.")
            return color_frame, depth_frame, color_image, gray, corners, ids
        else:
            print("Aucun marqueur détecté. Veuillez reprendre la photo.")

'''
---------------------------------------------------------------------------------------------------
                                - Code principal  - 
----------------------------------------------------------------------------------------------------
'''

try:
    n = int(input("Combien de prises de vue souhaitez-vous effectuer ? "))

    point_clouds = []
    color_images = []

    for i in range(n):
        print(f"Prise de vue {i+1}/{n}")
        color_frame, depth_frame, color_image, gray, corners, ids = capture_image()
        color_images.append(color_image)
        
        # Traitement de l'image
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, taille_marqueur, camera_matrix, dist_coeffs)
            for j in range(len(ids)):
                cv2.aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvecs[j], tvecs[j], 0.1)

                position = tvecs[j].flatten()
                rotation_matrix, _ = cv2.Rodrigues(rvecs[j])

                point_cloud, _ = depth_to_point_cloud(depth_frame, color_frame)
                rotation_matrix = rotation_matrix.T
                translation = -rotation_matrix @ position
                transformed_points = (rotation_matrix @ point_cloud.T).T + translation

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(transformed_points)
                point_clouds.append(pcd)

    # Combiner les nuages de points
    if point_clouds:
        pcd_combined = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            pcd_combined += pcd
        
        # Downscale the point cloud
        num_points = len(pcd_combined.points)
        downscale_factor = 8*n
        indices = np.random.choice(num_points, num_points // downscale_factor, replace=False)
        downscaled_pcd = pcd_combined.select_by_index(indices)

        # Enregistrer le nuage de points combiné dans un fichier .ply
        output_filename = "combined_point_cloud.ply"
        o3d.io.write_point_cloud(output_filename, downscaled_pcd)
        print(f"Nuage de points combiné et réduit enregistré dans {output_filename}")

        # Afficher le nuage de points combiné et réduit
        o3d.visualization.draw_geometries([downscaled_pcd], window_name="Downscaled Combined Point Clouds")

    # Afficher les images avec les marqueurs détectés
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(cv2.cvtColor(color_images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'Image {i+1}')
        plt.axis('off')

    plt.show()

finally:
    pipeline.stop()
    plt.close()
