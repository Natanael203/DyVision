'''
--------------------------------------------------------
Natanaël Jamet - contact : natanael.jamet@etu.unilim.fr |
Code datant de juillet 2024                             |
dernière modification : 19/08/2024                      |
Version 2.0 du pack "DyVision"                          |
--------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


Le but de ce script est d'intentifier des obstacles dans une scène fixe scannée au préalable via le script '2.0 - Reconstruction scene fixe.py'
Celui-ci genére un fichier .ply et un ficchier .Parquet qui sont ouverts et lus par le code suivant, il doit donc être dans le même répertoire.

Une camera realsense (une D435 a été utilisée pour develloper ce script) est utilisée pour recuperer les nuages de points
qui seront traités.

Un marqueur Aruco doit etre placé sans avoir été déplacé depuis le scan de la scène fixe, il a besoins d'être visible uniquement pour
le calcul du placement de la caméra par rapport à celui-ci.

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


les versions des librairies utilisées au moment de l'écriture du script sont les suivantes :

cupy       : 13.2.0
numpy      : 1.26.2
sklearn    : 1.5.1
scipy      : 1.13.0
csv        : 1.0
open3d     : 0.18.0
opencv     : 4.5.5
matplotlib : 3.8.1

La version de open cv est importante : elle est necessaire pour la detection des Aruco, la derniere version en date ne fonctionnait pas

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


En cas de changement de caméra ou de marqueur : verifier que les paramètres dist_coeffs, camera_matrix et taille_marqueur soient modifiés 
'''

import os
import cupy as cp
import numpy as np
from sklearn.cluster import Birch
from scipy.spatial import cKDTree
import time
import pyrealsense2 as rs
import csv
import open3d as o3d
import cv2
import pandas as pd
from numba import cuda




'''
------------------------------------------------------------------------------------------------------------------------
                                        - Definition des parametres et variables - 
------------------------------------------------------------------------------------------------------------------------
'''

# Configuration du pipeline de la caméra RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Chargement du dictionnaire de marqueurs ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

taille_marqueur = 0.105  # Taille du marqueur en mètres
th2 = 0.1  # Seuil pour le clustering Birch dynamique

# Matrice de la caméra et coefficients de distorsion (à ajuster selon la caméra)
camera_matrix = np.array([[615.0, 0.0, 320.0],
                          [0.0, 615.0, 240.0],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


'''
------------------------------------------------------------------------------------------------------------------------
                                        - Definition des fonctions - 
------------------------------------------------------------------------------------------------------------------------
'''


def id_cube(cluster):
    """Calculer les dimensions et le centre d'une boîte englobante entourant un cluster."""
    cluster = np.array(cluster)
    xmin, xmax = np.min(cluster[:, 0]), np.max(cluster[:, 0])
    ymin, ymax = np.min(cluster[:, 1]), np.max(cluster[:, 1])
    zmin, zmax = np.min(cluster[:, 2]), np.max(cluster[:, 2])

    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    centre = [xmin + Lx / 2, ymin + Ly / 2, zmin + Lz / 2]

    return [Lx, Ly, Lz], centre

def capture_point_cloud(pipeline, max_distance=2.0):
    """Capturer un nuage de points depuis la caméra RealSense et filtrer par distance."""
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None

    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    v = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    
    # Filtrer les points au-delà de la distance maximale
    mask = np.linalg.norm(v, axis=1) <= max_distance
    v = v[mask]
    
    return v

def capture_image():
    """Capture une image jusqu'à ce qu'un marqueur ArUco soit détecté."""
    print('Capture d\'une image pour placer les captures dynamiques dans le repère du marqueur : \n')
    while True:
        print("Appuyez sur 'Enter' pour capturer une image... \n")
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


def downsample_point_cloud_random(points_cp, factor):
    total_points = points_cp.shape[0]
    num_points_to_select = total_points // factor
    random_indices = cp.random.choice(total_points, num_points_to_select, replace=False)
    return points_cp[random_indices]


@cuda.jit
def filter_points_kernel(points, centers, zones_fixes, filtered_mask):
    idx = cuda.grid(1)
    if idx < points.shape[0]:
        x, y, z = points[idx]
        for j in range(centers.shape[0]):
            cx, cy, cz = centers[j]
            Lx, Ly, Lz = zones_fixes[j]
            if (cx - Lx / 2 <= x <= cx + Lx / 2 and
                cy - Ly / 2 <= y <= cy + Ly / 2 and
                cz - Lz / 2 <= z <= cz + Lz / 2):
                filtered_mask[idx] = 0
                return
        filtered_mask[idx] = 1

def filter_points(points, centers, zones_fixes):
    filtered_mask = cp.ones(len(points), dtype=cp.bool_)
    centers_cp = cp.array([zone[0] for zone in zones_fixes], dtype=cp.float32)
    zones_fixes_cp = cp.array([zone[1] for zone in zones_fixes], dtype=cp.float32)
    
    threads_per_block = 512  # Increase the number of threads per block
    blocks_per_grid = (len(points) + threads_per_block - 1) // threads_per_block
    if blocks_per_grid < 128:  # Ensure there are enough blocks
        blocks_per_grid = 128
    
    filter_points_kernel[blocks_per_grid, threads_per_block](points, centers_cp, zones_fixes_cp, filtered_mask)
    
    return points[filtered_mask.astype(cp.bool_)]



def read_bounding_boxes_from_parquet(parquet_file):
    """
    Lit un fichier Parquet contenant des informations sur les boîtes et les transforme en liste de tuples.

    :param parquet_file: Chemin vers le fichier Parquet.
    :return: Une liste de tuples, chaque tuple contient les coordonnées du centre et les dimensions de la boîte.
    """
    # Lire le fichier Parquet en DataFrame
    df = pd.read_parquet(parquet_file)
    
    # Transformer les données en liste de tuples (center, dimensions)
    bounding_boxes = []
    for _, row in df.iterrows():
        center = (row['CenterX'], row['CenterY'], row['CenterZ'])
        dimensions = (row['DimX'], row['DimY'], row['DimZ'])
        bounding_boxes.append((center, dimensions))
    
    return bounding_boxes
'''
------------------------------------------------------------------------------------------------------------------------
                                        - Recuperation et traitement de la scene statique - 
------------------------------------------------------------------------------------------------------------------------
'''
#-----------------------------------------------------------------
#------- Recuperation du nuage -----------------------------------
#-----------------------------------------------------------------

# Chemin du fichier .ply
ply_file_path = "combined_point_cloud.ply"

# Charger le nuage de points depuis le fichier .ply
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# Vérifier si le nuage de points est chargé correctement
if point_cloud.is_empty():
    print("Le nuage de points est vide.")
else:
    print("La scène statique a été chargée avec succès. \n")
# Convertir le nuage de points en un tableau NumPy de type float32
points_np = np.asarray(point_cloud.points, dtype=np.float32)
points_cp = cp.asarray(points_np)

Zones_fixes = read_bounding_boxes_from_parquet('selected_bounding_boxes.parquet')
print('Zones fixes chargées ')


'''
------------------------------------------------------------------------------------------------------------------------
                                        - Boucle Principale - 
------------------------------------------------------------------------------------------------------------------------
'''

#-----------------------------------------------------------------
#------- Definition des variables de traitement ------------------
#-----------------------------------------------------------------
# Définition du KD tree
centers = np.array([zone[0] for zone in Zones_fixes])
kdtree = cKDTree(centers)

# Définition du clustering Birch pour les données dynamiques
birch_model = Birch(n_clusters=None, threshold=th2)

# Variables de récupération des données
Historique_zones = []
Historique_clusters = []




#-----------------------------------------------------------------
#------- Detection de la position de la camera  ------------------
#-----------------------------------------------------------------
# Capture de l'image


color_frame, depth_frame, color_image, gray, corners, ids = capture_image()

start_time1 = time.time()  # Démarrer le chronométrage - optionel 

color_images = [color_image]

if ids is not None:
    cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, taille_marqueur, camera_matrix, dist_coeffs)

    position = tvecs[0].flatten()
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])

    rotation_matrix = rotation_matrix.T
    translation = -rotation_matrix @ position

rotation_matrix_cp = cp.asarray(rotation_matrix)
translation_cp = cp.asarray(translation)  

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"La detection de la position de la caméra a pris {elapsed_time1:.2f} secondes.")


#-----------------------------------------------------------------
#------- Boucle de récupération et traitement dynamique ----------
#-----------------------------------------------------------------
c = 1
print('Début de la boucle de traitement \n')
while True :

    #-------- Capture d'un nouveau point
    
    points_np2 = capture_point_cloud(pipeline)
    if points_np2 is None:
        print("Échec de la capture d'un nouveau nuage de points.")
        continue

    # Transformation des points
    points_cp2 = cp.asarray(points_np2)
    points_cpT = cp.dot(rotation_matrix_cp, points_cp2.T).T + translation_cp
    points_cpT_downsampled = downsample_point_cloud_random(points_cpT, 8)

    start_time = time.time()     
    # Filtrage des points
    filtered_points_cp = filter_points(points_cpT_downsampled, centers, Zones_fixes)
    
    #-------- Clustering ---------------------------------------------------------------
    if len(filtered_points_cp) > 0:
        filtered_points_np = cp.asnumpy(filtered_points_cp)
        cluster_labels2 = birch_model.fit_predict(filtered_points_np)
        num_clusters2 = len(set(cluster_labels2)) - (1 if -1 in cluster_labels2 else 0)
        clusters2 = [[] for _ in range(num_clusters2)]
        for i, label in enumerate(cluster_labels2):
            if label != -1 and label < num_clusters2:
                clusters2[label].append(filtered_points_np[i])
                

        #------- Calcul des bounding box  ----------------------------------------------
        Zones_dynamique = []
        for cluster in clusters2:
            if len(cluster) > 0:
                dimensions, center = id_cube(cluster)
                Zones_dynamique.append((center, dimensions))

        #-------- Préparation des données pour pandas ---------------------------------
        data = []
        for cluster_idx, cluster in enumerate(clusters2):
            for point in cluster:
                data.append([cluster_idx, point[0], point[1], point[2]])


        bbox_data = []
        for bbox_idx, (center, dimensions) in enumerate(Zones_dynamique):
            bbox_data.append([bbox_idx, center[0], center[1], center[2], dimensions[0], dimensions[1], dimensions[2]])

        # Création des DataFrames pandas
        df_clusters = pd.DataFrame(data, columns=['Cluster', 'X', 'Y', 'Z'])
        df_bboxes = pd.DataFrame(bbox_data, columns=['BoundingBox', 'CenterX', 'CenterY', 'CenterZ', 'DimX', 'DimY', 'DimZ'])

        # Écriture des DataFrames dans des fichiers Parquet
        df_clusters.to_parquet('clusters.parquet', index=False)
        df_bboxes.to_parquet('bounding_boxes.parquet', index=False)
                
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Iteration {c} a pris {elapsed_time:.2f} secondes.") #  optionel ( prend du temps pour afficher )

    c += 1

pipeline.stop()
