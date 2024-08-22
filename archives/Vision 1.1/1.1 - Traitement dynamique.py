'''
--------------------------------------------------------
Natanaël Jamet - contact : natanael.jamet@etu.unilim.fr |
Code datant de juillet 2024                             |
dernière modification : xx.xx.xxxx                      |
Version 1.1 du pack "Vision"                            |
--------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


Le but de ce script est d'intentifier des obstacles dans une scène fixe scannée au préalable via le script 'A  - Reconstruction scene fixe.py'
Celui-ci genére un fichier .ply qui est ouvert et lu par le code suivant, il doit donc être dans le même répertoire.

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
th1 = 0.06  # Seuil pour le clustering Birch statique
th2 = 0.02  # Seuil pour le clustering Birch dynamique

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

def filter_points(points, kdtree_centers, zones_fixes):
    """Filtrer les points en fonction des zones fixes."""
    points_cp = cp.asarray(points)
    centers_cp = cp.asarray(kdtree_centers)

    # Créer un masque filtrant initialement à True
    filtered_mask_cp = cp.ones(len(points_cp), dtype=cp.bool_)

    # Vectorisation des boîtes
    for centre, dimensions in zones_fixes:
        Lx, Ly, Lz = dimensions
        min_x, max_x = centre[0] - Lx / 2, centre[0] + Lx / 2
        min_y, max_y = centre[1] - Ly / 2, centre[1] + Ly / 2
        min_z, max_z = centre[2] - Lz / 2, centre[2] + Lz / 2
        
        # Filtrage des points en utilisant le broadcasting
        in_box = (points_cp[:, 0] >= min_x) & (points_cp[:, 0] <= max_x) & \
                 (points_cp[:, 1] >= min_y) & (points_cp[:, 1] <= max_y) & \
                 (points_cp[:, 2] >= min_z) & (points_cp[:, 2] <= max_z)
        
        # Mise à jour du masque filtrant
        filtered_mask_cp &= ~in_box

    # Application du masque filtrant aux points
    filtered_points_cp = points_cp[filtered_mask_cp]
    return cp.asnumpy(filtered_points_cp)


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

#-----------------------------------------------------------------
#------- Clustering du nuage -------------------------------------
#-----------------------------------------------------------------
birch_model = Birch(n_clusters=None, threshold=th1)

# Conversion en numpy pour le clustering
points_np = cp.asnumpy(points_cp)
cluster_labels = birch_model.fit_predict(points_np)

num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
clusters = [[] for _ in range(num_clusters)]
for i, cluster_label in enumerate(cluster_labels):
    if cluster_label != -1:
        if 0 <= cluster_label < num_clusters:
            clusters[cluster_label].append(points_np[i])

#-----------------------------------------------------------------
#------- Calcul des zones fixes ----------------------------------
#-----------------------------------------------------------------
Zones_fixes = []
for cluster in clusters:
    if len(cluster) > 0:
        dimensions, center = id_cube(cluster)
        Zones_fixes.append((center, dimensions))
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

color_images = [color_image]

if ids is not None:
    cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, taille_marqueur, camera_matrix, dist_coeffs)

    position = tvecs[0].flatten()
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])

    rotation_matrix = rotation_matrix.T
    translation = -rotation_matrix @ position

rotation_matrix_cp = cp.asarray(rotation_matrix)  # Convertir en cupy.ndarray
translation_cp = cp.asarray(translation)  # Convertir en cupy.ndarray




#-----------------------------------------------------------------
#------- Boucle de récupération et traitement dynamique ----------
#-----------------------------------------------------------------
c = 1
print('Début de la boucle de traitement \n')
while True:  # Changez à while True pour un traitement continu
    start_time = time.time()  # Démarrer le chronométrage

    # Capture d'un nouveau point
    
    points_np2 = capture_point_cloud(pipeline)
    if points_np2 is None:
        print("Échec de la capture d'un nouveau nuage de points.")
        continue

    # Transformation des points
    points_cp2 = cp.asarray(points_np2)
    points_cpT = cp.dot(rotation_matrix_cp, points_cp2.T).T + translation_cp

    # Filtrage des points
    filtered_points_cp = filter_points(points_cpT, centers, Zones_fixes)

    # Clustering
    if len(filtered_points_cp) > 0:
        filtered_points_np = cp.asnumpy(filtered_points_cp)
        cluster_labels2 = birch_model.fit_predict(filtered_points_np)
        num_clusters2 = len(set(cluster_labels2)) - (1 if -1 in cluster_labels2 else 0)
        clusters2 = [[] for _ in range(num_clusters2)]
        for i, label in enumerate(cluster_labels2):
            if label != -1 and label < num_clusters2:
                clusters2[label].append(filtered_points_np[i])

        # Enregistrement des clusters dans un fichier CSV  iteration_{c}_
        with open(f'clusters.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Cluster', 'X', 'Y', 'Z'])
            for cluster_idx, cluster in enumerate(clusters + clusters2):
                for point in cluster:
                    csvwriter.writerow([cluster_idx] + list(point))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Itération {c} a pris {elapsed_time:.2f} secondes.")
    c += 1

pipeline.stop()