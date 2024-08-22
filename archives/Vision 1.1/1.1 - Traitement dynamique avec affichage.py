'''
--------------------------------------------------------
Natanaël Jamet - contact : natanael.jamet@etu.unilim.fr |
Code datant de juillet 2024                             |
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
import matplotlib.pyplot as plt
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
# Démarrer le flux de la caméra
pipeline.start(config)


# Chargement du dictionnaire de marqueurs ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

taille_marqueur = 0.105  # Taille du marqueur en m

th1 = 0.05 #treshold pour le clustering de birch statique
th2 = 0.04 #treshold pour le clustering de birch dynamique

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


''' Cette fonction prends en argument : cluster, une liste contenant des coordonnees [x, y, z]
et renvoie [Lx, Ly, Lz], centre les longueurs et le centre d'une boite entourant au plus pres le nuage'''

def id_cube(cluster):
    """Calculer les dimensions et le centre d'une boîte englobante entourant un cluster."""
    cluster = np.array(cluster)
    xmin, xmax = np.min(cluster[:, 0]), np.max(cluster[:, 0])
    ymin, ymax = np.min(cluster[:, 1]), np.max(cluster[:, 1])
    zmin, zmax = np.min(cluster[:, 2]), np.max(cluster[:, 2])

    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    centre = [xmin + Lx / 2, ymin + Ly / 2, zmin + Lz / 2]

    return [Lx, Ly, Lz], centre


''' Cette fonction permet de prendre une photo avec une camera realsense connectee en USB et renvoie un nuage de point
max_distance permet de retirer les points plus loin que cette distance'''

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
    print('Capture d\'une image pour placer les captures dynamique dans le repere du marqueur')
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


def visualize_clusters(points_np, cluster_labels, num_clusters):
    """Visualize clusters with different colors."""
    if num_clusters == 0:
        print("Aucun cluster à visualiser.")
        return
    
    # Créer un nuage de points pour la visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Créer une liste de couleurs pour chaque point
    colors = np.zeros((len(points_np), 3))
    color_map = plt.get_cmap("tab20")  # Utiliser une palette de couleurs

    for i in range(num_clusters):
        cluster_points = points_np[np.array(cluster_labels) == i]
        colors[np.array(cluster_labels) == i] = color_map(i / num_clusters)[:3]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Afficher le nuage de points avec clusters colorisés
    o3d.visualization.draw_geometries([pcd], window_name="Clusters")

def visualize_boxes(pcd, zones):
    """Visualize bounding boxes around clusters in black."""
    if not zones:
        print("Aucune zone à visualiser.")
        return

    # Créer une liste de boîtes englobantes en tant que LineSet
    line_sets = []
    for center, dimensions in zones:
        box = o3d.geometry.LineSet()

        # Définir les points des coins de la boîte
        corners = np.array([
            [center[0] - dimensions[0] / 2, center[1] - dimensions[1] / 2, center[2] - dimensions[2] / 2],
            [center[0] + dimensions[0] / 2, center[1] - dimensions[1] / 2, center[2] - dimensions[2] / 2],
            [center[0] + dimensions[0] / 2, center[1] + dimensions[1] / 2, center[2] - dimensions[2] / 2],
            [center[0] - dimensions[0] / 2, center[1] + dimensions[1] / 2, center[2] - dimensions[2] / 2],
            [center[0] - dimensions[0] / 2, center[1] - dimensions[1] / 2, center[2] + dimensions[2] / 2],
            [center[0] + dimensions[0] / 2, center[1] - dimensions[1] / 2, center[2] + dimensions[2] / 2],
            [center[0] + dimensions[0] / 2, center[1] + dimensions[1] / 2, center[2] + dimensions[2] / 2],
            [center[0] - dimensions[0] / 2, center[1] + dimensions[1] / 2, center[2] + dimensions[2] / 2]
        ])

        # Définir les lignes reliant les coins
        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])

        box.points = o3d.utility.Vector3dVector(corners)
        box.lines = o3d.utility.Vector2iVector(lines)
        box.colors = o3d.utility.Vector3dVector(np.full((len(lines), 3), [0, 0, 0]))  # Couleur noire

        line_sets.append(box)

    # Afficher le nuage de points et les boîtes englobantes
    o3d.visualization.draw_geometries([pcd] + line_sets, window_name="Bounding Boxes")

def downsample_point_cloud_random(points_cp, factor):
    total_points = points_cp.shape[0]
    num_points_to_select = total_points // factor
    random_indices = cp.random.choice(total_points, num_points_to_select, replace=False)
    return points_cp[random_indices]


'''
------------------------------------------------------------------------------------------------------------------------
                                        - Recuperation et traitement de la scene statique - 
------------------------------------------------------------------------------------------------------------------------
'''

# Chemin du fichier .ply
ply_file_path = "combined_point_cloud.ply"

# Charger le nuage de points depuis le fichier .ply
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# Vérifier si le nuage de points est chargé correctement
if point_cloud.is_empty():
    print("Le nuage de points est vide.")
else:
    print("Le nuage de points a été chargé avec succès.")

# Convertir le nuage de points en un tableau NumPy de type float32
points_np = np.asarray(point_cloud.points, dtype=np.float32)
points_cp = cp.asarray(points_np)


#-----------------------------------------------------------------
#------- Clustering du nuage -------------------------------------
#-----------------------------------------------------------------

birch_model = Birch(n_clusters=None, threshold=th1)
cluster_labels = birch_model.fit_predict(cp.asnumpy(points_cp))

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


#------- Definition du KD tree 
centers = np.array([zone[0] for zone in Zones_fixes])
kdtree = cKDTree(centers)

#------- Definition du clustering BIRCH

birch_model = Birch(n_clusters=None, threshold=th2)


#------- Variables de recuperation des donnees
c = 1
Historique_zones = []
Historique_clusters = []

#-----------------------------------------------------------------
#------- Detection de la position de la camera  ------------------
#-----------------------------------------------------------------

point_clouds = []
color_images = []

#capture de l'image 
color_frame, depth_frame, color_image, gray, corners, ids = capture_image()
color_images.append(color_image)


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
#------- Boucle de recuperation et traitement dynamique ----------
#-----------------------------------------------------------------

print('Début de la boucle de traitement ')
while c <= 10:  # Changez à while True pour un traitement continu
    start_time = time.time()  # Démarrer le chronométrage

    # Capture d'un nouveau point
    
    points_np2 = capture_point_cloud(pipeline)
    if points_np2 is None:
        print("Échec de la capture d'un nouveau nuage de points.")
        continue

    # Transformation des points
    points_cp2 = cp.asarray(points_np2)
    points_cpT = cp.dot(rotation_matrix_cp, points_cp2.T).T + translation_cp
    points_cpT_downsampled = downsample_point_cloud_random(points_cpT, 8)

    #------- Filtrage avec KD -----------------------------
    # On passe par l'arbre pour identifier les boîtes dans lesquelles sont contenus les points
    points_np2 = cp.asnumpy(points_cpT_downsampled)
    distances, indices = kdtree.query(points_np2, k=1)
    
    filtered_mask = np.full(len(points_np2), True, dtype=bool)
    for i, point in enumerate(points_np2):
        idx = indices[i]
        centre, dimensions = Zones_fixes[idx]
        Lx, Ly, Lz = dimensions
        
        if not (centre[0] - Lx / 2 <= point[0] <= centre[0] + Lx / 2 and
                centre[1] - Ly / 2 <= point[1] <= centre[1] + Ly / 2 and
                centre[2] - Lz / 2 <= point[2] <= centre[2] + Lz / 2):
            filtered_mask[i] = True
        else:
            filtered_mask[i] = False
            
    # On retire grâce au mask les points qui sont dans une boîte
    filtered_points_np = points_np2[filtered_mask]

    # Permettre de récupérer les clusters sous forme de liste
    # et les boîtes sous forme de liste également 
    if len(filtered_points_np) > 0:
        cluster_labels2 = birch_model.fit_predict(filtered_points_np)
        num_clusters2 = len(set(cluster_labels2)) - (1 if -1 in cluster_labels2 else 0)
        clusters2 = [[] for _ in range(num_clusters2)]
        for i, label in enumerate(cluster_labels2):
            if label != -1 and label < num_clusters2:
                clusters2[label].append(filtered_points_np[i])

        #------- Calcul des bounding box (optionnel) ---------------------------------
        Zones_dynamique = []
        for cluster in clusters2:
            if len(cluster) > 0:
                dimensions, center = id_cube(cluster)
                Zones_dynamique.append((center, dimensions))

    Historique_zones.append(Zones_fixes + Zones_dynamique)
    Historique_clusters.append(clusters + clusters2)

    # Enregistrer les clusters dans un fichier CSV
    with open(f'iteration_{c}_clusters.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Cluster', 'X', 'Y', 'Z'])
        for cluster_idx, cluster in enumerate(clusters + clusters2):
            for point in cluster:
                csvwriter.writerow([cluster_idx] + list(point))

    # Visualiser les clusters et les boîtes
    if len(filtered_points_np) > 0:
        # Créer un nuage de points pour la visualisation
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points_np)
        
        # Visualiser les clusters
        visualize_clusters(filtered_points_np, cluster_labels2, num_clusters2)
        
        # Visualiser les boîtes
        visualize_boxes(pcd_filtered, Zones_fixes + Zones_dynamique)

    end_time = time.time()  # Fin du chronométrage
    elapsed_time = end_time - start_time
    print(f"Itération {c} a pris {elapsed_time:.2f} secondes.")
    
    c += 1

pipeline.stop()
