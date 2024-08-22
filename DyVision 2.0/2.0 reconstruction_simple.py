'''
--------------------------------------------------------
Natanaël Jamet - contact : natanael.jamet@etu.unilim.fr |
Code datant de juillet 2024                             |
Dernière modification :16/08/2024                       |
Version 2.0 du pack "DYVision"                            |
--------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


Le but de ce script est de scanner une scéne fixe et d'identifier les volumes primitifs. Ce code enregistre trois fichier: 1 .ply contenant le nuage combiné tel quel
 et 2 .Parquet contenant respectivement les point par appartenance aux boites et la liste des boites. 

Une camera realsense (une D435 a été utilisée pour develloper ce script) est utilisée pour recuperer les nuages de points
qui seront traités.

Un marqueur Aruco doit etre placé sans avoir été déplacé depuis le scan de la scène fixe, il a besoins d'être visible uniquement pour
le calcul du placement de la caméra par rapport à celui-ci.

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


les versions des librairies utilisées au moment de l'écriture du script sont les suivantes :

NumPy        : 1.26.2
Open3D       : 0.18.0
SciPy        : 1.13.0
Scikit-Learn : 1.5.1
OpenCV       : 4.5.5.64
Pandas       : 2.2.2
Tkinter      : 8.6

La version de open cv est importante : elle est necessaire pour la detection des Aruco, la derniere version en date ne fonctionnait pas
au besoin, il faut désinstaller et reinstaller la librairie via les commands suivantes sur windows en utilisant pip :

pip uninstall opencv-python opencv-python-headless
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64
pip install opencv-contrib-python==4.5.5.64


-------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------

En cas de changement de caméra ou de marqueur : verifier que les paramètres dist_coeffs, camera_matrix et taille_marqueur soient modifiés 

'''

import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import cv2
from rtree import index
import os
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt



'''
----------------------------------------------------------------------------------------------
           - Definition des fonctions de recuperation et preraitement -
----------------------------------------------------------------------------------------------
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
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            # Visualiser le marqueur ArUco dans plt
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            plt.imshow(color_image)
            plt.title('Marqueur ArUco détecté')
            plt.show()

            # Générer le nuage de points
            point_cloud, _ = depth_to_point_cloud(depth_frame, color_frame)
            visualize_point_cloud(point_cloud)

            Nadine=input('Voulez vous garder ce nuage ? [o/n]')
            if Nadine=='o' :
                return color_frame, depth_frame, color_image, gray, corners, ids
            else :
                print('Reprise de la photo')
        else:
            print("Aucun marqueur détecté. Veuillez reprendre la photo.")



def visualize_point_cloud(points):
    """Visualize the point cloud using Open3D."""
    if points is not None and points.size > 0:
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Display the point cloud
        o3d.visualization.draw_geometries([pcd])
    else:
        print("No points to display.")

def count_points_within_radius(points, center, radius):
    """Count the number of points within a sphere of given radius centered on a point."""
    distances = np.linalg.norm(points - center, axis=1)
    count = np.sum(distances <= radius)
    return count

def filter_points_by_density(points, radius, threshold):
    """Remove points with density below a threshold using KDTree."""
    tree = KDTree(points)
    filtered_points = []
    for i, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)
        density = len(indices)
        if density >= threshold:
            filtered_points.append(point)
    return np.array(filtered_points)

'''
----------------------------------------------------------------------------------------------
                  - Definition des fonctions de l'optimisation  -
----------------------------------------------------------------------------------------------
'''

def id_cube(cluster):
    """Calculer les dimensions et le centre d'une boîte englobante entourant un cluster."""
    cluster = np.array(cluster)
    xmin, xmax = np.min(cluster[:, 0]), np.max(cluster[:, 0])
    ymin, ymax = np.min(cluster[:, 1]), np.max(cluster[:, 1])
    zmin, zmax = np.min(cluster[:, 2]), np.max(cluster[:, 2])

    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    centre = [xmin + Lx / 2, ymin + Ly / 2, zmin + Lz / 2]

    return (Lx, Ly, Lz), centre 

def Cout(AABB, alpha, beta):
    """Calculer le coût total en fonction du nombre de boîtes et de leur volume."""
    N = len(AABB)
    V = sum(Lx * Ly * Lz for (Lx, Ly, Lz), _ in AABB)
    print('Nombre de boîtes : ', N, '\nVolume des boîtes : ', V)
    return alpha * N + beta * V

def Cout_sa(AABB, alpha, beta):
    N = len(AABB)
    V = sum(Lx * Ly * Lz for (Lx, Ly, Lz), _ in AABB)
    return alpha * N + beta * V

def build_rtree(aabbs):
    """Build an R-tree index for the bounding boxes."""
    p = index.Property()
    p.dimension = 3
    rtree_idx = index.Index(properties=p)

    for i, (dimensions, center) in enumerate(aabbs):
        Lx, Ly, Lz = dimensions
        min_coords = np.array(center) - np.array(dimensions) / 2
        max_coords = np.array(center) + np.array(dimensions) / 2
        rtree_idx.insert(i, min_coords.tolist() + max_coords.tolist())
    
    return rtree_idx


def fusion_boite(boite1, boite2):
    (Lx1, Ly1, Lz1), centre1 = boite1
    (Lx2, Ly2, Lz2), centre2 = boite2

    X1_max = centre1[0] + Lx1 / 2
    X1_min = centre1[0] - Lx1 / 2

    X2_max = centre2[0] + Lx2 / 2
    X2_min = centre2[0] - Lx2 / 2

    Y1_max = centre1[1] + Ly1 / 2
    Y1_min = centre1[1] - Ly1 / 2

    Y2_max = centre2[1] + Ly2 / 2
    Y2_min = centre2[1] - Ly2 / 2

    Z1_max = centre1[2] + Lz1 / 2
    Z1_min = centre1[2] - Lz1 / 2

    Z2_max = centre2[2] + Lz2 / 2
    Z2_min = centre2[2] - Lz2 / 2

    Xmax = max(X1_max, X2_max)
    Xmin = min(X1_min, X2_min)

    Ymax = max(Y1_max, Y2_max)
    Ymin = min(Y1_min, Y2_min)

    Zmax = max(Z1_max, Z2_max)
    Zmin = min(Z1_min, Z2_min)

    centre = [(Xmax + Xmin) / 2, (Ymax + Ymin) / 2, (Zmax + Zmin) / 2]
    Lx, Ly, Lz = abs(Xmax - Xmin), abs(Ymax - Ymin), abs(Zmax - Zmin)

    return (Lx, Ly, Lz), centre


def points_in_boxes(points, aabbs):
    """Assign points to their respective bounding boxes."""
    points_in_each_box = [[] for _ in range(len(aabbs))]
    assigned_points = np.zeros(points.shape[0], dtype=bool)  # Boolean array to track assigned points
    
    for i, bbox in enumerate(aabbs):
        in_box = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
        for idx in in_box:
            if not assigned_points[idx]:  # Only add the point if it hasn't been assigned to another box
                points_in_each_box[i].append(points[idx])
                assigned_points[idx] = True  # Mark point as assigned

    # Convert lists of points to NumPy arrays for each box
    points_in_each_box = [np.array(box_points) for box_points in points_in_each_box]
    return points_in_each_box


def optimisation(combined_points, alpha, beta, K):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(combined_points)
    labels = kmeans.labels_
    clusters = [[] for _ in range(K)]
    for point, label in zip(combined_points, labels):
        clusters[label].append(point.tolist())

    bounding_boxes = []
    AABB = []
    for cluster in clusters:
        dimensions, center = id_cube(np.array(cluster))
        AABB.append((dimensions, center))
        bbox = create_bounding_box(center, dimensions)
        bounding_boxes.append(bbox)
    C = Cout_sa(AABB, alpha, beta)

    rtree_idx = build_rtree(AABB)

    continuer = True 
    c = 0
    while continuer:
        c += 1
        rtree_idx = build_rtree(AABB)
        Nouvelles_boites = AABB.copy()
        fusion = [False] * len(AABB)
        nf = 0
        for i, boite_i in enumerate(AABB):
            dimensions_i, center_i = boite_i
            min_coords_i = np.array(center_i) - np.array(dimensions_i) / 2
            max_coords_i = np.array(center_i) + np.array(dimensions_i) / 2

            possible_collisions = list(rtree_idx.intersection(min_coords_i.tolist() + max_coords_i.tolist()))

            fusion = [False for _ in range(len(AABB))]

            for j in possible_collisions:
                if i != j and not fusion[j]:
                    boite_j = AABB[j]
                    dimensions_j, center_j = boite_j
                    min_coords_j = np.array(center_j) - np.array(dimensions_j) / 2
                    max_coords_j = np.array(center_j) + np.array(dimensions_j) / 2

                    cout_avant = Cout_sa(Nouvelles_boites, alpha, beta)

                    Jean = Nouvelles_boites.copy()
                    Jean = [lst for lst in Jean if lst != boite_j]
                    Jean = [lst for lst in Jean if lst != boite_i]

                    nouvelle_boite = fusion_boite(boite_i, boite_j)
                    Jean.append(nouvelle_boite)

                    cout_apres = Cout_sa(Jean, alpha, beta)

                    if cout_avant >= cout_apres:
                        Nouvelles_boites = Jean.copy()
                        fusion[i], fusion[j] = True, True
                        nf += 1

                        rtree_idx.delete(j, min_coords_j.tolist() + max_coords_j.tolist())  
                        rtree_idx.delete(i, min_coords_i.tolist() + max_coords_i.tolist())
        print('Generation: ', c)
        optimized_C = Cout_sa(Nouvelles_boites, alpha, beta)

        AABB = Nouvelles_boites
        if nf == 0:
            continuer = False
            optimized_bounding_boxes = []

    optimized_bounding_boxes = []
    for dimensions, center in Nouvelles_boites:
        bbox = create_bounding_box(center, dimensions)
        optimized_bounding_boxes.append(bbox)




    return optimized_bounding_boxes

'''
----------------------------------------------------------------------------------------------
                  - Definition des fonctions de l'interface  -
----------------------------------------------------------------------------------------------
'''


def interactive_box_selection(points, aabbs):
    def on_mouse_over(event):
        nonlocal highlighted_index
        index = listbox.nearest(event.y)
        if highlighted_index is not None and highlighted_index != index:
            if is_selected[highlighted_index]:
                aabbs[highlighted_index].color = (1, 0, 0)  # Red if selected
            else:
                aabbs[highlighted_index].color = (0, 0, 0)  # Black otherwise
            vis.update_geometry(aabbs[highlighted_index])

        highlighted_index = index
        aabbs[highlighted_index].color = (0, 1, 0)  # Green for highlight
        vis.update_geometry(aabbs[highlighted_index])

    def select_box():
        nonlocal highlighted_index
        index = listbox.curselection()[0]
        if not is_selected[index]:
            is_selected[index] = True
            selected_indices.append(index)
            deleted_history.append(index)
            selected_box = aabbs[index]
            selected_box.color = (1, 0, 0)  # Red for selection
            vis.update_geometry(selected_box)

    def undo_last_deletion():
        if deleted_history:
            index = deleted_history.pop()
            is_selected[index] = False
            selected_indices.remove(index)
            restored_box = aabbs[index]
            restored_box.color = (0, 0, 0)  # Black for restored
            vis.update_geometry(restored_box)

    def finish():
        root.quit()  # This will stop the Tkinter main loop

    def update_scene():
        vis.poll_events()
        vis.update_renderer()
        root.after(100, update_scene)

    selected_indices = []
    deleted_history = []

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(pcd)

    for box in aabbs:
        box.color = (0, 0, 0)  # Black color
        vis.add_geometry(box)

    highlighted_index = None
    is_selected = [False] * len(aabbs)

    root = tk.Tk()
    root.title("Selectable Boxes")

    listbox = tk.Listbox(root)
    for i, box in enumerate(aabbs):
        listbox.insert(tk.END, f"Box {i}")
    listbox.pack()

    listbox.bind('<Motion>', on_mouse_over)

    select_button = tk.Button(root, text="Supprimer", command=select_box)
    select_button.pack()

    undo_button = tk.Button(root, text="Annuler", command=undo_last_deletion)
    undo_button.pack()

    finish_button = tk.Button(root, text="Terminer", command=finish)
    finish_button.pack()

    root.after(100, update_scene)
    root.mainloop()

    vis.destroy_window()
    
    # Retrieve remaining boxes that are not selected
    remaining_indices = [i for i, selected in enumerate(is_selected) if not selected]
    
    return selected_indices, remaining_indices


    def select_box():
        nonlocal highlighted_index
        index = listbox.curselection()[0]
        is_selected[index] = True
        selected_indices.append(index)
        selected_box = aabbs[index]
        selected_box.color = (1, 0, 0)  # Red for selection
        vis.update_geometry(selected_box)
        
    def finish():
        root.quit()  # This will stop the Tkinter main loop

    def update_scene():
        vis.poll_events()
        vis.update_renderer()
        root.after(100, update_scene)

    selected_indices = []

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(pcd)

    for box in aabbs:
        box.color = (0, 0, 0)  # Black color
        vis.add_geometry(box)

    highlighted_index = None
    is_selected = [False] * len(aabbs)

    root = tk.Tk()
    root.title("Selectable Boxes")

    listbox = tk.Listbox(root)
    for i, box in enumerate(aabbs):
        listbox.insert(tk.END, f"Box {i}")
    listbox.pack()

    listbox.bind('<Motion>', on_mouse_over)
    button = tk.Button(root, text="Suppimer", command=select_box)
    button.pack()

    button_finish = tk.Button(root, text="Terminer", command=finish)
    button_finish.pack()

    root.after(100, update_scene)
    root.mainloop()

    vis.destroy_window()
    
    # Retrieve remaining boxes that are not selected
    remaining_indices = [i for i, selected in enumerate(is_selected) if not selected]
    
    return selected_indices, remaining_indices

def filter_points_in_boxes(points, aabbs, indices):
    """Filter points inside the bounding boxes."""
    filtered_points = []
    all_points = []
    for i in indices:
        bbox = aabbs[i]
        in_box = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
        filtered_points.extend(points[in_box])
        all_points
    return np.array(filtered_points)

'''
----------------------------------------------------------------------------------------------
                  - Definition des fonctions de l'affichage  -
----------------------------------------------------------------------------------------------
'''




def visualize_point_cloud(points, bounding_boxes=None, color_boxes=False):
    """Visualize the point cloud and optionally bounding boxes using Open3D."""
    if points is not None and points.size > 0:
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        geometries = [pcd]
        
        # Add bounding boxes if provided
        if bounding_boxes:
            if color_boxes:
                for bbox in bounding_boxes:
                    bbox.color = (0, 0, 0)  # Black color for the second set of boxes
            geometries.extend(bounding_boxes)
        
        # Display the point cloud and bounding boxes
        o3d.visualization.draw_geometries(geometries)
    else:
        print("No points to display.")

def create_bounding_box(center, dimensions):
    """Create a bounding box given the center and dimensions."""
    Lx, Ly, Lz = dimensions
    half_Lx, half_Ly, half_Lz = Lx / 2, Ly / 2, Lz / 2
    cx, cy, cz = center

    min_bound = (cx - half_Lx, cy - half_Ly, cz - half_Lz)
    max_bound = (cx + half_Lx, cy + half_Ly, cz + half_Lz)

    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return aabb



def get_selected_boxes_and_points(combined_points, selected_boxes_indices, bounding_boxes):
    """
    Prend en argument les points combinés et les indices des boîtes sélectionnées,
    et renvoie une liste des boîtes contenant les centres et dimensions ainsi qu'une
    liste contenant les points appartenant à chaque boîte sélectionnée.

    :param combined_points: np.array de forme (N, 3), les points combinés.
    :param selected_boxes_indices: Liste des indices des boîtes sélectionnées.
    :param bounding_boxes: Liste des boîtes englobantes (instances de AxisAlignedBoundingBox d'Open3D).

    :return: selected_bounding_boxes, points_per_selected_box
             selected_bounding_boxes : Liste des boîtes contenant les centres et dimensions.
             points_per_selected_box : Liste contenant des listes de points appartenant à chaque boîte.
    """
    
    selected_bounding_boxes = []
    points_per_selected_box = []
    
    for index in selected_boxes_indices:
        bbox = bounding_boxes[index]
        min_bound = np.asarray(bbox.min_bound)
        max_bound = np.asarray(bbox.max_bound)
        
        # Calculer le centre et les dimensions de la boîte sélectionnée
        center = (min_bound + max_bound) / 2
        dimensions = max_bound - min_bound
        selected_bounding_boxes.append((center, dimensions))
        
        # Filtrer les points appartenant à la boîte actuelle
        in_box = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(combined_points))
        points_in_box = combined_points[in_box]
        points_per_selected_box.append(points_in_box)
    
    return selected_bounding_boxes, points_per_selected_box

'''
----------------------------------------------------------------------------------------------
                                    - Initialisation et Definition des parametres  -
----------------------------------------------------------------------------------------------
'''
# Initialiastion du pipeline realsense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 

# Start streaming
pipeline.start(config)
time.sleep(1)  # Pour stabiliser avant la premiere prise de vue


# Chargement du dictionnaire de marqueurs ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()


# Matrice de la caméra et coefficients de distorsion (à ajuster selon la caméra)
camera_matrix   = np.array([[615.0, 0.0, 320.0],[0.0, 615.0, 240.0],[0.0, 0.0, 1.0]])
dist_coeffs     = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

taille_marqueur = 0.105 # Taille du marqueur en m
# Parametres du filtrage par densité : 
radius          = 0.05  # Rayon du voisinage d'un point en m
threshold       = 150    # Nombre de voisin minimal pour garder le point
# Parametres de l'optimisation :
K               = 1000  # Nombre de clusters
alpha           = 1
beta            = 100
beta2           = 1000



'''
----------------------------------------------------------------------------------------------
                                    - Code Principal  -
----------------------------------------------------------------------------------------------
'''
try:
    #---------------------------------------------------------------------------------------
    # Récupératioin des nuages
    #---------------------------------------------------------------------------------------

    print('Début de la prise des photos pour reconstruire la scéne statique. \nEntrez le nombre de Points de vue souhaités. \nAssurez-vous que le marqueur soit visible à chaque prise de vue. ')
    n = int(input("\nCombien de prises de vue souhaitez-vous effectuer ? "))
    point_clouds = []
    color_images = []
    for i in range(n):
        print(f"Prise de vue {i+1}/{n}")
        color_frame, depth_frame, color_image, gray, corners, ids = capture_image()
        color_images.append(color_image)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, taille_marqueur, camera_matrix, dist_coeffs)
            for j in range(len(ids)):
                cv2.aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvecs[j], tvecs[j], 0.1)

                position = tvecs[j].flatten()
                rotation_matrix, _ = cv2.Rodrigues(rvecs[j])

                point_cloud, _ = depth_to_point_cloud(depth_frame, color_frame)
                rotation_matrix = rotation_matrix.T
                translation = -rotation_matrix @ position
                transformed_points = (rotation_matrix @ point_cloud.T).T + translation

                filtered_points = filter_points_by_density(transformed_points, radius, threshold)        
                point_clouds.append(filtered_points)
    
    #---------------------------------------------------------------------------------------
    # Combinaison et enregitrement des nuages
    #---------------------------------------------------------------------------------------
    if point_clouds:
        combined_points = np.concatenate(point_clouds, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        
        output_file = "combined_point_cloud.ply"
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"nuage sauvegardé dans : {output_file}")
        print('Fin de la récuperation des points de vue - Début de l\'optimisation')

        #---------------------------------------------------------------------------------------
        # Optimisation 
        #---------------------------------------------------------------------------------------   
        optimized_bounding_boxes = optimisation(combined_points, alpha, beta, K)

        print('L\'optimisation est finie. Veuillez sélectionner les boîtes à supprimer dans l\'interface graphique')

        #---------------------------------------------------------------------------------------
        # Sélection des boîtes à garder
        #---------------------------------------------------------------------------------------   
        selected_boxes, remaining_boxes_indices = interactive_box_selection(combined_points, optimized_bounding_boxes)
        
        #---------------------------------------------------------------------------------------
        # Obtenir les boîtes sélectionnées et les points correspondants
        #---------------------------------------------------------------------------------------
        selected_bounding_boxes, points_per_selected_box = get_selected_boxes_and_points(combined_points, remaining_boxes_indices, optimized_bounding_boxes)
        
        # Visualiser les points restant après sélection
        remaining_points = filter_points_in_boxes(combined_points, optimized_bounding_boxes, remaining_boxes_indices)
        visualize_point_cloud(remaining_points, [optimized_bounding_boxes[i] for i in remaining_boxes_indices])

        #---------------------------------------------------------------------------------------
        # Exportation des données au format Parquet
        #---------------------------------------------------------------------------------------   
        # Exportation des points par boîte
        points_data = []
        for i, points in enumerate(points_per_selected_box):
            for point in points:
                points_data.append([i, point[0], point[1], point[2]])
        
        points_df = pd.DataFrame(points_data, columns=['BoxIndex', 'X', 'Y', 'Z'])
        points_df.to_parquet('points_per_selected_box.parquet', index=False)
        
        # Exportation des boîtes avec leurs dimensions et centres
        boxes_data = []
        for i, (center, dimensions) in enumerate(selected_bounding_boxes):
            boxes_data.append([i, center[0], center[1], center[2], dimensions[0], dimensions[1], dimensions[2]])

        boxes_df = pd.DataFrame(boxes_data, columns=['BoxIndex', 'CenterX', 'CenterY', 'CenterZ', 'DimX', 'DimY', 'DimZ'])
        boxes_df.to_parquet('selected_bounding_boxes.parquet', index=False)

        print("Les fichiers 'points_per_selected_box.parquet' et 'selected_bounding_boxes.parquet' ont été sauvegardés correctement.")

finally:
    pipeline.stop()
