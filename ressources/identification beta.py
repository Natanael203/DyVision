import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import cv2
from rtree import index
import os

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
            return color_frame, depth_frame, color_image, gray, corners, ids
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


def save_screenshot(vis, filename):
    """Save a screenshot of the current visualization."""
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)

'''
----------------------------------------------------------------------------------------------
                  - Definition des fonctions de l'affichage  -
----------------------------------------------------------------------------------------------
'''
def create_bounding_box(center, dimensions):
    """Create a bounding box given the center and dimensions."""
    Lx, Ly, Lz = dimensions
    half_Lx, half_Ly, half_Lz = Lx / 2, Ly / 2, Lz / 2

    # Define 8 vertices of the bounding box
    points = np.array([
        [center[0] - half_Lx, center[1] - half_Ly, center[2] - half_Lz],
        [center[0] + half_Lx, center[1] - half_Ly, center[2] - half_Lz],
        [center[0] + half_Lx, center[1] + half_Ly, center[2] - half_Lz],
        [center[0] - half_Lx, center[1] + half_Ly, center[2] - half_Lz],
        [center[0] - half_Lx, center[1] - half_Ly, center[2] + half_Lz],
        [center[0] + half_Lx, center[1] - half_Ly, center[2] + half_Lz],
        [center[0] + half_Lx, center[1] + half_Ly, center[2] + half_Lz],
        [center[0] - half_Lx, center[1] + half_Ly, center[2] + half_Lz],
    ])

    # Define the 12 lines of the bounding box
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ])

    # Create a LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # Red color for bounding boxes
    
    return line_set

def visualize_point_cloud(points, bounding_boxes=None):
    """Visualize the point cloud and optionally bounding boxes using Open3D."""
    if points is not None and points.size > 0:
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        geometries = [pcd]
        
        # Add bounding boxes if provided
        if bounding_boxes:
            geometries.extend(bounding_boxes)
        
        # Display the point cloud and bounding boxes
        o3d.visualization.draw_geometries(geometries)
    else:
        print("No points to display.")



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
threshold       = 30    # Nombre de voisin minimal pour garder le point
# Parametres de l'optimisation :
K               = 1000 # Nombre de clusters
alpha=1
beta=1000


'''
----------------------------------------------------------------------------------------------
                                    - Code Principal  -
----------------------------------------------------------------------------------------------
'''
try:
    n = int(input("Combien de prises de vue souhaitez-vous effectuer ? "))

    point_clouds = []
    color_images = []

    for i in range(n):
        print(f"Prise de vue {i+1}/{n}")
        color_frame, depth_frame, color_image, gray, corners, ids = capture_image()
        color_images.append(color_image)
        
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

                filtered_points = filter_points_by_density(transformed_points, radius, threshold)
                point_clouds.append(filtered_points)  # Append filtered points
                
    if point_clouds:
        combined_points = np.concatenate(point_clouds, axis=0)
        
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(combined_points)
        labels = kmeans.labels_
        clusters = [[] for _ in range(K)]
        for point, label in zip(combined_points, labels):
            clusters[label].append(point.tolist())

        AABB_save = []
        for cluster in clusters:
            dimensions, center = id_cube(np.array(cluster))
            AABB_save.append((dimensions, center))
        
       
        beta_values = range(100, 2001, 100)

        for beta in beta_values:
            print(f"Optimisation avec beta = {beta}")

            continuer = True
            AABB=AABB_save.copy()
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
                        if i != j and not fusion[j]:  # Check for collision and ensure boxes aren't already merged
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

                                # Remove the merged bounding boxes from the R-tree
                                rtree_idx.delete(j, min_coords_j.tolist() + max_coords_j.tolist())  
                                rtree_idx.delete(i, min_coords_i.tolist() + max_coords_i.tolist())
                print('Génération : ', c)
                optimized_C = Cout(Nouvelles_boites, alpha, beta)
                print('Cout total après fusion : ', optimized_C, '\n')

                optimized_bounding_boxes = []
                for dimensions, center in Nouvelles_boites:
                    bbox = create_bounding_box(center, dimensions)
                    optimized_bounding_boxes.append(bbox)

                AABB = Nouvelles_boites
                if nf == 0:
                    continuer = False

            # Save screenshot of the visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined_points)
            vis.add_geometry(pcd)
            for bbox in optimized_bounding_boxes:
                vis.add_geometry(bbox)
            save_screenshot(vis, f"visualization_alpha_{alpha}_beta_{beta}.png")
            vis.destroy_window()

finally:
    # Stop the pipeline after use
    pipeline.stop()
