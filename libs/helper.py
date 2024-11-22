import numpy as np
from libs.transform import *


NORMALMAT_DEFAULT = np.identity(4, 'f') 
PROJECTION_DEFAULT = ortho(-0.5, 2.5, -0.5, 1.5, -1, 1)
# PROJECTION_DEFAULT = perspective(45, 1.5, 0.1, 100)
MODELVIEW_DEFAULT = np.identity(4, 'f')
SHININESS_DEFAULT = 50.0
MODE_DEFAULT = 2

# I_LIGHT_DEFAULT = np.array([
#     [0.5, 0.5, 0.5],  # diffuse
#     [1.0, 1.0, 1.0],  # specular
#     [0.5, 0.5, 0.5]  # ambient
# ], dtype=np.float32).T
# I_LIGHT_DEFAULT = np.array([
#     [0.8, 0.8, 0.8],  # diffuse
#     [1.0, 1.0, 1.0],  # specular
#     [0.8, 0.8, 0.8]  # ambient
# ], dtype=np.float32).T

# I_LIGHT_DEFAULT = np.array([
#     [0.9, 0.9, 0.9],  # diffuse (brighter light source)
#     [1.0, 1.0, 1.0],  # specular (sharp highlights)
#     [0.4, 0.4, 0.4]   # ambient (softer base light)
# ], dtype=np.float32).T
I_LIGHT_DEFAULT = np.array([
    [0.7, 0.7, 0.7],  # diffuse (more prominent light reflection)
    [0.9, 0.9, 0.9],  # specular (sharp highlights for realism)
    [0.3, 0.3, 0.3]   # ambient (subtle base lighting)
], dtype=np.float32).T


LIGHT_POS_DEFAULT = np.array([-0.5, -0.5, 2], dtype=np.float32)

# K_MATERIALS_DEFAULT = np.array([
#     [0.54,      0.89,       0.63],  # diffuse
#     [0.316228,	0.316228,	0.316228],  # specular
#     [0.135,	    0.2225,	    0.1575]  # ambient
# ], dtype=np.float32).T
# K_MATERIALS_DEFAULT = np.array([
#     [0.75164, 0.60648, 0.22648],  # diffuse
#     [0.628281, 0.555802, 0.366065],  # specular
#     [0.24725, 0.1995, 0.0745]  # ambient
# ], dtype=np.float32).T

K_MATERIALS_DEFAULT = np.array([
    [0.4, 0.7, 0.4],  # diffuse (matching greenish mesh color)
    [0.2, 0.2, 0.2],  # specular (subtle to blend with the mesh)
    [0.1, 0.3, 0.1]   # ambient (matching the diffuse tone for shadow regions)
], dtype=np.float32).T



# def generate_normals(vertices, indices):
#     normals = np.zeros((len(vertices), 3), dtype=np.float32)

#     for i in range(2, len(indices)):
#         v1 = vertices[indices[i-2]]
#         v2 = vertices[indices[i-1]]
#         v3 = vertices[indices[i]]
#         normal = np.cross(v2-v1, v3-v1)
#         length = np.linalg.norm(normal)
#         if length > 0:
#             normal /= length
#             normals[indices[i-2]] += normal
#             normals[indices[i-1]] += normal
#             normals[indices[i]] += normal
            
#     normals_len = np.linalg.norm(normals, axis=1)
#     normals_len[normals_len == 0] = 1e-7
#     normals /= normals_len.reshape(-1, 1)

#     return normals

def generate_normals(vertices, indices):
    # Initialize normals array with zeros
    normals = np.zeros((len(vertices), 3), dtype=np.float32)
    
    # Precompute vertex positions for each triangle in the strip
    v1 = vertices[indices[:-2]]
    v2 = vertices[indices[1:-1]]
    v3 = vertices[indices[2:]]

    # Calculate normals for each triangle in the strip
    triangle_normals = np.cross(v2 - v1, v3 - v1)
    lengths = np.linalg.norm(triangle_normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1e-7  # Prevent division by zero
    triangle_normals /= lengths  # Normalize each triangle normal

    # Accumulate the normalized triangle normals for each vertex in the strip
    np.add.at(normals, indices[:-2], triangle_normals)
    np.add.at(normals, indices[1:-1], triangle_normals)
    np.add.at(normals, indices[2:], triangle_normals)

    # Normalize the accumulated normals at each vertex
    normals_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_len[normals_len == 0] = 1e-7  # Avoid division by zero
    normals /= normals_len

    return normals