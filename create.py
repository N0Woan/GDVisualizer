import numpy as np
import random
import torch
import torch.optim as optim
import matplotlib
import numexpr as ne
from copy import deepcopy
from sympy import *
from libs import *
from matplotlib import cm
from scipy.interpolate import griddata
from sympy import symbols, sympify, lambdify
from libs.transform import *


def generate_function(x, y, num_gaussians=50, seed=93):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gaussians = torch.zeros_like(x)
    for _ in range(num_gaussians):
        center_x = torch.tensor(np.random.uniform(-5, 5), dtype=torch.float32)
        center_y = torch.tensor(np.random.uniform(-5, 5), dtype=torch.float32)
        amplitude = torch.tensor(np.random.uniform(-3, 3), dtype=torch.float32)
        width = torch.tensor(np.random.uniform(0.5, 2.0), dtype=torch.float32)
        gaussians += amplitude * torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))

    return gaussians


def get_trajectory(x, y, size=15, optimizer='SGD', lr=0.04, alpha=0.99, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, momentum=0):
    model = generate_function
    trajectory_x, trajectory_y, trajectory_z, gradient_x, gradient_y = [], [], [], [], []
    x = torch.tensor([[x]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[y]], dtype=torch.float32, requires_grad=True)

    if optimizer == 'SGD':
        optimizer = optim.SGD([x, y], lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam([x, y], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        optimizer = optim.RMSprop([x, y], lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == 'Adagrad':
        optimizer = optim.Adagrad([x, y], lr=lr, lr_decay=0, weight_decay=weight_decay, initial_accumulator_value=0, eps=eps)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW([x, y], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    step = 0
    while True:
        optimizer.zero_grad()

        x_pre = x.clone().detach()
        y_pre = y.clone().detach()
        z = model(x, y)

        loss = z

        loss.backward()

        gradient_x.append(x.grad.item())
        gradient_y.append(y.grad.item())

        optimizer.step()

        if (len(trajectory_z) > 1 and (abs(z.item() - trajectory_z[-1]) < 1e-6)) or (step > 3000):
            break
        if x.item() < -size/2 or x.item() > size/2 or y.item() < -size/2 or y.item() > size/2:
            break

        trajectory_x.append(x_pre.item())
        trajectory_y.append(y_pre.item())
        trajectory_z.append(z.item())
        step += 1

    return (np.stack((trajectory_x,trajectory_y,trajectory_z), axis=-1), np.stack((gradient_x, gradient_y), axis=-1))

def newcontour(points, num_levels):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xi = np.linspace(x.min(), x.max(), 200)  # Increase grid resolution
    yi = np.linspace(y.min(), y.max(), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    levels = np.linspace(zi.min(), zi.max(), num_levels)
    z_max = zi.max()
    z_min = zi.min()
    
    vertices, indices, colors = [], [], []
    idx = 0
    
    for level in levels:
        for i in range(len(xi)-1):
            for j in range(len(yi)-1):
                corners = [(xi[i, j], yi[i, j], zi[i, j]),
                            (xi[i+1, j], yi[i, j], zi[i, j]),
                            (xi[i, j+1], yi[i, j+1], zi[i, j+1]),
                            (xi[i+1, j+1], yi[i+1, j+1], zi[i+1, j+1])]
                contour_segment = []
                for k in range(4):
                    p1, p2 = corners[k], corners[(k+1) % 4]
                    if (p1[2] - level) * (p2[2] - level) < 0:  # Different signs
                        # Linear interpolation to find the intersection point
                        t = (level - p1[2]) / (p2[2] - p1[2])
                        x_interp = p1[0] + t * (p2[0] - p1[0])
                        y_interp = p1[1] + t * (p2[1] - p1[1])
                        z_interp = level
                        contour_segment.append([x_interp, y_interp, z_interp])
                if len(contour_segment) == 2:  # A contour line segment
                    for point in contour_segment:
                        vertices.append(point[:2] + [0])
                        indices.append(idx)
                        
                        normalized_z = (point[2] - z_min) / (z_max - z_min)
                        color = [normalized_z, 1 - normalized_z, 0.3 * (1 - normalized_z)]
                        colors.append(color)
                        
                        idx += 1

    vertices = np.array(vertices, dtype=np.float32)
    vertices[:, 2] = z_max + 1  # Offset z-coordinate to make lines visible if needed
    indices = np.array(indices, dtype=np.uint32)
    colors = np.array(colors, dtype=np.float32)

    return vertices, indices, colors

def newsphere(radius, sides):
    vertices, indices, color, texcoords = [], [], [], []
    for i in range(sides+1):
        for j in range(sides+1):
            theta = np.pi * i / sides
            phi = 2 * np.pi * j / sides
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            vertices += [[x, y, z]]
            color += [[1, 1, 1]]
            texcoords += [[j / sides, i / sides]]
            
    
    for j in range(sides):
        for i in range(sides):
            point = (sides+1)*j+i
            indices += [point, point+sides+1, point+1, point+sides+2]
            
    vertices = np.array(vertices, dtype=np.float32)
    color = np.array(color, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32) 
    texcoords = np.array(texcoords, dtype=np.float32)       
    return vertices, indices, color, texcoords

def newmesh(points):
    vertices, indices, color = points, [], []
    numEdge = sqrt(len(points)) - 1
    # geopotential_heights = points[:, 2]
    geopotential_heights = np.asarray([point[2] for point in points])

    min_height, max_height = geopotential_heights.min(), geopotential_heights.max()
    normalized_heights = (geopotential_heights - min_height) / (max_height - min_height)
    normalized_heights = normalized_heights.astype(float)**0.4
    def sigmoid(x, steepness=13.5):
        return 1 / (1 + np.exp(-steepness * (x - 0.5)))

    sigmoid_heights = sigmoid(normalized_heights)
    colormap = cm.RdYlGn.reversed()
    colors = [colormap(value)[:3] for value in sigmoid_heights]

    for j in range(numEdge):
        for i in range(numEdge):
            point = (numEdge+1)*j+i
            indices += [point, point, point+numEdge+1, point+1, point+numEdge+2]
            if i==numEdge-1:
                for k in range(numEdge):
                    indices += [(numEdge+1)*j+(numEdge-k-1), (numEdge+1)*j+(numEdge-k-1)]
    
    vertices = np.array(vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    
    return vertices, indices, colors

def newmesh_2d(points):
    vertices = np.array(points, dtype=np.float32)
    # vertices = points
    num_points = len(points)
    numEdge = int(sqrt(num_points)) - 1
    indices = []
    
    # Extract the z (height) values for the geopotential heights
    # geopotential_heights = vertices[:, 2]
    geopotential_heights = np.asarray([point[2] for point in points])

    # Normalize the geopotential heights to a 0-1 range
    min_height, max_height = geopotential_heights.min(), geopotential_heights.max()
    normalized_heights = (geopotential_heights - min_height) / (max_height - min_height)
    normalized_heights = normalized_heights.astype(float)**0.4  # Adjust with power scaling if needed

    # Apply sigmoid transformation to enhance middle contrast
    def sigmoid(x, steepness=13.5):
        return 1 / (1 + np.exp(-steepness * (x - 0.5)))
    
    sigmoid_heights = sigmoid(normalized_heights)

    # Create a discrete version of the RdYlGn colormap with 20 levels
    base_cmap = cm.RdYlGn.reversed()
    discrete_cmap = matplotlib.colors.ListedColormap(base_cmap(np.linspace(0, 1, 15)))

    # Map sigmoid-transformed heights to 20 discrete color levels
    colors = [discrete_cmap(int(value * 14))[:3] for value in sigmoid_heights]

    # Generate indices for the mesh grid
    for j in range(numEdge):
        for i in range(numEdge):
            point = (numEdge + 1) * j + i
            indices += [
                point, point + 1, point + numEdge + 1,
                point + 1, point + numEdge + 2, point + numEdge + 1
            ]

    # Convert colors and indices to arrays for OpenGL usage
    colors = np.array(colors, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    return vertices, indices, colors

def newpoints(mesh_size, num_edge):
    edge_size = mesh_size/num_edge
    points = [[-mesh_size/2+edge_size*j, mesh_size/2-edge_size*i, 0] for i in range(num_edge+1) for j in range(num_edge+1)]
    
    return np.asarray(points)

def func(points, f_str):
    # Define symbols and parse the expression as a string for numexpr
    x, y = symbols("x y")
    expression = sympify(f_str)
    expr_str = str(expression)

    # Convert points to a numpy array for efficient computation
    points = np.array(points, dtype=np.float64)

    # Evaluate the expression with numexpr for all points at once
    z_values = ne.evaluate(expr_str, local_dict={'x': points[:, 0], 'y': points[:, 1]})
    
    # Assign z values back to the points array
    points[:, 2] = z_values
    
    return list(points)

def compute_grad(x_val, y_val, f_str):
    # Define symbols
    x, y = symbols('x y')
    
    # Convert the function string into a symbolic expression
    expression = sympify(f_str)
    
    # Compute the partial derivatives (gradients) symbolically
    grad_x = diff(expression, x)  # Partial derivative with respect to x
    grad_y = diff(expression, y)  # Partial derivative with respect to y
    
    # Substitute the actual values of x and y into the derivatives
    grad_x_val = grad_x.subs({x: x_val, y: y_val})
    grad_y_val = grad_y.subs({x: x_val, y: y_val})
    
    # Convert symbolic results to float (if needed)
    return float(grad_x_val), float(grad_y_val)

def GD(f_str, mesh_size, initial_point=None, l_rate=0.02):
    if initial_point:
        x, y = initial_point
    else:
        x = random.uniform(-mesh_size/2+0.1, mesh_size/2-0.1)
        y = random.uniform(-mesh_size/2+0.1, mesh_size/2-0.1)
    points, gradient = func([[x, y, 0]], f_str), []

    while True:
        grad_x, grad_y = compute_grad(x, y, f_str)
        if (abs(grad_x)<0.0001) and (abs(grad_y)<0.0001):
            gradient.append([0, 0])
            break
        gradient.append(deepcopy([grad_x, grad_y]))
        x = x - l_rate * grad_x
        y = y - l_rate * grad_y
        points.append(func([[x, y, 0]], f_str)[0])

    return points, gradient

def GD_momen(f_str, mesh_size, initial_point=None, l_rate=0.05):
    if initial_point:
        x, y = initial_point
    else:
        x = random.uniform(-mesh_size/2+0.1, mesh_size/2-0.1)
        y = random.uniform(-mesh_size/2+0.1, mesh_size/2-0.1)
    points, gradient = func([[x, y, 0]], f_str), []
    last_grad_x=0
    last_grad_y=0

    while True:
        grad_x, grad_y = compute_grad(x, y, f_str)
        grad_x = last_grad_x * (1-l_rate) + grad_x * l_rate
        grad_y = last_grad_y * (1-l_rate) + grad_y * l_rate
        last_grad_x = grad_x
        last_grad_y = grad_y
        if (abs(grad_x)<0.0001) and (abs(grad_y)<0.0001):
            gradient.append([0, 0])
            break
        gradient.append(deepcopy([grad_x, grad_y]))
        x = x - l_rate * grad_x
        y = y - l_rate * grad_y
        points.append(func([[x, y, 0]], f_str)[0])

    return points, gradient