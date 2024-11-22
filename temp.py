from objects import *
from create import *
import time
import torch
import dearpygui.dearpygui as dpg

    
algo = "SGD"
start_program = False
initial_x = 0.0
initial_y = 0.0
fps = 60
coor_size = 10

def update_method(sender, app_data):
    global algo
    algo = app_data

def start_callback(sender, app_data):
    global start_program
    start_program = True

def update_initial_x(sender, app_data):
    global initial_x
    initial_x = app_data

def update_initial_y(sender, app_data):
    global initial_y
    initial_y = app_data

def update_fps(sender, app_data):
    global fps
    fps = app_data

def update_coor_size(sender, app_data):
    global coor_size
    coor_size = app_data
    
def setup_ui():
    with dpg.window(label="Settings", width=400, height=400, pos=(0, 0)):
        dpg.add_combo(label="Optimizer", items=["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"], default_value="SGD", callback=update_method)
        dpg.add_input_int(label="Coordinate Size", default_value=10, callback=update_coor_size)
        dpg.add_input_float(label="Initial X", default_value=0.0, callback=update_initial_x)
        dpg.add_input_float(label="Initial Y", default_value=0.0, callback=update_initial_y)
        dpg.add_input_int(label="FPS", default_value=60, callback=update_fps)
        dpg.add_button(label="Start", callback=start_callback)

def main():
    global algo, start_program, initial_x, initial_y, coor_size, fps
    viewer = Viewer()
    setup_ui()
    dpg.create_viewport(title='DearPyGui Controller', width=400, height=400)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    points = newpoints(mesh_size=coor_size, num_edge=200)
    values = np.asarray(generate_function(torch.tensor(points[:, 0]), torch.tensor(points[:, 1])))
    viewer.points = np.stack([points[:, 0], points[:, 1], values], axis=1)
    
    mesh_3d = Mesh(viewer.points, './resources/shaders/phong.vert', './resources/shaders/phong.frag').setup()
    ball_3d = Sphere(viewer.radius, './resources/shaders/phong_texture.vert', './resources/shaders/phong_texture.frag').setup()
    mesh_2d = Mesh2D(viewer.points, './resources/shaders/gouraud.vert', './resources/shaders/gouraud.frag').setup()
    ball_2d = Sphere2D(viewer.radius/2, viewer.points, './resources/shaders/gouraud.vert', './resources/shaders/gouraud.frag').setup()
    
    viewer.drawables = [mesh_3d, ball_3d]
    viewer.contour_map = [mesh_2d, ball_2d]
    viewer.fps = fps
    viewer.points_line, viewer.gradient = get_trajectory(initial_x, initial_y, coor_size, algo)

    line = Linear(viewer.points_line, './resources/shaders/phong.vert', './resources/shaders/phong.frag').setup()
    line_2d = Linear2D(viewer.points_line, viewer.points, './resources/shaders/gouraud.vert', './resources/shaders/gouraud.frag').setup()

    viewer.drawables.append(line)
    viewer.contour_map.append(line_2d)

    while not glfw.window_should_close(viewer.win):
        viewer.run()

if __name__ == '__main__':
    glfw.init()
    dpg.create_context()
    main()
    dpg.destroy_context()
    glfw.terminate()
