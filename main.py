from objects import *
from create import *
import time
import torch
import dearpygui.dearpygui as dpg

    
algo = "SGD"
start_program = False
lr_rate = 0.04
alpha = 0.99
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
weight_decay = 0
momentum = 0
initial_x = -0.35
initial_y = -1.5
speed = 1
fps = 60
coor_size = 15
seed = 93
list_algo = ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW", "all"]

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
    
def update_lr_rate(sender, app_data):
    global lr_rate
    lr_rate = app_data

def update_beta1(sender, app_data):
    global beta1
    beta1 = app_data

def update_beta2(sender, app_data):
    global beta2
    beta2 = app_data
    
def update_eps(sender, app_data):
    global eps
    eps = app_data
    
def update_weight_decay(sender, app_data):
    global weight_decay
    weight_decay = app_data

def update_alpha(sender, app_data):
    global alpha
    alpha = app_data

def update_momentum(sender, app_data):
    global momentum
    momentum = app_data

def update_speed(sender, app_data):
    global speed
    speed = app_data

def update_seed(sender, app_data):
    global seed
    seed = app_data

def setup_ui():
    with dpg.window(label="Settings", width=400, height=400, pos=(0, 0)):
        dpg.add_combo(label="Optimizer", items=list_algo, default_value="SGD", callback=update_method)
        dpg.add_input_float(label="Learning Rate", default_value=0.04, callback=update_lr_rate)
        dpg.add_input_float(label="alpha", default_value=0.99, callback=update_alpha)
        with dpg.group(horizontal=True):
            dpg.add_input_float(label="Beta 1", default_value=0.9, width=100, callback=update_beta1)
            dpg.add_input_float(label="Beta 2", default_value=0.999, width=100, callback=update_beta2)
        dpg.add_input_float(label="Epsilon", default_value=1e-8, callback=update_eps)
        dpg.add_input_float(label="Weight Decay", default_value=0, callback=update_weight_decay)
        dpg.add_input_float(label="momentum", default_value=0, callback=update_momentum)
        dpg.add_input_int(label="Coordinate Size", default_value=10, callback=update_coor_size)
        dpg.add_input_float(label="Initial X", default_value=-0.35, callback=update_initial_x)
        dpg.add_input_float(label="Initial Y", default_value=-1.5, callback=update_initial_y)
        dpg.add_input_int(label="FPS", default_value=60, callback=update_fps)
        dpg.add_input_float(label="Speed", default_value=1, callback=update_speed)
        dpg.add_button(label="Start", callback=start_callback)

def main():
    global algo, start_program, initial_x, initial_y, coor_size, fps, lr_rate, speed, alpha, beta1, beta2, eps, weight_decay, momentum, seed
    viewer = Viewer()
    setup_ui()
    dpg.create_viewport(title='DearPyGui Controller', width=400, height=400)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    first_time = True
    points = newpoints(mesh_size=coor_size, num_edge=200)
    values = np.asarray(generate_function(torch.tensor(points[:, 0]), torch.tensor(points[:, 1]), seed=seed))
    viewer.points = np.stack([points[:, 0], points[:, 1], values], axis=1)

    while not glfw.window_should_close(viewer.win):
        dpg.render_dearpygui_frame()
        if first_time or start_program==True:
            first_time = False
            start_program = False
            viewer.mesh_3d = Mesh(viewer.points, './resources/shaders/phong.vert', './resources/shaders/phong.frag').setup()
            viewer.mesh_2d = Mesh2D(viewer.points, './resources/shaders/gouraud.vert', './resources/shaders/gouraud.frag').setup()
            viewer.contour_map = Contour(viewer.points, 20, './resources/shaders/gouraud.vert', './resources/shaders/gouraud.frag').setup()
            viewer.ball_3d = []
            viewer.ball_2d = []
            viewer.line_3d = []
            viewer.line_2d = []
            viewer.points_line = []
            viewer.gradient = []
            viewer.fps = fps
            viewer.length = 0
            viewer.num_frame = 0
            viewer.speed = speed
            
            list_alg = ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"] if algo=="all" else [algo]
            viewer.rot_matrix = [identity() for _ in range(len(list_alg))]

            for alg in list_alg:
                viewer.ball_3d.append(Sphere(viewer.radius, alg, './resources/shaders/phong_texture.vert', './resources/shaders/phong_texture.frag').setup())
                viewer.ball_2d.append(Sphere2D(viewer.radius/2, viewer.points, alg, './resources/shaders/gouraud.vert', './resources/shaders/gouraud.frag').setup())
                points_line, gradient = get_trajectory(initial_x, initial_y, coor_size, alg, lr_rate, alpha, beta1, beta2, eps, weight_decay, momentum)
                viewer.points_line.append(points_line)
                viewer.gradient.append(gradient)
                if len(points_line) > viewer.length:
                    viewer.length = len(points_line)
                viewer.line_3d.append(Linear(points_line, alg, './resources/shaders/phong.vert', './resources/shaders/phong.frag').setup())
                viewer.line_2d.append(Linear2D(points_line, viewer.points, alg, './resources/shaders/gouraud.vert', './resources/shaders/gouraud.frag').setup())

        viewer.run()

if __name__ == '__main__':
    glfw.init()
    dpg.create_context()
    main()
    dpg.destroy_context()
    glfw.terminate()
