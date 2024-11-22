import OpenGL.GL as GL
import numpy as np
import glfw
import time
from itertools import cycle
from time import sleep
from sympy import *
from libs import *
from create import *
import dearpygui.dearpygui as dpg


color_algo = {
    "SGD": [0.5, 0.5, 0],
    "Adam": [0.5, 0, 0.5],
    "RMSprop": [0, 0.5, 0.5],
    "Adagrad": [0.5, 0.5, 0.5],
    "AdamW": [1, 1, 1],
}

texture_algo = {
    "SGD": '.\\resources\\textures\\earth.jpg',
    "Adam": '.\\resources\\textures\\sun.jpg',
    "RMSprop": '.\\resources\\textures\\saturn.jpg',
    "Adagrad": '.\\resources\\textures\\mars.jpg',
    "AdamW": '.\\resources\\textures\\moon.jpg',
}

class Object3D(object):
    def __init__(self, vert_shader, frag_shader):
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

        self.normalMat = NORMALMAT_DEFAULT
        self.projection = PROJECTION_DEFAULT
        self.modelview = MODELVIEW_DEFAULT

        self.I_light = I_LIGHT_DEFAULT
        self.light_pos = LIGHT_POS_DEFAULT

        self.K_materials = K_MATERIALS_DEFAULT        
        self.shininess = SHININESS_DEFAULT
        self.mode = MODE_DEFAULT

        self.vertices = []
        self.colors = []
        self.indices = []
        self.texcoords = []
        self.normals = []

    def prepare_vao(self):
        self.vao = VAO()
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

    def prepare_uma(self):
        self.uma.upload_uniform_matrix4fv(self.normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(self.projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(self.modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(self.I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(self.light_pos, 'light_pos')

        self.uma.upload_uniform_matrix3fv(self.K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(self.mode, 'mode')

    def setup(self):
        self.prepare_vao()
        GL.glUseProgram(self.shader.render_idx)
        self.prepare_uma()
        
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

class Mesh(Object3D):
    def __init__(self, points, vert_shader, frag_shader):
        super().__init__(vert_shader, frag_shader)
        self.vertices, self.indices, self.colors = newmesh(points)
        self.normals = generate_normals(self.vertices, self.indices)

class Linear(Object3D):
    def __init__(self, points, algo, vert_shader, frag_shader):
        super().__init__(vert_shader, frag_shader)
        self.vertices = points
        self.indices, self.colors = [], []
        self.indices = [i for i in range(len(self.vertices))]
        self.colors = [color_algo[algo] for i in range(len(self.vertices))]
        
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)    
        
        self.normals = generate_normals(self.vertices, self.indices)

class Sphere(Object3D):
    def __init__(self, radius, algo, vert_shader, frag_shader):   
        super().__init__(vert_shader, frag_shader)
        self.algo = algo
        self.radius = radius
        self.vertices, self.indices, self.colors, self.texcoords = newsphere(self.radius, 30)     
        self.normals = generate_normals(self.vertices, self.indices)
    
    def prepare_vao(self):
        self.vao = VAO()
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

    def prepare_uma(self):
        self.uma.upload_uniform_matrix4fv(self.normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(self.projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(self.modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(self.I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(self.light_pos, 'light_pos')

        self.uma.upload_uniform_matrix3fv(self.K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(self.mode, 'mode')

        # Preload texture and store its ID
        texture_file = texture_algo[self.algo]
        self.texture_id = self.uma.setup_texture(self.algo, texture_file)

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view @ model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)

        # Bind the preloaded texture
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

class Contour(Object3D):
    def __init__(self, points, num_levels, vert_shader, frag_shader):
        super().__init__(vert_shader, frag_shader)
        points = np.asarray(points)
        self.vertices, self.indices, self.colors = newcontour(points, num_levels)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()
        self.z_min, self.z_max = z.min(), z.max()

    def prepare_vao(self):
        self.vao = VAO()
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

    def prepare_uma(self):
        projection = ortho(self.x_min, self.x_max, self.y_min, self.y_max, -10, 10)
        modelview = np.identity(4, 'f')
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

    def draw(self):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glPointSize(20.0)
        GL.glDrawElements(GL.GL_LINES,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

        GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)       
        GL.glPointSize(1.0)
        
class Mesh2D(Object3D):
    def __init__(self, points, vert_shader, frag_shader):
        super().__init__(vert_shader, frag_shader)
        points = np.asarray(points)
        self.vertices, self.indices, self.colors = newmesh_2d(points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()
        self.z_min, self.z_max = z.min(), z.max()

    def prepare_vao(self):
        self.vao = VAO()
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

    def prepare_uma(self):
        projection = ortho(self.x_min, self.x_max, self.y_min, self.y_max, -10, 10)
        modelview = np.identity(4, 'f')
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
    
    def draw(self, model=None):
        GL.glUseProgram(self.shader.render_idx)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

class Linear2D(Object3D):
    def __init__(self, points, mesh_points, algo, vert_shader, frag_shader):
        super().__init__(vert_shader, frag_shader)
        self.vertices = points
        self.indices, self.colors = [], []
        self.indices = [i for i in range(len(self.vertices))]
        self.colors = [color_algo[algo] for i in range(len(self.vertices))]
        
        mesh_points = np.asarray(mesh_points)
        x, y, z = mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2]
        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()
        self.z_min, self.z_max = z.min(), z.max()
        
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertices[:,2] += 1
        self.colors = np.array(self.colors, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)    

    def prepare_vao(self):
        self.vao = VAO()
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

    def prepare_uma(self):
        projection = ortho(self.x_min, self.x_max, self.y_min, self.y_max, -10, 10)
        modelview = np.identity(4, 'f')
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
    
    def draw(self, model=None):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glPointSize(15.0)
        # GL.glDrawElements(GL.GL_LINES,
        #                   self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        GL.glDrawElements(GL.GL_LINE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)       
        GL.glPointSize(1.0)
        
class Sphere2D(Object3D):
    def __init__(self, radius, mesh_points, algo, vert_shader, frag_shader):   
        super().__init__(vert_shader, frag_shader)
        self.radius = radius
        self.vertices, self.indices, self.colors, self.texcoords = newsphere(self.radius, 15)     
        self.colors = np.array([color_algo[algo] for i in range(len(self.vertices))], dtype=np.float32)
        mesh_points = np.asarray(mesh_points)
        x, y, z = mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2]
        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()
        self.z_min, self.z_max = z.min(), z.max()

    def prepare_vao(self):
        self.vao = VAO()
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

    def prepare_uma(self):
        projection = ortho(self.x_min, self.x_max, self.y_min, self.y_max, -10, 10)
        self.translate = np.identity(4)
        self.translate[2][3] = 2
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
    
    def draw(self, model=None):
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(self.translate @ model, 'modelview', True)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        

class Viewer:
    def __init__(self, width=1500, height=1000):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # version hints: create GL windows with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0, 0, 0, 0.1)
        # GL.glEnable(GL.GL_CULL_FACE)   # enable backface culling (Exercise 1)
        # GL.glFrontFace(GL.GL_CCW) # GL_CCW: default

        GL.glEnable(GL.GL_DEPTH_TEST)  # enable depth test (Exercise 1)
        GL.glDepthFunc(GL.GL_LESS)   # GL_LESS: default

        # initially empty list of object to draw
        self.mesh_3d = None
        self.mesh_2d = None
        self.ball_3d = []
        self.ball_2d = []
        self.line_3d = []
        self.line_2d = []
        self.points_line = []
        self.gradient = []
        self.rot_matrix = []
        self.fps = None

        self.radius = 0.2
        self.num_frame = 0
        self.fps = 60
        self.points = None
        self.length = 0

        self.contour_map = None
        self.speed = 1

    def run(self):
        """ Main render loop for this OpenGL windows """
        start_time = time.time()
        # clear draw buffer
        if int(self.num_frame) >= self.length:
            self.num_frame = 0
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        win_size = glfw.get_window_size(self.win)
        win_width, win_height = win_size
        view = self.trackball.view_matrix()
        projection = self.trackball.projection_matrix((win_width // 2, win_height))

        GL.glViewport(0, 0, win_width - win_height // 2, win_height)
        self.mesh_3d.draw(projection, view, None)
        GL.glViewport(win_width - win_height // 2, 0, win_height // 2, win_height//2)
        self.mesh_2d.draw(None)
        # Render the right upper half of the window (Viewport 3)
        GL.glViewport(win_width - win_height // 2, win_height//2, win_height // 2, win_height//2)
        self.contour_map.draw()

        for i in range(len(self.ball_3d)):
            frame = min(int(self.num_frame), len(self.points_line[i])-1)
            normal_vec = normalized(vec(-self.gradient[i][frame][0], -self.gradient[i][frame][1], 1))
            trans_matrix = translate(vec((self.points_line[i][frame][0], self.points_line[i][frame][1], self.points_line[i][frame][2])) + self.radius * normal_vec)
            
            if int(self.num_frame)==0 or int(self.num_frame)>=len(self.points_line[i]):
                self.rot_matrix[i] = identity()
            elif int(self.num_frame)!=int(self.num_frame-self.speed):
                velo_vec = vec((self.points_line[i][frame][0] - self.points_line[i][frame-1][0], 
                                self.points_line[i][frame][1] - self.points_line[i][frame-1][1], 
                                self.points_line[i][frame][2] - self.points_line[i][frame-1][2]))
                d = np.linalg.norm(velo_vec)
                rot_axis = normalized(np.cross(normal_vec, velo_vec + self.radius * normal_vec))
                alpha = 360*d/(self.radius*2*np.pi)
                self.rot_matrix[i] = rotate(rot_axis, alpha) @ self.rot_matrix[i]
                
            model = trans_matrix @ self.rot_matrix[i]
            
            # draw our scene objects
            GL.glViewport(0, 0, win_width - win_height // 2, win_height)
            self.ball_3d[i].draw(projection, view, model)
            self.line_3d[i].draw(projection, view, model)

            # Render the right half of the window (Viewport 2)
            GL.glViewport(win_width - win_height // 2, 0, win_height // 2, win_height//2)
            self.ball_2d[i].draw(model)
            self.line_2d[i].draw(model)

            GL.glViewport(win_width - win_height // 2, win_height//2, win_height // 2, win_height//2)
            self.ball_2d[i].draw(model)
            self.line_2d[i].draw(model)
        
        # flush render commands, and swap draw buffers
        glfw.swap_buffers(self.win)

        # Poll for and process events
        glfw.poll_events()
        
        self.num_frame += self.speed
        end_time = time.time()
        period = start_time - end_time
        sleep(max(1/self.fps-period, 0))

    def add(self, *drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

            # for drawable in self.drawables:
            #     if hasattr(drawable, 'key_handler'):
            #         drawable.key_handler(key)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])