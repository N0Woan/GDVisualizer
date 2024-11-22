import dearpygui.dearpygui as dpg
import numpy as np
import OpenGL.GL as GL
import glfw
import time

# A placeholder for the radius variable, assuming this is a global or part of a class
radius = 0.2

# Initialize DearPyGui context
dpg.create_context()

# Define a callback function for the slider
def update_radius(sender, app_data):
    global radius
    radius = app_data  # Update the radius with the slider value

# Set up the UI in DearPyGui
def setup_ui():
    with dpg.window(label="Settings", width=300, height=200, pos=(10, 10)):
        dpg.add_slider_float(label="Radius", default_value=0.2, min_value=0.05, max_value=1.0, callback=update_radius)
        dpg.add_text("Other UI elements can go here.")

setup_ui()

# Set up the viewport
dpg.create_viewport(title='DearPyGui Controller', width=400, height=300)
dpg.setup_dearpygui()
dpg.show_viewport()

# Initialize GLFW and OpenGL if needed for your rendering application
if not glfw.init():
    raise Exception("GLFW could not be initialized!")

# Set up OpenGL window
width, height = 1200, 800
win = glfw.create_window(width, height, 'OpenGL Window', None, None)
glfw.make_context_current(win)
GL.glClearColor(0.1, 0.1, 0.1, 1)

# Main render loop
while not glfw.window_should_close(win):
    # OpenGL Rendering
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    # Insert your OpenGL rendering code here that uses `radius` or other variables

    # Draw the DearPyGui UI
    dpg.render_dearpygui_frame()

    # GLFW Buffer swap and poll events
    glfw.swap_buffers(win)
    glfw.poll_events()

    # Sleep to limit frame rate
    time.sleep(1 / 60)

# Clean up DearPyGui and GLFW
dpg.destroy_context()
glfw.terminate()
