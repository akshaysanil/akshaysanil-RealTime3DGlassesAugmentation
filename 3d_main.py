import cv2
import trimesh
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glfw

# Load the 3D sunglasses model
glasses_mesh = trimesh.load('3dmodels/3dglassdata/oculos.obj')

cap = cv2.VideoCapture(0) #using my default camera

# loading face detection model-Haarcascade
face_cascade = cv2.CascadeClassifier('detection_model/haarcascade_frontalface_default.xml')

# it is the screen for rendering the obj, 
# glfw is to create and manage OpenGL windows.3d_render_env
if not glfw.init():
    raise Exception("GLFW initialization failed")

# here we are creating the window 

glfw.window_hint(glfw.VISIBLE, False)
window = glfw.create_window(1280, 960, "Invisible window", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

glfw.make_context_current(window)

def setup_lighting():
    glEnable(GL_LIGHTING)
    # Enable individual light sources
   
    glEnable(GL_LIGHT3)
   
    # Set the positions for the light sources
    glLightfv(GL_LIGHT3, GL_POSITION, (-1.0, -1.0, 1.0, 0.0))# Directional light
   

def draw_glasses(scale, x_offset, y_offset):
    glPushMatrix()
    print(x_offset,y_offset)

    glTranslatef(x_offset*2.5, y_offset*2,0) # movement of glass with face 
    glScalef(scale, scale, scale)
    
    # Set material properties for the glasses (black color)
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])  # Ambient color (black)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.0, 0.0, 0.0, 1.0])  # Diffuse color (black)
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])  # Specular color (black)
    glMaterialf(GL_FRONT, GL_SHININESS, 0.0)  # Shininess (0.0 for non-shiny)

    # Draw the glasses
    glBegin(GL_TRIANGLES)
    for face in glasses_mesh.faces:
        for vertex_index in face:
            vertex = glasses_mesh.vertices[vertex_index]
            glVertex3fv(vertex)
    glEnd()
    
    glPopMatrix()


def render_glasses(width, height, scale, x_offset, y_offset):
    # Set the viewport and projection matrix
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width/height, 0.1, 100.0)

    # setting the model view and camera
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 5, 0) # Set the camera position eyepoint,centerofview,updirection

    # Clear the color and depth buffers 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    setup_lighting()
    draw_glasses(scale, x_offset, y_offset)
    
    # Read the pixel data and flip the image vertically
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    return cv2.flip(image, 0)


# Parameters for face detection
min_neighbors = 5
min_size = (50, 50)

previous_x_offset = None
previous_y_offset = None

alpha = 1 # Smoothing factor

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=min_size)
    
    for (x, y, w, h) in faces:
        
        # draw bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        scale = w / 200  # Adjust scale based on face width
        x_center = x + w / 2
        y_center = y + h / 2

        x_offset = (x_center - frame.shape[1] / 2) / (frame.shape[1] / 2)
        y_offset = -(y_center - frame.shape[0] / 2) / (frame.shape[0] / 2)

        print(frame.shape)
        print(x_center)
        print(x_offset)
        print(y_offset)

        
        # Apply smoothing
        if previous_x_offset is not None and previous_y_offset is not None:
            x_offset = alpha * x_offset + (1 - alpha) * previous_x_offset
            y_offset = alpha * y_offset + (1 - alpha) * previous_y_offset

        previous_x_offset = x_offset
        previous_y_offset = y_offset

        glasses_overlay = render_glasses(frame.shape[1], frame.shape[0], scale, x_offset, y_offset)

        # Ensure glasses_overlay has the same dimensions as the frame
        if glasses_overlay.shape[:2] != frame.shape[:2]:
            glasses_overlay = cv2.resize(glasses_overlay, (frame.shape[1], frame.shape[0]))
        
        mask = glasses_overlay[:,:,3] > 0
        frame[mask] = cv2.addWeighted(frame, 0.5, glasses_overlay[:,:,:3], 0.5, 0)[mask]
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
glfw.terminate()
