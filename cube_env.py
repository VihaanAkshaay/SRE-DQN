#Imports

import numpy as np
import plotly.graph_objects as go
import random

# GENERIC OPERATIONS

def face_clock(mat):
  '''
  Rotates the face in clockwise direction

  INPUT: Matrix (3x3)
  initial face = 1 2 3
                 4 5 6
                 7 8 9

  OUTPUT: Matrix (3x3)
  rotated_face = 7 4 1
                 8 5 2
                 9 6 3
  '''

  return np.array([[mat[2][0],mat[1][0],mat[0][0]],[mat[2][1],mat[1][1],mat[0][1]],[mat[2][2],mat[1][2],mat[0][2]]])

def face_anti(mat):
  '''
  initial face = 1 2 3
                 4 5 6
                 7 8 9

  rotated_face = 3 6 9
                 2 5 8
                 1 4 7
  '''

  return np.array([[mat[0][2],mat[1][2],mat[2][2]],[mat[0][1],mat[1][1],mat[2][1]],[mat[0][0],mat[1][0],mat[2][0]]])


def visualise(c):
    '''
    FUNCTION: Displays the cube 

    INPUTS: Cube object
    OUTPUT: None
    '''
    U1=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[0], alphahull=0,opacity=1)#top-mid
    U2=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[1,1,-1,-1,1,1,-1,-1], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[1], alphahull=0,opacity=1)#top-mid
    U3=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[3,3,1,1,3,3,1,1], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[2], alphahull=0,opacity=1)#top-mid
    U4=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[3], alphahull=0,opacity=1)#top-mid
    U5=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[1,1,-1,-1,1,1,-1,-1], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[4], alphahull=0,opacity=1)#top-mid
    U6=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[3,3,1,1,3,3,1,1], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[5], alphahull=0,opacity=1)#top-mid
    U7=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[6], alphahull=0,opacity=1)#top-mid
    U8=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[1,1,-1,-1,1,1,-1,-1], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[7], alphahull=0,opacity=1)#top-mid
    U9=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[3,3,1,1,3,3,1,1], z=[3,3,3,3,3.1,3.1,3.1,3.1], color=c[8], alphahull=0,opacity=1)#top-mid
    #####################################################################################################################################
    D1=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[9], alphahull=0,opacity=1)#botton-mid
    D2=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[1,1,-1,-1,1,1,-1,-1], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[10], alphahull=0,opacity=1)#botton-mid
    D3=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[3,3,1,1,3,3,1,1], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[11], alphahull=0,opacity=1)#botton-mid
    D4=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[12], alphahull=0,opacity=1)#botton-mid
    D5=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[1,1,-1,-1,1,1,-1,-1], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[13], alphahull=0,opacity=1)#botton-mid
    D6=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[3,3,1,1,3,3,1,1], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[14], alphahull=0,opacity=1)#botton-mid
    D7=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[15], alphahull=0,opacity=1)#botton-mid
    D8=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[1,1,-1,-1,1,1,-1,-1], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[16], alphahull=0,opacity=1)#botton-mid
    D9=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[3,3,1,1,3,3,1,1], z=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], color=c[17], alphahull=0,opacity=1)#botton-mid
    #####################################################################################################################################
    B1=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[3,3,1,1,3,3,1,1], z=[3,1,1,3,3,1,1,3], color=c[18], alphahull=0,opacity=1)#back-mid 
    B2=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[1,1,-1,-1,1,1,-1,-1], z=[-1,1,1,3,3,1,1,3], color=c[19], alphahull=0,opacity=1)#back-mid
    B3=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[3,1,1,3,3,1,1,3], color=c[20], alphahull=0,opacity=1)#back-mid
    B4=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[3,3,1,1,3,3,1,1], z=[1,-1,-1,1,1,-1,-1,1], color=c[21], alphahull=0,opacity=1)#back-mid
    B5=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[1,1,-1,-1,1,1,-1,-1], z=[1,-1,-1,1,1,-1,-1,1], color=c[22], alphahull=0,opacity=1)#back-mid
    B6=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[1,-1,-1,1,1,-1,-1,1], color=c[23], alphahull=0,opacity=1)#back-mid
    B7=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[3,3,1,1,3,3,1,1], z=[-1,-3,-3,-1,-1,-3,-3,-1], color=c[24], alphahull=0,opacity=1)#back-mid
    B8=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[1,1,-1,-1,1,1,-1,-1], z=[-1,-3,-3,-1,-1,-3,-3,-1], color=c[25], alphahull=0,opacity=1)#back-mid
    B9=go.Mesh3d( x=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z=[-1,-3,-3,-1,-1,-3,-3,-1], color=c[26], alphahull=0,opacity=1)#back-mid
    #####################################################################################################################################
    F1=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z= [3,1,1,3,3,1,1,3], color=c[27], alphahull=0,opacity=1)
    F2=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[1,1,-1,-1,1,1,-1,-1], z= [3,1,1,3,3,1,1,3], color=c[28], alphahull=0,opacity=1)
    F3=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[3,3,1,1,3,3,1,1], z= [3,1,1,3,3,1,1,3], color=c[29], alphahull=0,opacity=1)
    F4=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z= [1,-1,-1,1,1,-1,-1,1], color=c[30], alphahull=0,opacity=1)#front-mid
    F5=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[1,1,-1,-1,1,1,-1,-1], z= [1,-1,-1,1,1,-1,-1,1], color=c[31], alphahull=0,opacity=1)#front-mid
    F6=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[3,3,1,1,3,3,1,1], z= [1,-1,-1,1,1,-1,-1,1], color=c[32], alphahull=0,opacity=1)#front-mid
    F7=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[-1,-1,-3,-3,-1,-1,-3,-3], z= [-1,-3,-3,-1,-3,-1,-1,-1], color=c[33], alphahull=0,opacity=1)
    F8=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[1,1,-1,-1,1,1,-1,-1], z= [-1,-3,-3,-1,-3,-1,-1,-1], color=c[34], alphahull=0,opacity=1)
    F9=go.Mesh3d(x=[3,3,3,3,3.1,3.1,3.1,3.1], y=[3,3,1,1,3,3,1,1], z= [-1,-3,-3,-1,-3,-1,-1,-1], color=c[35], alphahull=0,opacity=1)
    ####################################################################################################################################
    L1=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[3,3,1,1,3,3,1,1], color=c[36], alphahull=0,opacity=1)#left-mid
    L2=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[3,3,1,1,3,3,1,1], color=c[37], alphahull=0,opacity=1)#left-mid
    L3=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[3,3,1,1,3,3,1,1], color=c[38], alphahull=0,opacity=1)#left-mid
    L4=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[1,1,-1,-1,1,1,-1,-1], color=c[39], alphahull=0,opacity=1)#left-mid
    L5=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[1,1,-1,-1,1,1,-1,-1], color=c[40], alphahull=0,opacity=1)#left-mid
    L6=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[1,1,-1,-1,1,1,-1,-1], color=c[41], alphahull=0,opacity=1)#left-mid
    L7=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[-1,-1,-3,-3,-1,-1,-3,-3], color=c[42], alphahull=0,opacity=1)#left-mid
    L8=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[-1,-1,-3,-3,-1,-1,-3,-3], color=c[43], alphahull=0,opacity=1)#left-mid
    L9=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[-3,-3,-3,-3,-3.1,-3.1,-3.1,-3.1], z=[-1,-1,-3,-3,-1,-1,-3,-3], color=c[44], alphahull=0,opacity=1)#left-mid
    ####################################################################################################################################
    R1=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[3,3,1,1,3,3,1,1], color=c[45], alphahull=0,opacity=1)#right-mid
    R2=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[3,3,1,1,3,3,1,1], color=c[46], alphahull=0,opacity=1)#right-mid
    R3=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[3,3,1,1,3,3,1,1], color=c[47], alphahull=0,opacity=1)#right-mid
    R4=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[1,1,-1,-1,1,1,-1,-1], color=c[48], alphahull=0,opacity=1)#right-mid
    R5=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[1,1,-1,-1,1,1,-1,-1], color=c[49], alphahull=0,opacity=1)#right-mid
    R6=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[1,1,-1,-1,1,1,-1,-1], color=c[50], alphahull=0,opacity=1)#right-mid
    R7=go.Mesh3d(x=[3,1,1,3,3,1,1,3], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[-1,-1,-3,-3,-1,-1,-3,-3], color=c[51], alphahull=0,opacity=1)#right-mid
    R8=go.Mesh3d(x=[1,-1,-1,1,1,-1,-1,1], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[-1,-1,-3,-3,-1,-1,-3,-3], color=c[52], alphahull=0,opacity=1)#right-mid
    R9=go.Mesh3d(x=[-1,-3,-3,-1,-1,-3,-3,-1], y=[3,3,3,3,3.1,3.1,3.1,3.1], z=[-1,-1,-3,-3,-1,-1,-3,-3], color=c[53], alphahull=0,opacity=1)#right-mid
    #####################################################################################################################################
    fig = go.Figure(data=[U1,U2,U3,U4,U5,U6,U7,U8,U9,D1,D2,D3,D4,D5,D6,D7,D8,D9,B1,B2,B3,B4,B5,B6,B7,B8,B9,L1,L2,L3,L4,L5,L6,L7,L8,L9,R1,R2,R3,R4,R5,R6,R7,R8,R9,F1,F2,F3,F4,F5,F6,F7,F8,F9])
    fig.show()

    return


# Cube Class

class Cube:
                  
    def __init__(self):
        self.front = np.array([[1,1,1],[1,1,1],[1,1,1]])
        self.right = np.array([[2,2,2],[2,2,2],[2,2,2]])
        self.left = np.array([[4,4,4],[4,4,4],[4,4,4]])
        self.back = np.array([[3,3,3],[3,3,3],[3,3,3]])
        self.up = np.array([[5,5,5],[5,5,5],[5,5,5]])
        self.down = np.array([[6,6,6],[6,6,6],[6,6,6]]) 
        
    def reset(self):
        self.front = np.array([[1,1,1],[1,1,1],[1,1,1]])
        self.right = np.array([[2,2,2],[2,2,2],[2,2,2]])
        self.left = np.array([[4,4,4],[4,4,4],[4,4,4]])
        self.back = np.array([[3,3,3],[3,3,3],[3,3,3]])
        self.up = np.array([[5,5,5],[5,5,5],[5,5,5]])
        self.down = np.array([[6,6,6],[6,6,6],[6,6,6]])

    # Returning Faces
    def showUp(self):
        print(self.front)

    def showDown(self):
        print(self.down)

    def showLeft(self):
        print(self.left)

    def showRight(self):
        print(self.right)

    def showFront(self):
        print(self.front)

    def showBack(self):
        print(self.back)
    

    # Moves
    def MoveUp(self):
        
        #rotate face tiles
        self.up = face_clock(self.up)
        
        #move adjacent layer tiles
        temp = np.array(self.front[0])
        self.front[0] = self.right[0]
        self.right[0] = self.back[0]
        self.back[0] = self.left[0]
        self.left[0] = temp
        
    def MoveUpI(self):
        
        #rotate face tiles
        self.up = face_anti(self.up)
        
        #move adjacent layer tiles
        temp = np.array(self.front[0])
        self.front[0] = self.left[0]
        self.left[0] = self.back[0]
        self.back[0] = self.right[0]
        self.right[0] = temp
    
    def MoveUp2(self):
        
        #rotate face tiles
        self.up = face_anti(face_anti(self.up))
        
        #move adjacent layers
        temp = np.array(self.front[0])
        self.front[0] = self.back[0]
        self.back[0] = temp
        
        temp = np.array(self.right[0])
        self.right[0] = self.left[0]
        self.left[0] = temp

    def MoveDown(self):
        
        #rotate face tiles
        self.down = face_clock(self.down)
        
        #move adjacent layer tiles
        temp = np.array(self.front[2])
        self.front[2] = self.left[2]
        self.left[2] = self.back[2]
        self.back[2] = self.right[2]
        self.right[2] = temp

    def MoveDownI(self):

        #rotate face tiles
        self.down = face_anti(self.down)

        #move adjacent layer tiles
        temp = np.array(self.front[2])
        self.front[2] = self.right[2]
        self.right[2] = self.back[2]
        self.back[2] = self.left[2]
        self.left[2] = temp

    def MoveDown2(self):

        #rotate face tiles
        self.down = face_clock(face_clock(self.down))

        #move adjacent layer tiles
        temp = np.array(self.front[2])
        self.front[2] = self.back[2]
        self.back[2] = temp
        temp = np.array(self.left[2])
        self.left[2] = self.right[2]
        self.right[2] = temp
       
    def MoveRight(self):
        
        #rotate face tiles
        self.right = face_clock(self.right)
        
        #move adjacent layer tiles
        temp = np.array(self.front[:,2])
        self.front[:,2] = self.down[:,2]
        self.down[:,2] = np.flip(self.back[:,0])
        self.back[:,0] = np.flip(self.up[:,2])
        self.up[:,2] = temp

    def MoveRightI(self):

        # rotate face tiles
        self.right = face_anti(self.right)

        #move adjacent layer tiles
        temp = np.array(self.front[:,2])
        self.front[:,2] = self.up[:,2]
        self.up[:,2] = np.flip(self.back[:,0])
        self.back[:,0] = np.flip(self.down[:,2])
        self.down[:,2] = temp
        
    def MoveRight2(self):
        
        #rotate face tiles
        self.right = face_clock(face_clock(self.right))

        # move adjacent layer tiles
        temp = np.array(self.front[:,2])
        self.front[:,2] = np.flip(self.back[:,0])
        self.back[:,0] = np.flip(temp)
        temp = np.array(self.up[:,2])
        self.up[:,2] = self.down[:,2]
        self.down[:,2] = temp
        
    def MoveLeft(self):
        
        #rotate face tiles
        self.left = face_clock(self.left)
        
        #move adjacent layer tiles
        temp = np.array(self.front[:,0])
        self.front[:,0] = self.up[:,0]
        self.up[:,0] = np.flip(self.back[:,2])
        self.back[:,2] = np.flip(self.down[:,0])
        self.down[:,0] = temp
        
    def MoveLeftI(self):
        
        #rotate face tiles
        self.left = face_anti(self.left)
        
        #move adjacent layer tiles
        temp = np.array(self.front[:,0])
        self.front[:,0] = self.down[:,0]
        self.down[:,0] = np.flip(self.back[:,2])
        self.back[:,2] = np.flip(self.up[:,0])
        self.up[:,0] = temp
        
    def MoveLeft2(self):
        
        #rotate face tiles
        self.left = face_clock(face_clock(self.left))
        
        #move adjacent layer tiles
        temp = np.array(self.front[:,0])
        self.front[:,0] = np.flip(self.back[:,2])
        self.back[:,2] = np.flip(temp)
        temp = np.array(self.down[:,0])
        self.down[:,0] = self.up[:,0]
        self.up[:,0] = temp
        
    def MoveFront(self):
        
        #rotate face tiles
        self.front = face_clock(self.front)
        
        #move adjacent layer tiles
        temp = np.array(self.left[:,2])
        self.left[:,2] = self.down[0]
        self.down[0] = np.flip(self.right[:,0])
        self.right[:,0] = self.up[2]
        self.up[2] = np.flip(temp)
        
    def MoveFrontI(self):
        
        #rotate face tiles
        self.front = face_anti(self.front)
        
        #move adjacent layer tiles
        temp = np.array(self.left[:,2])
        self.left[:,2] = np.flip(self.up[2])
        self.up[2] = self.right[:,0]
        self.right[:,0] = np.flip(self.down[0])
        self.down[0] = temp
        
    def MoveFront2(self):
        
        #rotate face tiles
        self.front = face_clock(face_clock(self.front))
        
        #move adjacent layer tiles
        temp = np.array(self.left[:,2])
        self.left[:,2] = np.flip(self.right[:,0])
        self.right[:,0] = np.flip(temp)
        temp = np.array(self.up[2])
        self.up[2] = np.flip(self.down[0])
        self.down[0] = np.flip(temp)
        
    def MoveBack(self):
        
        #rotate face tiles
        self.back = face_clock(self.back)
        
        #move adjacent layer tiles
        temp = np.array(self.up[0])
        self.up[0] = self.right[:,2]
        self.right[:,2] = np.flip(self.down[2])
        self.down[2] = self.left[:,0]
        self.left[:,0] = np.flip(temp)
        
    def MoveBackI(self):
        
        #rotate face tiles
        self.back = face_anti(self.back)
        
        #move adjacent layer tiles
        temp = np.array(self.up[0])
        self.up[0] = np.flip(self.left[:,0])
        self.left[:,0] = self.down[2]
        self.down[2] = np.flip(self.right[:,2])
        self.right[:,2] = temp
        
    def MoveBack2(self):
        
        #rotate face tiles
        self.back =face_clock(face_clock(self.back))
        
        #move adjacent layer tiles
        temp = np.array(self.up[0])
        self.up[0] = np.flip(self.down[2])
        self.down[2] = np.flip(temp)
        temp = np.array(self.left[:,0])
        self.left[:,0] = np.flip(self.right[:,2])
        self.right[:,2] = np.flip(temp)
        
        
    def GenerateColorList(self):
        '''
        Order:
        Up
        Down
        Back
        Front
        Left
        Right'''
        
        #Empty list to fill with strings of color name
        colors = []

        for i in np.matrix.flatten(self.up):
            if i == 1:
                colors.append('blue')
            if i == 2:
                colors.append('red')
            if i == 3:
                colors.append('green')
            if i == 4:
                colors.append('orange')
            if i == 5:
                colors.append('yellow')
            if i == 6:
                colors.append('white')
                
                        
        for i in np.matrix.flatten(self.down):
            if i == 1:
                colors.append('blue')
            if i == 2:
                colors.append('red')
            if i == 3:
                colors.append('green')
            if i == 4:
                colors.append('orange')
            if i == 5:
                colors.append('yellow')
            if i == 6:
                colors.append('white')
                
                        
        for i in np.matrix.flatten(self.back):
            if i == 1:
                colors.append('blue')
            if i == 2:
                colors.append('red')
            if i == 3:
                colors.append('green')
            if i == 4:
                colors.append('orange')
            if i == 5:
                colors.append('yellow')
            if i == 6:
                colors.append('white')
        
                
        for i in np.matrix.flatten(self.front):
            if i == 1:
                colors.append('blue')
            if i == 2:
                colors.append('red')
            if i == 3:
                colors.append('green')
            if i == 4:
                colors.append('orange')
            if i == 5:
                colors.append('yellow')
            if i == 6:
                colors.append('white')


                
        for i in np.matrix.flatten(self.left):
            if i == 1:
                colors.append('blue')
            if i == 2:
                colors.append('red')
            if i == 3:
                colors.append('green')
            if i == 4:
                colors.append('orange')
            if i == 5:
                colors.append('yellow')
            if i == 6:
                colors.append('white')
                              
       
        for i in np.matrix.flatten(self.right):
            if i == 1:
                colors.append('blue')
            if i == 2:
                colors.append('red')
            if i == 3:
                colors.append('green')
            if i == 4:
                colors.append('orange')
            if i == 5:
                colors.append('yellow')
            if i == 6:
                colors.append('white')
        return colors

    
    #### ALL REWARDS ARE ADDED HERE ####
    def checkReward(self):
        
        #Condition to check if cube is completely solved
        if self.checkSolved():
            return +100
        
        #CFOP - Reward
        #cum_reward = self.checkCFOP() 
        
        
        
        
        #return cum_reward
        return -1
            
        
        
        
    def checkSolved(self):
        
        if(not(self.front == np.array([[1,1,1],[1,1,1],[1,1,1]])).all()):
            return False
        if(not(self.right == np.array([[2,2,2],[2,2,2],[2,2,2]])).all()):
            return False
        if(not(self.left == np.array([[4,4,4],[4,4,4],[4,4,4]])).all()):
            return False
        if(not(self.back == np.array([[3,3,3],[3,3,3],[3,3,3]])).all()):
            return False
        if(not(self.up == np.array([[5,5,5],[5,5,5],[5,5,5]])).all()):
            return False
        if(not(self.down == np.array([[6,6,6],[6,6,6],[6,6,6]])).all()):
            return False
        return True
    

            
    def returnState(self):
        
        '''
        front
        back
        left
        right
        up
        down
        '''
    
          
        return np.hstack((self.front.flatten(),self.back.flatten(),self.left.flatten(),self.right.flatten(),self.up.flatten(),self.down.flatten()))

    
    def randomMove(self):
        
        choice = int(18*random.random())
        print(choice)
        self.step(choice)     

        return choice 
        
    def shuffleCube(self,n):
        for _ in range(n):
            self.randomMove()

    def step(self,choice):
                
        '''
        0,1,2: up
        3,4,5: down
        6,7,8: right
        9,10,11: left
        12,13,14: front
        15,16,17: back
        '''
        choice = int(choice)
        if choice == 0:
            self.MoveUp()
        if choice == 1:
            self.MoveUpI()
        if choice == 2:
            self.MoveUp2()
        
        if choice == 3:
            self.MoveDown() 
        if choice == 4:
            self.MoveDownI()
        if choice == 5:
            self.MoveDown2()
        
        if choice == 6:
            self.MoveRight()
        if choice == 7:
            self.MoveRightI()
        if choice == 8:
            self.MoveRight2()
            
        if choice == 9:
            self.MoveLeft()
        if choice == 10:
            self.MoveLeftI()
        if choice == 11:
            self.MoveLeft2()
        
        if choice == 12:
            self.MoveFront()
        if choice == 13:
            self.MoveFrontI()
        if choice == 14:
            self.MoveFront2()
        
        if choice == 15:
            self.MoveBack()
        if choice == 16:
            self.MoveBackI()
        if choice == 17:
            self.MoveBack2()

    def rev_action(self,choice):

        choice = int(choice)
        if choice == 0:
            return 1
        if choice == 1:
            return 0
        if choice == 2:
            return 2
        
        if choice == 3:
            return 4
        if choice == 4:
            return 3
        if choice == 5:
            return 5
        
        if choice == 6:
            return 7
        if choice == 7:
            return 6
        if choice == 8:
            return 8
            
        if choice == 9:
            return 10
        if choice == 10:
            return 9
        if choice == 11:
            return 11
        
        if choice == 12:
            return 13
        if choice == 13:
            return 12
        if choice == 14:
            return 14
        
        if choice == 15:
            return 16
        if choice == 16:
            return 15
        if choice == 17:
            return 17


