"""hw3_controller controller."""

# You may need to import some classes of the controller module. Ex:
# from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from controller import CameraRecognitionObject
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

x, y, theta = 0, 0.5, 0
v = 0.06
r = 0.5
omega = -v/r
wheelRadius = 0.0205
axleLength = 0.0568 # Data from Webots website seems wrong. The real axle length should be about 56-57mm
updateFreq = 200 # update every 200 timesteps
plotFreq = 50 # plot every 50 time steps
timeLimit = (2*np.pi)/(-omega) * 1

# helper functions
def omegaToWheelSpeeds(omega, v):
    wd = omega * axleLength * 0.5
    return v - wd, v + wd

def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def plot_cov(cov_mat, prob=0.95, num_pts=50):
    conf = chi2.ppf(0.95, df=2)
    L, V = np.linalg.eig(cov_mat)
    s1 = np.sqrt(conf*L[0])
    s2 = np.sqrt(conf*L[1])
    
    thetas = np.linspace(0, 2*np.pi, num_pts)
    xs = np.cos(thetas)
    ys = np.sin(thetas)

    standard_norm = np.vstack((xs, ys))
    S = np.array([[s1, 0],[0, s2]])
    scaled = np.matmul(S, standard_norm)
    R= V
    rotated = np.matmul(R, scaled)
    
    return(rotated)

# Task: Finish EKFPropagate and EKFRelPosUpdate
def EKFPropagate(x_hat_t, # robot position and orientation
                 Sigma_x_t, # estimation uncertainty
                 u, # control signals
                 Sigma_n, # uncertainty in control signals
                 dt # timestep
    ):
    # TODO: Calculate the robot state estimation and variance for the next timestep
    
    #Sigma Prediction
    phi_k = np.array([[1,0,(-dt*u[0]*np.sin(x_hat_t[2]))],[0,1,(dt*u[0]*np.cos(x_hat_t[2]))],[0,0,1]])
    g_k = np.array([[(-dt*np.cos(x_hat_t[2])),0],[(-dt*np.sin(x_hat_t[2])),0],[0,-dt]])
    Sigma_x_t = (phi_k@Sigma_x_t@phi_k.T)+(g_k@Sigma_n@g_k.T)
        
    # Pose prediction
    x_hat_t[0] = x_hat_t[0] + dt*u[0]*np.cos(x_hat_t[2])
    x_hat_t[1] = x_hat_t[1] + dt*u[0]*np.sin(x_hat_t[2])
    x_hat_t[2] = x_hat_t[2] + dt*u[1] 
    
    return x_hat_t, Sigma_x_t

def EKFRelPosUpdate(x_hat_t, # robot position and orientation
                    Sigma_x_t, # estimation uncertainty
                    z, # measurements
                    Sigma_m, # measurements' uncertainty
                    G_p_L, # landmarks' global positions
                    dt # timestep
                   ):
    # TODO: Update the robot state estimation and variance based on the received measurement
    C = np.array([[np.cos(x_hat_t[2]),-np.sin(x_hat_t[2])],[np.sin(x_hat_t[2]),np.cos(x_hat_t[2])]])
    G_p_L = np.array(G_p_L)
    Sigma_m = np.array(Sigma_m)
    z_k1 = C.T @ (G_p_L[:2]-x_hat_t[:2])
    error = z - z_k1
    J = np.array([[0,-1],[1,0]])
    H=np.zeros((2,3))
    H[:,:2]= -C.T
    H[:,2] = -C.T@J@(G_p_L[:2]-x_hat_t[:2])
    S = H@Sigma_x_t@H.T + Sigma_m
    K_gain = Sigma_x_t@H.T@np.linalg.inv(S)
    Sigma_x_t = Sigma_x_t - Sigma_x_t@H.T@np.linalg.inv(S)@H@Sigma_x_t
    x_hat_t = (x_hat_t + K_gain@error)
    return x_hat_t, Sigma_x_t


def EKFRelRangeUpdate(x_hat_t, # robot position and orientation
                    Sigma_x_t, # estimation uncertainty
                    z, # measurements
                    Sigma_m, # measurements' uncertainty
                    G_p_L, # landmarks' global positions
                    dt # timestep)        
        		):

    G_p_L = np.array(G_p_L)
    Sigma_m = np.array(Sigma_m)
    test = np.array([G_p_L[0]-x_hat_t[0],G_p_L[1]-x_hat_t[1]])
    z_k1 = np.linalg.norm(test)
    error = np.linalg.norm(z) - z_k1
    J = np.array([[0,-1],[1,0]])
    H=np.zeros((1,3))
    H[:,:2] = -(G_p_L[:2]-x_hat_t[:2]/np.linalg.norm(G_p_L[:2]-x_hat_t[:2]))
    H[:,2] = 0
    S = H@Sigma_x_t@H.T + Sigma_m[0,0]
    K_gain = Sigma_x_t@H.T*np.linalg.inv(S)
    Sigma_x_t = Sigma_x_t - Sigma_x_t@H.T@np.linalg.inv(S)@H@Sigma_x_t
    x_hat_t = (x_hat_t + np.squeeze(K_gain*error))
    return x_hat_t, Sigma_x_t


# create the Robot instance.
robot = Supervisor()
camera = robot.getDevice('camera')
camera.enable(1)

if camera.hasRecognition():
    camera.recognitionEnable(1)
    camera.enableRecognitionSegmentation()
else:
    print("Your camera does not have recognition")

timestep = int(robot.getBasicTimeStep())
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# init some variables
dt = timestep / 1000.0
x_hat_t = np.array([0, 0.5, 0])
Sigma_x_t = np.zeros((3,3))
Sigma_x_t[0,0], Sigma_x_t[1,1], Sigma_x_t[2,2] = 0.01, 0.01, np.pi/90
Sigma_n = np.zeros((2,2))
std_n_v = 0.01
std_n_omega = np.pi/60
Sigma_n[0,0] = std_n_v * std_n_v
Sigma_n[1,1] = std_n_omega * std_n_omega
u = np.array([v, omega])
counter = 0
timer = 0

# plot settings
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 10) 


while robot.step(timestep) != -1:
    # end the simulation when simulation reaches time limit
    timer += timestep / 1000
    if (timer > timeLimit):
        plt.show()
        break

    # Robot state
    robotNode = robot.getFromDef("e-puck")
    G_p_R = robotNode.getPosition()
    G_ori_R = robotNode.getOrientation()

    # Control signals
    left_v, right_v = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
    leftMotor.setVelocity(left_v/wheelRadius)
    rightMotor.setVelocity(right_v/wheelRadius)

    # EKF Propergate
    x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)

    # EKF Update
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    z_pos = np.zeros((recObjsNum, 2)) # relative position measurements   

    if counter % updateFreq == 0:
        for i in range(0, recObjsNum):
            landmark = robot.getFromId(recObjs[i].get_id())
            G_p_L = landmark.getPosition()
            rel_lm_trans = landmark.getPose(robotNode)

            std_m = 0.05
            Sigma_m = [[std_m*std_m, 0], [0,std_m*std_m]]
            z_pos[i] = [rel_lm_trans[3]+np.random.normal(0,std_m), rel_lm_trans[7]+np.random.normal(0,std_m)]                
            
            x_hat_t, Sigma_x_t = EKFRelPosUpdate(x_hat_t, Sigma_x_t, z_pos[i], Sigma_m, G_p_L, dt)
            
            # Uncomment below line for Bonus Task
            # x_hat_t, Sigma_x_t = EKFRelRangeUpdate(x_hat_t, Sigma_x_t, z_pos[i], Sigma_m, G_p_L, dt)
    counter = counter + 1

    if counter % plotFreq == 0:
        pts = plot_cov(Sigma_x_t[0:2,0:2])
        pts[0] += x_hat_t[0]
        pts[1] += x_hat_t[1]
        plt.scatter([pts[0,:]], [pts[1,:]])
        plt.scatter(x_hat_t[0],x_hat_t[1])
        plt.axis('equal')
    pass

# Enter here exit cleanup code.
