#!/usr/bin/env python3

import numpy as np
import scipy.linalg as scp
from numpy.random import randn
import matplotlib.pyplot as plt

x = np.array([[1],
              [1]])
A = np.array([[1,1],
              [0,1]])

B = 1
K = np.array([[0,0],
             [-0.35,-0.55]])
print("The eigenvalues of A+BK are of magnitude",
      np.abs(scp.eig(A+B*K)[0][0]),"and",
      np.abs(scp.eig(A+B*K)[0][1]),
      "and so lie inside the unit circle, so K is a stabilising gain for (A,B).")

Q=np.array([[0.00001, 0],
            [0, 0.00004]])
P=np.array([[1,0],
            [0,0.5]])
R=0.125#std=0.25
C=np.array([[1,0]])

def move(x,u,A=A,B=B,Q=Q):
    #moves the simulation forward 1 step
    w = Q @ np.array([[randn()],
                      [randn()]])
    return A@x+B*u+w

def predict(x,u,P,A=A,B=B,Q=Q):
    #returns prediction of x and its covariance matrix P
    return A@x+B*u, A@P@(A.T)+Q

def measure(x,R=R):
    return C@x + randn()*np.sqrt(R)

def update(x,P,y,R=R,C=C):
    diff = y - C@x
    kalman_gain = P@C.T/(P[0,0]+R)
    estimate = x + kalman_gain*diff
    P = (np.identity(2)-kalman_gain@C)@P
    return estimate, P

xs=[]
ys=[]
ests=[]
us=[]

#initialise filter
xs.append(x)
ys.append(measure(x))
ests.append(np.array([ys[0][0],[0]]))

for i in range(1000):
    us.append(K@ests[i])
    prediction,P = predict(ests[i],us[i],P)
    estimate,P = update(prediction,P,ys[i])
    ests.append(estimate)
    xs.append(move(xs[i],us[i]))
    ys.append(measure(xs[i+1]))

plt.clf()
plt.plot(np.asarray(xs)[:,0])
plt.xlabel("Steps")
plt.ylabel("Position (units)")
plt.savefig("positions.png")
plt.clf()
plt.plot(np.asarray(xs)[:,1])
plt.xlabel("Steps")
plt.ylabel("Velocity (units/step)")
plt.savefig("velocities.png")
plt.clf()
plt.plot(np.asarray(us)[:,1])
plt.xlabel("Steps")
plt.ylabel("Inputs (units/step^2)")
plt.savefig("inputs.png")
plt.clf()
plt.plot(np.asarray(ys)[:,0,0])
plt.xlabel("Steps")
plt.ylabel("Measurements (units)")
plt.savefig("measurements.png")
plt.clf()
plt.plot(np.asarray(ests)[:,0])
plt.xlabel("Steps")
plt.ylabel("Position Estimates (units)")
plt.savefig("estimates.png")
