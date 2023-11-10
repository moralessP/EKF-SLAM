import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ExKalmanFilter:

  def __init__(self,initial_cov,process_cov,meas_cov,data,landmarks, animation = False):
    self.t, self.x, self.y, self.theta, self.v, self.omega, self.r, self.psi = data.T
    self.x_est = np.array([self.x[0], self.y[0], self.theta[0]])
    self.P_est = initial_cov
    self.Q = process_cov
    self.R = meas_cov
    self.landmarks = landmarks
    self.fig, self.ax = plt.subplots()
    self.animation = animation 

    ##### OUTPUT ######
    self.x_estimated = [self.x[0]]
    self.y_estimated = [self.y[0]]
    self.theta_estimated = [self.theta[0]]
    ###################

  def normalize(self,psi):
    if psi >= np.pi:
      psi = psi - 2*np.pi
    elif psi < -np.pi:
      psi = psi + 2*np.pi 
    return psi
  
  def animate(self,i): 
    self.ax.clear()
    self.ax.set_xlim(-20,20)
    self.ax.set_ylim(-10,25)
    self.ax.plot(self.x_estimated[:i],self.y_estimated[:i], label="Estimated position",color="red")
    self.ax.plot(self.x[:i],self.y[:i],'--', label="Real position", color="blue")
    self.ax.plot(self.landmarks[0,0], self.landmarks[0,1],"o", label="Landmarks", color="black")
    self.ax.plot(self.landmarks[1,0], self.landmarks[1,1],"o", color="black")
    #### MEAS FROM SENSOR
    if abs(self.psi[i]) <=np.pi/4 and self.r[i] != 0:
      self.ax.plot(self.x[i] + self.r[i] * np.cos(self.theta[i] + self.psi[i]), self.y[i] + self.r[i] * np.sin(self.theta[i] + self.psi[i]),"x", label="Meaurements", color="green")
    ####
    self.ax.set_xlabel("X(m)")
    self.ax.set_ylabel("Y(m)")
    self.ax.set_title("Evolution of the robot's position")
    self.ax.legend(loc="upper left")
    self.ax.grid(True)


  def jacobian_Fx(self,v,theta,dt):
    return np.array([[1,0,-v*np.sin(theta)*dt],
                     [0,1,v*np.cos(theta)*dt],
                     [0,0,1]])
  
  def jacobian_Fu(self,theta,dt):
    return np.array([[np.cos(theta)*dt, 0],
                     [np.sin(theta)*dt, 0],
                     [0, dt]])
  
  def jacobian_H(self,x,y,r):
    return np.array([[x/np.sqrt(r),y/np.sqrt(r), 0],
                     [-y/r,x/r, -1]])
  
  def predict(self,u,Fx,Fu):
    self.x_est = self.x_est + np.dot(Fu,u)
    self.P_est = np.dot(np.dot(Fx,self.P_est),Fx.T) + np.dot(np.dot(Fu,self.Q),Fu.T)

  def update(self,y,H):
    S = np.dot(np.dot(H,self.P_est),H.T) + self.R
    K = np.dot(np.dot(self.P_est,H.T), np.linalg.inv(S))
    self.x_est = self.x_est + np.dot(K,y)
    self.P_est = self.P_est - np.dot(np.dot(K,H),self.P_est)
    self.x_est[2] = self.normalize(self.x_est[2])

  def loop(self):
    dt = self.t[1] - self.t[0]
    for i in range(1,len(self.x)):

      # Prediction State
      u = np.array([self.v[i],self.omega[i]]) 
      Fx = self.jacobian_Fx(self.v[i],self.x_est[2],dt)
      Fu = self.jacobian_Fu(self.x_est[2],dt)
      self.predict(u,Fx,Fu)
      self.x_est[2] = self.normalize(self.x_est[2])

      # Update State
      z = np.array([self.r[i],self.psi[i]])
      if z[0] != 0:
        min_dist = float('inf')
        for lm in self.landmarks:
          r = (self.x_est[0] - lm[0])**2 + (self.x_est[1] - lm[1])**2
          alpha = np.arctan2(lm[1] - self.x_est[1], lm[0] - self.x_est[0]) - self.x_est[2]
          z_pred = np.array([np.sqrt(r), alpha])
          H = self.jacobian_H(self.x_est[0] - lm[0],self.x_est[1] - lm[0],r)
          y = z - z_pred
          S = np.dot(np.dot(H,self.P_est),H.T) + self.R
          dist = np.dot(np.dot(y.T,np.linalg.inv(S)),y) # Mahalanobis distance
          if min_dist > dist:
            min_dist = dist
            best_y = y
            best_H = H
        best_y[1] = self.normalize(best_y[1])
        self.update(best_y,best_H)

      # Append 
      self.x_estimated.append(self.x_est[0])
      self.y_estimated.append(self.x_est[1])
      self.theta_estimated.append(self.x_est[2])

    # Show
    if self.animation == True:
        ani = FuncAnimation(self.fig, self.animate, frames=2000, interval=100, repeat=False)
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        plt.show()




def main():
  initial_cov = np.diag([0.0,0.0,0.0])
  process_cov = np.diag([0.5, 0.05])**2 
  meas_cov = np.diag ([0.5, 0.1])**2 
  data = np.loadtxt("Data/data3.txt", delimiter=None, dtype=float)
  landmarks = np.array([[0,0],[10,0]])
  ekf = ExKalmanFilter(initial_cov,process_cov,meas_cov,data,landmarks, True)
  ekf.loop()
  
if __name__ == "__main__":
  main()
 
