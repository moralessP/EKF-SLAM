import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SLAM:
    STATE_SIZE = 3
    LM_SIZE = 2

    def __init__(self, initial_pose, initial_cov, process_cov, meas_cov, data, animation = False):
        self.t, self.x, self.y, self.theta, self.v, self.omega, self.r, self.psi = data.T
       
        self.x_est = initial_pose
        self.P_est = initial_cov
        
        self.Q = process_cov
        self.R = meas_cov

        self.fig, self.ax = plt.subplots()
        ######## OUTPUTS ########## 
        self.x_estimated = [self.x_est[0]]
        self.y_estimated = [self.x_est[1]]
        self.theta_estimated = [self.x_est[2]]
        self.landmark = 0
        ######## OUTPUTS ##########

        self.animation = animation 

    def normalize(self, psi):
        if psi >= np.pi:
            psi = psi - 2 * np.pi
        elif psi < -np.pi:
            psi = psi + 2 * np.pi 
        return psi
    
    def animate(self,i): 
        self.ax.clear()
        self.ax.plot(self.x_estimated[:i],self.y_estimated[:i], label="Estimated position",color="red")
        self.ax.plot(self.x[:i],self.y[:i],'--', label="Real position", color="blue")
        n = i if i < self.nb_lm()-1 else self.nb_lm()-1
        for j in range(n):
            plt.plot(self.landmark[2*j], self.landmark[2*j+1], "xg")
        self.ax.set_xlabel("X(m)")
        self.ax.set_ylabel("Y(m)")
        self.ax.set_title("Evolution of the robot's position")
        self.ax.legend(loc="upper left")
        self.ax.grid(True)

    
    def jacobian_Fx(self, v, theta, dt):
        return np.array([[1, 0, -v * np.sin(theta) * dt],
                         [0, 1, v * np.cos(theta) * dt],
                         [0, 0, 1]])
  
    def jacobian_Fu(self, theta, dt):
        return np.array([[np.cos(theta) * dt, 0],
                         [np.sin(theta) * dt, 0],
                         [0, dt]])

    
    def jacobian_H(self, r, x, y, lm_ID):
        sr = np.sqrt(r)
        G = (1 / r) * np.array([[-sr * x, -sr * y, 0, sr * x, sr * y],
                            [y, -x, -r, -y, x]])
        nb_lm = self.nb_lm()
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nb_lm))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (lm_ID - 1))),
                    np.eye(2), np.zeros((2, 2 * nb_lm - 2 * lm_ID))))

        F = np.vstack((F1, F2))
        return np.dot(G, F)

    def nb_lm(self):
        return int((len(self.x_est) - self.STATE_SIZE) / self.LM_SIZE)
    
    def search_lm_ID(self, z):
        nb_lm = self.nb_lm()
        min_dist = []
        for i in range(nb_lm):
            lm = self.lm_position_from_state(i)
            y, S, H = self.update(lm, i, z)
            min_dist.append(np.dot(np.dot(y.T,np.linalg.inv(S)),y))
        min_dist.append(4) 
        min_ID = min_dist.index(min(min_dist)) 
        return min_ID
    
    def lm_position_from_state(self, i):
        return self.x_est[self.STATE_SIZE + self.LM_SIZE * i: self.STATE_SIZE + self.LM_SIZE * (i + 1)]
    
    def lm_position(self, z):
        return np.array([self.x_est[0] + z[0] * np.cos(self.x_est[2] + z[1]), self.x_est[1] + z[0] * np.sin(self.x_est[2] + z[1])])
    
    def predict(self, u, Fx, Fu):
        S = self.STATE_SIZE
        self.x_est[0:S] = np.dot(np.eye(3), self.x_est[0:S]) + np.dot(Fu, u)
        self.P_est[0:S, 0:S] = np.dot(np.dot(Fx, self.P_est[0:S, 0:S]), Fx.T) + np.dot(np.dot(Fu, self.Q), Fu.T)
    
    def update(self, lm, lm_ID, z):
        r = (self.x_est[0] - lm[0])**2 + (self.x_est[1] - lm[1])**2
        alpha = np.arctan2(lm[1] - self.x_est[1], lm[0] - self.x_est[0]) - self.x_est[2]
        z_pred = np.array([np.sqrt(r), self.normalize(alpha)])
        H = self.jacobian_H(r, self.x_est[0] - lm[0], self.x_est[1] - lm[1], lm_ID + 1)
        y = (z - z_pred).T
        y[1] = self.normalize(y[1])
        S = np.dot(np.dot(H, self.P_est), H.T) + self.R
        return y, S, H


    def loop(self):
        dt = self.t[1] - self.t[0]
        for i in range(1,len(self.x)):
            
            # Prediction State 
            u = np.array([self.v[i], self.omega[i]]).T 
            Fx = self.jacobian_Fx(self.v[i], self.x_est[2], dt)
            Fu = self.jacobian_Fu(self.x_est[2], dt)
            self.predict(u, Fx, Fu)
            self.x_est[2] = self.normalize(self.x_est[2])

            # Update State
            z = np.array([self.r[i], self.psi[i]])
            if abs((z[1])) <= np.pi/4 and z[0] != 0:
                min_ID = self.search_lm_ID(z)
                if min_ID == self.nb_lm():
                    self.x_est = np.hstack((self.x_est, self.lm_position(z)))
                    self.P_est = np.block([[self.P_est, np.zeros((self.P_est.shape[0], self.LM_SIZE))], 
                                           [np.zeros((self.LM_SIZE, self.P_est.shape[1])), np.eye(self.LM_SIZE)]])

                lm = self.lm_position_from_state(min_ID)
                y, S, H = self.update(lm, min_ID, z)
                    
                K = np.dot(np.dot(self.P_est, H.T), np.linalg.inv(S))
                self.x_est = self.x_est + np.dot(K, y)
                self.P_est = np.dot((np.eye(len(self.x_est)) - np.dot(K, H)), self.P_est)

            # Append
            self.x_estimated.append(self.x_est[0])
            self.y_estimated.append(self.x_est[1])
            self.theta_estimated.append(self.x_est[2])
            self.landmark = self.x_est[self.STATE_SIZE:]

            # Show
        if self.animation == True:
            ani = FuncAnimation(self.fig, self.animate, frames=2000, interval=100, repeat=False)
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.show()

def main():
    initial_pose = np.array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
    initial_cov = np.diag([5, 5, 0])
    process_cov = np.diag([0.5**2, 0.05**2])
    meas_cov = np.diag([0.5**2, 0.1**2])
    data = np.loadtxt("Data/data4.txt", delimiter=None, dtype=float)
    slam = SLAM(initial_pose, initial_cov, process_cov, meas_cov, data, True)
    slam.loop()
    
if __name__ == "__main__":
    main()
    