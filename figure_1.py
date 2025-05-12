import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the matrix X (for example, a random 3x3 matrix)
np.random.seed(72)
X = np.random.uniform(0,1,size=(50,3)) #np.random.randn(50,3) #np.array([[0,1,-1],[-1,0,-1],[1,1,0]])
X[X > 2/3] = 1
X[X < 1/3] = 0
X[np.logical_and(X>.3, X<.7)] = -1

# Function to compute the L1 norm of X(I - vv^T)
def compute_l1_norm(X, v): # L1 robust PCA
    I = np.eye(3)  # Identity matrix of size 3
    vvT = np.outer(v, v)  # Outer product
    matrix = X @ (I - vvT)  # Matrix multiplication
    residuals = np.linalg.norm(matrix, axis=1, ord=1)
    return np.sum(residuals)

def compute_l2_norm(X, v): # PCA norm
    I = np.eye(3)  # Identity matrix of size 3
    vvT = np.outer(v, v)
    matrix = X @ (I - vvT) 
    residuals = np.linalg.norm(matrix, axis=1, ord=2)**2
    return np.sum(residuals)

def compute_l2r_norm(X, v): # L2 robust PCA (aka 
    I = np.eye(3)  # Identity matrix of size 3
    vvT = np.outer(v, v)
    matrix = X @ (I - vvT)
    residuals = np.linalg.norm(matrix, axis=1, ord=2)
    return np.sum(residuals)


# Generate points on the unit sphere S^2
num_points = 1000
phi = np.linspace(0, 2*np.pi, num_points)  # Polar angle
theta = np.linspace(0, np.pi, num_points)  # Azimuthal angle
phi, theta = np.meshgrid(phi, theta)

# Convert spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Compute the L1 norm for each point on the sphere
l1_values = np.zeros_like(x)
l2_values = np.zeros_like(x)
l2r_values = np.zeros_like(x)
for i in range(num_points):
    for j in range(num_points):
        v = np.array([x[i, j], y[i, j], z[i, j]])
        l1_values[i, j] = compute_l1_norm(X, v)
        l2_values[i, j] = compute_l2_norm(X, v)
        l2r_values[i, j] = compute_l2r_norm(X, v)

def get_label(x,y):
    text = ""
    if x > 90:
        text += f"{x-90:.2f}"+r"$^\circ$ N, split"
    else:
        text += f"{90-x:.2f}"+r"$^\circ$ S, split"
    if y > 180:
        text += f"{y-180:.2f}"+r"$^\circ$ E"
    else:
        text += f"{180-y:.2f}"+r"$^\circ$ W"
    return text

def antipode(x, y):
    res = list([x,y])
    res[0] = 180 - res[0]
    res[1] = (180 + res[1]) if res[1] < 180 else (res[1] - 180) 
    return tuple(res)

min_index = np.argmin(l2_values.flatten())
second_min_index = np.argsort(l2_values.flatten())[1] # got rid of this to avoid minor numerical issues
label_min = get_label(theta.flatten()[min_index]*180/np.pi, phi.flatten()[min_index]*180/np.pi)
label_min_lat, label_min_lon = label_min.split("split")
min_index_xy = (theta.flatten()[min_index]*180/np.pi, phi.flatten()[min_index]*180/np.pi)
second_min_index_xy = antipode(*min_index_xy) # (theta.flatten()[second_min_index]*180/np.pi, phi.flatten()[second_min_index]*180/np.pi) # because of numerical issues
label_second_min = get_label(*second_min_index_xy)
label_second_min_lat, label_second_min_lon = label_second_min.split("split")


min_index_l1 = np.argmin(l1_values.flatten())
min_index_l1_xy = (theta.flatten()[min_index_l1]*180/np.pi, phi.flatten()[min_index_l1]*180/np.pi)
second_min_index_l1_xy = antipode(*min_index_l1_xy)

min_index_l2r = np.argmin(l2r_values.flatten())
min_index_l2r_xy = (theta.flatten()[min_index_l2r]*180/np.pi, phi.flatten()[min_index_l2r]*180/np.pi)
second_min_index_l2r_xy = antipode(*min_index_l2r_xy)



fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.scatter(theta*180/np.pi, phi*180/np.pi, c=l1_values, s=2, alpha=0.5)
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(l1_values)
plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, label=r'sum of $\ell^1$ residuals')
plt.xlabel(r"colatitude/polar angle $\theta$ (degrees) of $v=(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$")
plt.ylabel(r"azimuthal angle $\phi$ (degrees) of $v=(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$")
ax.set_xlim(0, 180)
ax.set_ylim(0,360)
ax.set_title(r'Values of $\|\mathbf{X}(\mathbf{I} - vv^T)\|^2_{1,1}$ for $v \in S^2$, with $\mathbf{X}=\text{random}(\{+1,0,-1\})^{50\times 3}$') # \text{randn}(50,3)$')
plt.annotate('init point (PCA)', xy=tuple(np.array(min_index_xy) + np.array([-25,-14])), color='r', fontsize=16)
plt.annotate('init point (PCA)', xy=tuple(np.array(second_min_index_xy) + np.array([-25,+5])), color='r', fontsize=16)
plt.scatter(min_index_xy[0], min_index_xy[1], marker='x', color='r', s=60)
plt.scatter(second_min_index_xy[0], second_min_index_xy[1], marker='x', color='r', s=40)
plt.annotate('true minimum', xy=tuple(np.array(min_index_l1_xy) + np.array([2,2])), color='g', fontsize=16)
plt.annotate('true minimum', xy=tuple(np.array(second_min_index_l1_xy) + np.array([-54,3])), color='g', fontsize=16)
plt.scatter(min_index_l1_xy[0], min_index_l1_xy[1], marker='o', color='g', s=60)
plt.scatter(second_min_index_l1_xy[0], second_min_index_l1_xy[1], marker='o', color='g', s=60)
plt.axvline(x=90, color='g', linestyle='--', linewidth=2)
plt.axhline(y=180, color='g', linestyle='--', linewidth=2)
plt.tight_layout()
plt.show()



fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.scatter(theta*180/np.pi, phi*180/np.pi, c=np.around(l2_values, 10), s=2, alpha=0.5)
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(l2_values)
plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, label=r'sum of squared $\ell^2$ residuals')
plt.xlabel(r"colatitude/polar angle $\theta$ (degrees) of $v=(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$")
plt.ylabel(r"azimuthal angle $\phi$ (degrees) of $v=(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$")
ax.set_title(r'Values of $\|\mathbf{X}(\mathbf{I} - vv^T)\|^F_2$ for $v \in S^2$, with $\mathbf{X}=\text{random}(\{+1,0,-1\})^{50\times 3}$') #\text{randn}(50,3)$')
ax.set_xlim(0, 180)
ax.set_ylim(0,360)
plt.axvline(x=90, color='r', linestyle='--', linewidth=2)
plt.axhline(y=180, color='r', linestyle='--', linewidth=2)
plt.annotate(label_min_lat, xy=tuple(np.array(min_index_xy) + np.array([-16,-11])), color='r', fontsize=14)
plt.annotate(label_second_min_lat, xy=tuple(np.array(second_min_index_xy) + np.array([2,-2])), color='r', fontsize=14) 
plt.annotate(label_min_lon, xy=tuple(np.array(min_index_xy) + np.array([-16,-20])), color='r', fontsize=14) 
plt.annotate(label_second_min_lon, xy=tuple(np.array(second_min_index_xy) + np.array([2,-10])), color='r', fontsize=14) 
plt.scatter(min_index_xy[0], min_index_xy[1], marker='x', color='r', s=40)
plt.scatter(second_min_index_xy[0], second_min_index_xy[1], marker='x', color='r', s=40)
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.scatter(theta*180/np.pi, phi*180/np.pi, c=l1_values, s=2, alpha=0.5)
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(l1_values)
plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, label=r'sum of $\ell^2$ residuals')
plt.xlabel(r"colatitude/polar angle $\theta$ (degrees) of $v=(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$")
plt.ylabel(r"azimuthal angle $\phi$ (degrees) of $v=(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$")
ax.set_xlim(0, 180)
ax.set_ylim(0,360)
ax.set_title(r'Values of $\|\mathbf{X}(\mathbf{I} - vv^T)\|^2_{1,1}$ for $v \in S^2$, with $\mathbf{X}=\text{random}(\{+1,0,-1\})^{50\times 3}$') # \text{randn}(50,3)$')
plt.annotate('init point (PCA)', xy=tuple(np.array(min_index_xy) + np.array([-25,-14])), color='r', fontsize=16)
plt.annotate('init point (PCA)', xy=tuple(np.array(second_min_index_xy) + np.array([-25,+5])), color='r', fontsize=16)
plt.scatter(min_index_xy[0], min_index_xy[1], marker='x', color='r', s=60)
plt.scatter(second_min_index_xy[0], second_min_index_xy[1], marker='x', color='r', s=40)
plt.annotate('true minimum', xy=tuple(np.array(min_index_l2r_xy) + np.array([2,2])), color='w', fontsize=16)
plt.annotate('true minimum', xy=tuple(np.array(second_min_index_l2r_xy) + np.array([-54,3])), color='w', fontsize=16)
plt.scatter(min_index_l2r_xy[0], min_index_l2r_xy[1], marker='o', color='w', s=60)
plt.scatter(second_min_index_l2r_xy[0], second_min_index_l2r_xy[1], marker='o', color='w', s=60)
plt.axvline(x=90, color='g', linestyle='--', linewidth=2)
plt.axhline(y=180, color='g', linestyle='--', linewidth=2)
plt.tight_layout()
plt.show()

