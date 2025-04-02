import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the matrix X (for example, a random 3x3 matrix)
X = np.eye(3)

# Function to compute the L1 norm of X(I - vv^T)
def compute_l1_norm(X, v):
    I = np.eye(3)  # Identity matrix of size 3
    vvT = np.outer(v, v)  # Outer product
    matrix = X @ (I - vvT)  # Matrix multiplication
    residuals = np.linalg.norm(matrix, axis=1, ord=1)
    return np.sum(residuals)

def compute_l2_norm(X, v):
    I = np.eye(3)  # Identity matrix of size 3
    vvT = np.outer(v, v)
    matrix = X @ (I - vvT) 
    residuals = np.linalg.norm(matrix, axis=1, ord=2)**2
    return np.sum(residuals)

def compute_l2_norm_robustish(X, v):
    I = np.eye(3)  # Identity matrix of size 3
    vvT = np.outer(v, v)
    matrix = X @ (I - vvT)
    residuals = np.linalg.norm(matrix, axis=1, ord=2)
    return np.sum(residuals)


# grid the unit sphere S^2
num_points = 1000
phi = np.linspace(0, 2*np.pi, num_points)  # colatitude
theta = np.linspace(0, np.pi, num_points)  # azimuthal angle
phi, theta = np.meshgrid(phi, theta)

# convert from spherical (physics convention) to cartesian
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# compute the L1 norm, and (robust PCA, i.e. not squared) L2 norm for each point on the sphere
l1_values = np.zeros_like(x)
l2r_values = np.zeros_like(x)
for i in range(num_points):
    for j in range(num_points):
        v = np.array([x[i, j], y[i, j], z[i, j]])
        l1_values[i, j] = compute_l1_norm(X, v)
        l2r_values[i, j] = compute_l2_norm_robustish(X, v)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Use a surface plot with shading based on l1_values
surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(l1_values/np.max(l1_values)), rstride=1, cstride=1, antialiased=True)

# Add color bar
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(l1_values)
plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, label=r'sum of $\ell^1$ residuals')

# Labels and title
ax.set_xlabel(r'$x=\sin(\theta)\cos(\phi)$')
ax.set_ylabel(r'$y=\sin(\theta)\sin(\phi)$')
ax.set_zlabel(r'$z=\cos(\theta)$')
ax.set_title(r'Values of $\|\mathbf{X}(\mathbf{I} - vv^T)\|^2_{1,1}$ for $v \in S^2$, with $\mathbf{X}=\mathbf{I}$') #\text{randn}(50,3)$')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Use a surface plot with shading based on l1_values
surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(l2r_values/np.max(l2r_values)), rstride=1, cstride=1, antialiased=True)

# Add color bar
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(l2r_values)
plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, label=r'sum of $\ell^2$ residuals')

# Labels and title
ax.set_xlabel(r'$x=\sin(\theta)\cos(\phi)$')
ax.set_ylabel(r'$y=\sin(\theta)\sin(\phi)$')
ax.set_zlabel(r'$z=\cos(\theta)$')
ax.set_title(r'Values of $\|\mathbf{X}(\mathbf{I} - vv^T)\|_{2,1}$ for $v \in S^2$, with $\mathbf{X}=\mathbf{I}$') #\text{random}(\{+1,0,-1\}, (50, 3))$') #\text{randn}(50,3)$')
plt.tight_layout()
plt.show()

