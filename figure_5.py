import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from helperfunctions import *
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA


## load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = x_train.reshape(x_train.shape[0], -1) / 255.0
nrow = 28
ncol = 28
npix = nrow*ncol
ncomp = 100

pca = PCA(n_components=ncomp)
pca.fit(data)
principal_components = pca.components_
s = pca.singular_values_

img_graph = Image(nrow, ncol)
L, A, D = img_graph.get_laplacian_adjacency_degree_matrix()
eigval, eigvec = np.linalg.eigh(L)

## plot principal component smoothness and Laplacian eigenvector smoothness
plt.figure()
for i in range(100):
    if i == 0:
        plt.plot(i+1, np.dot(principal_components[i,:], np.dot(L, principal_components[i,:])), 'ro', label=f'Covariance eigenfunction smoothness')
        plt.plot(i+1, np.dot(eigvec[:,i], np.dot(L, eigvec[:,i])), 'bx', label='Laplacian eigenfunction smoothness')
    else:
        plt.plot(i+1, np.dot(principal_components[i,:], np.dot(L, principal_components[i,:])), 'ro')
        plt.plot(i+1, np.dot(eigvec[:,i], np.dot(L, eigvec[:,i])), 'bx')
plt.legend()
plt.tight_layout()
plt.ylabel("Dirichlet Energy of Principal Component or Laplacian Eigenvector")
plt.xlabel("Principal Component Number/Laplacian Eigenvector Number")
plt.title("Smoothness of MNIST")
plt.savefig(f"figure_5.png")





 





























'''
## train PCA (just regular PCA in this case)
means = np.mean(data, axis=0)
centered = data - means # the data matrix with the mean of each pixel subtracted; "broadcasting" is done automatically as it's a mean over columns not rows


N = len(data)
l = s**2 / N # eigenvalues of empirical covariance matrix

plt.figure()
plt.plot(1+np.arange(len(l)), 10*np.log10(l/sum(l)), 'o-')
plt.title("Explained variance ratio")
plt.xlabel("Principal component number")
plt.ylabel(r"$10\cdot \log_{10}$ explained variance ratio")
plt.savefig(r'figure_5.png')
plt.close()



## plot example marginal distribution, for 10 random pixels
for i in np.random.choice(data.shape[1], 10): # choose 10 random pixels
    sigma = np.sqrt(C[i,i]) # modeled marginal standard deviation
    m = means[i] # modeled marginal mean
    xs = np.linspace(m-4*sigma, m+4*sigma, 1000)

    plt.figure()
    plt.plot(xs, stats.norm.pdf(xs, m, sigma), color='k', label=f'marginal (from empirical-covariance-based model)')
    plt.hist(data[:,i], bins=20, alpha=0.5, color='g', label=f'empirical marginal', density=True, histtype='step')
    plt.legend()
    plt.title(f'Pixel {i}')
    plt.savefig(f'hist_marginal_pixel_{i}.png')
    plt.close()



## plot principal components and mean
plt.figure()
plt.imshow(means.reshape((nrow, ncol)), cmap='gray')
plt.title("Mean image")
plt.colorbar()
plt.axis('off')
plt.savefig('mean_image.png')

for c in range(ncomp):
    plt.figure()
    plt.imshow(principal_components[c,:].reshape((nrow, ncol)), cmap='gray')
    plt.title(f"principal component {c+1}")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'principal_component_{c+1}.png')
    plt.close()

## plot example reconstructed images, using PCA model, for 10 random images
for i in np.random.choice(n, 10):  # choose 10 random images
    img_i = data[i, :].reshape((1,npix))  # get the i-th image
    projected_i = (img_i - means) @ principal_components[:, :ncomp]  # project the centered image onto the first nc principal components
    reconstructed_i = V[:,:ncomp] @ projected_i.flatten() + means  # reconstruct the image from the projected data

    plt.figure()
    plt.imshow(reconstructed_i.reshape((nrow, ncol)), cmap='gray')
    plt.axis('off')
    plt.title(f"Reconstructed image {i}, using {ncomp} principal components")
    plt.savefig(f'reconstructed_image_{i}_using_{ncomp}_pcs.png')
    plt.close()

    plt.figure()
    plt.imshow(np.abs(img_i - reconstructed_i).reshape((nrow, ncol)), cmap='gray')
    plt.axis('off')
    plt.title(f"Reconstruction error, image {i}, using {ncomp} principal components")
    plt.colorbar()
    plt.savefig(f'reconstructed_error_image_{i}_using_{ncomp}_pcs.png')
    plt.close()

    ## IF WE HAVE A SMALL PATCH SIZE RELATIVE TO A DATA SET OF FAIRLY SMOOTH IMAGES, WE CAN SEE THAT THE PCA COMPONENTS ARE SIMILAR TO THE LAPLACE-BELTRAMI EIGENFUNCTIONS


## plot principal component smoothness and Laplacian eigenvector smoothness
plt.figure()
for i in range(100):
    if i == 0:
        plt.plot(i+1, np.dot(principal_components[i,:], np.dot(L, principal_components[i,:])), 'ro', label=f'Covariance eigenfunction smoothness')
        plt.plot(i+1, np.dot(eigvec[:,i], np.dot(L, eigvec[:,i])), 'bx', label='Laplacian eigenfunction smoothness')
    else:
        plt.plot(i+1, np.dot(principal_components[i,:], np.dot(L, principal_components[i,:])), 'ro')
        plt.plot(i+1, np.dot(eigvec[:,i], np.dot(L, eigvec[:,i])), 'bx')
plt.legend()
plt.tight_layout()
plt.ylabel("Dirichlet Energy of Principal Component or Laplacian Eigenvector")
plt.xlabel("Principal Component Number/Laplacian Eigenvector Number")
plt.title("Smoothness of MNIST")
plt.savefig(f"smoothness_comparison_{nrow}_{ncol}.png")


for i in range(ncomp):
    plt.figure()
    plt.imshow(eigvec[:, i].reshape((nrow, ncol)), cmap='gray')
    plt.title(f"Laplacian eigenfunction {i+1}")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'laplacian_eigenfunction_{i+1}.png')
    plt.close()
'''
