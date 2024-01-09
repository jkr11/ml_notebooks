import numpy as np
import matplotlib.pyplot as plt
from   types import SimpleNamespace

def pca(X, n_components=None):
    """Perform principal component analysis on data X with [samples x features].
    """    
    n, p  = X.shape
    dim   = n_components if n_components else p
    xmean = X.mean(axis=0)

    # Mean center our data.
    X = X - xmean

    # Do eigendecomposition via the SVD.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute eigenvalues from singular values.
    L = S**2 / (n-1)

    # Sort output by eigenvalues in descending order.
    inds = S.argsort()[::-1]
    eigvals = L[inds][:dim]
    pcs = Vt[inds][:dim]

    # For consistency with the blog post text, transpose to make the columns of
    # `pcs` the eigenvectors,
    pcs = pcs.T
    
    # Transform X into latent variables Z.
    Z = (U*S)[:, :dim]

    out = {}
    out['pcs'] = pcs
    out['explained_variance'] = eigvals
    out['total_variance'] = eigvals.sum()
    out['Z'] = Z
    out['n_components'] = dim
    out['mean'] = xmean
    
    return SimpleNamespace(**out)



from sklearn import datasets

def visualize_pca_results(X, n_components=None):
    pca_results = pca(X, n_components)

    # Plotting the PCA results
    plt.scatter(pca_results.Z[:, 0], pca_results.Z[:, 1], c=iris.target, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Results - Iris Dataset')
    plt.show()
    

from   sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces(return_X_y=False)
result = pca(X=faces.data)
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(result.pcs[:, i].reshape(faces.images[0].shape), cmap='gray')
    ax.axis('off')
plt.show()
