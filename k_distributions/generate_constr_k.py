import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

def exponential_covariance(distance, length, sigma):
#     return np.exp(- distance[0]**2 / np.float(length[0]**2) - distance[1]**2 / np.float(length[1]**2)  ) * sigma**2
    return np.exp(- distance / np.float(length) ) * sigma**2

def build_covariance_matrix(nx, ny,  lx, ly, length, sigma):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xv, yv = np.meshgrid(x, y)

    coords = list(zip(xv.ravel(), yv.ravel()))
#     distance_x = distance_matrix(xv.ravel().reshape(-1,1), xv.ravel().reshape(-1,1), p=1)
#     distance_y = distance_matrix(yv.ravel().reshape(-1,1), yv.ravel().reshape(-1,1), p=1)
#     distance_mat = [distance_x, distance_y]
    distance_mat = cdist(coords, coords)
    covariance_matrix = exponential_covariance(distance_mat, length, sigma)
    return covariance_matrix

def generate(nx, ny, lx, ly, length, sigma, sample_size=1, seed=1):
    """ generate multiple realizations """
    np.random.seed(seed)
    covariance_matrix = build_covariance_matrix(nx, ny, lx, ly, length, sigma)
    chol = cholesky(covariance_matrix, lower=True)
    ubatch = np.random.randn(ny, nx, sample_size).reshape(ny * nx, sample_size)
    vbatch = chol.dot(ubatch)
    vbatch = vbatch.T.reshape(sample_size, ny, nx).squeeze()
    return vbatch

def generate_cond(nx, ny, lx, ly, length, sigma, cond_idx, sample_size=1, seed=1):
    """ generate multiple realizations zero condition at specific grid indices 1D format """
    np.random.seed(seed)
    cond_idx  = np.array(cond_idx).flatten()
    assert cond_idx.max() < (nx*ny) and cond_idx.min() >= 0 and np.unique(cond_idx).shape == cond_idx.shape

    K_mat = build_covariance_matrix(nx, ny, lx, ly, length, sigma)
    
    x_star = np.ones(K_mat.shape[0], np.bool)
    x_star[cond_idx] = False
    x = ~x_star
    # equation 2.19 in the book http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
    K_new = K_mat[x_star, :][:, x_star] - \
        np.dot(K_mat[x_star,:][:, x], np.dot(np.linalg.inv(K_mat[x, :][:, x]), K_mat[x, :][:, x_star]))

    # chol = cholesky(K_new, lower=True) # wouldn't work now -- K_new is not symmetric anymore
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = np.linalg.eigh(K_new)

    # Construct c, so c*c^T = r.
    c = np.dot(evecs, np.diag(np.sqrt(evals)))
    # Convert the data to correlated random variables. 
    vbatch = np.dot(c, np.random.randn(c.shape[0], sample_size))
    
    vbatch_full = np.zeros((K_mat.shape[0], sample_size))
    vbatch_full[x_star, :] = vbatch

    vbatch_full = vbatch_full.T.reshape(sample_size, ny, nx).squeeze()

    return vbatch_full


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def generate_cond_corrected(nx, ny, lx, ly, length, sigma, log_mean, log_values_cond, sigma_error, cond_idx, sample_size=1, seed=1):
    """ generate multiple realizations zero condition at specific grid indices 1D format """
    np.random.seed(seed)
    cond_idx  = np.array(cond_idx).flatten()
    f = np.array(log_values_cond).flatten() # log-k values from observation, notations from http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
    assert cond_idx.max() < (nx*ny) and cond_idx.min() >= 0 and np.unique(cond_idx).shape == cond_idx.shape
    assert cond_idx.shape[0] == f.shape[0], 'cond_idx and log values should have same length'
    n_data = cond_idx.shape[0]

    K_mat = build_covariance_matrix(nx, ny, lx, ly, length, sigma)
    
    x_star = np.ones(K_mat.shape[0], np.bool)
    x_star[cond_idx] = False
    x = ~x_star
    # equation 2.19 in the book http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
    K_new = K_mat[x_star, :][:, x_star] - \
        np.dot(K_mat[x_star,:][:, x], np.dot(np.linalg.inv(K_mat[x, :][:, x]+sigma_error*np.diag(np.ones(n_data))), K_mat[x, :][:, x_star]))
    
    
    y = f.reshape(-1,1) + sigma_error*np.random.randn(n_data,sample_size)
    K_new_mean = np.dot(K_mat[x_star,:][:, x], np.dot(np.linalg.inv(K_mat[x, :][:, x]+sigma_error*np.diag(np.ones(n_data))), (log_mean-y)))

    # chol = cholesky(K_new, lower=True) # wouldn't work now -- K_new is not symmetric anymore
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = np.linalg.eigh(K_new)

    # Construct c, so c*c^T = r.
    c = np.dot(evecs, np.diag(np.sqrt(evals)))
    # Convert the data to correlated random variables. 
    vbatch = np.dot(c, np.random.randn(c.shape[0], sample_size))

    vbatch_full = np.zeros((K_mat.shape[0], sample_size))
    vbatch_full[x_star, :] = vbatch
    
    mbatch_full = np.zeros((K_mat.shape[0], sample_size))
    mbatch_full[x_star, :] = K_new_mean + log_mean
    mbatch_full[x, :] = y
    
#     print(mbatch_full.shape)
#     print(vbatch_full.shape)
    
    vbatch_full = mbatch_full + vbatch_full
#     print(samples.shape)

    vbatch_full = vbatch_full.T.reshape(sample_size, ny, nx).squeeze()

    return vbatch_full