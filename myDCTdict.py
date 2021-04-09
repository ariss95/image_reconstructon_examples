import numpy as np

def to_img(D):
    """
    Transforms the dictionary columns into patches and orders them for plotting purposes.
    Returns:
    Reordered dictionary matrix
    """
    # dictionary dimensions
    
    n, K = D.shape
    M = D 
    # stretch atoms
    for k in range(K):
        M[:, k] = M[:, k] - (M[:, k].min())
        if M[:, k].max():
            M[:, k] = M[:, k] / D[:, k].max()
    n_r = int(np.sqrt(n))

    # patches per row / column
    K_r = int(np.sqrt(K))

    # we need n_r*K_r+K_r+1 pixels in each direction
    dim = n_r * K_r + K_r + 1
    V = np.ones((dim, dim)) * np.min(D)

    # compute the patches
    patches = [np.reshape(D[:, i], (n_r, n_r)) for i in range(K)]

    # place patches
    for i in range(K_r):
        for j in range(K_r):
            V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
                i * K_r + j]
    return V

def dctii(v, normalized=True, sampling_factor=None):
    """
    Computes the inverse discrete cosine transform of type II,
    cf. https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    Args:
        v: Input vector to transform
        normalized: Normalizes the output to make output orthogonal
        sampling_factor: Can be used to "oversample" the input to create overcomplete dictionaries
    Returns:
        Discrete cosine transformed vector
    """
    n = v.shape[0]
    if sampling_factor:
        K = sampling_factor
    else:
        K = n
    
    y = np.array([sum(np.multiply(v, np.cos((0.5 + np.arange(n)) * k * np.pi / K))) for k in range(K)])
    if normalized:
        y[0] = 1 / np.sqrt(2) * y[0]
        y = np.sqrt(2 / n) * y
    return y
def DCTDictionary(K, n):
  
    H = np.zeros((K, n))
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        H[:, i] = dctii(v, True, sampling_factor=K)

    H = H.T
    DCT = np.kron(H.T, H.T)
    D = to_img(DCT)
    return D