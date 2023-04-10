import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images ((N,D) arrays) as well as color images ((N,D,3) arrays)
        In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
        each image (2 channels for black and white and 3 channels for RGB)
        Image is the matrix X.

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images

        Return:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V^T: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
        """
        if X.ndim == 2:
            U, S, V_T = np.linalg.svd(X)
        else:
            N,D,_ = np.shape(X)
            U = np.zeros((N, N, 3))
            S = np.zeros((np.minimum(N, D), 3))
            V_T = np.zeros((D, D, 3))

            for col in range(3):
                u_channel, s_channel, v_channel = np.linalg.svd(X[:, :, col])
                U[:, :, col] = u_channel
                S[:, col] = s_channel
                V_T[:, :, col] = v_channel

        return U, S, V_T


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.

        Args:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            Xrebuild: (N,D) numpy array of reconstructed image / (N,D,3) numpy array for color images

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if U.ndim == 2:
            S = S[:k]
            U = U[ : , : k]
            V = V[ : k, : ]
            Xrebuild = np.matmul(U, np.matmul(np.diag(S), V)) 
        else:
            Xrebuild = np.zeros((U.shape[0], V.shape[0], 3))
            for c in range(3):
                Xrebuild[:, :, c] = np.matmul(U[:, :k, c], np.matmul(np.diag(S[:k, c]), V[:k, :, c]))
                
        return Xrebuild


    def compression_ratio(self, X, k): # [5pts]
        """
        Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        return (k *(X.shape[0]+X.shape[1]+1)) / (X.shape[0]*X.shape[1])


    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (min(N,D), ) numpy array black and white images / (min(N,D),3) numpy array for color images
           k: int, rank of approximation

        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        a = np.power(S[:k], 2).sum(axis=0, dtype=np.float) 
        b = np.power(S, 2).sum(axis=0, dtype=np.float)
        return a/b
        
