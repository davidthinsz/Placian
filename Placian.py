import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh
import gudhi as gd


class CombinatorialLaplacian:
    """A class used to construct combinatorial Laplacians from GUDHI simplex trees."""

    def __init__(self, simplicial_complex):
        self.dimension = simplicial_complex.dimension()
        self.n_simplices = [[] for i in range(self.dimension + 1)]
        self.simplex_coord = {}

        simplices = simplicial_complex.get_simplices()
        for simplex_tuple in simplices:
            simplex = tuple(simplex_tuple[0])
            simplex_dim = len(simplex) - 1
            self.simplex_coord[simplex] = len(self.n_simplices[simplex_dim])
            self.n_simplices[simplex_dim].append(simplex)

    def get_boundary(self, q):
        if q > self.dimension:
            raise ValueError(
                f"The value of q needs to be less than or equal to the dimension of the simplicial complex (dimension = {self.dimension}).")
        elif q <= 0:
            raise ValueError(f"The value of q needs to be greater than 0.")

        from_simplices = self.n_simplices[q]
        to_simplices = self.n_simplices[q - 1]

        boundary_matrix = dok_matrix((len(to_simplices), len(from_simplices)), dtype=np.float64)

        for from_simplex in from_simplices:
            # Row index of matrix
            j = self.simplex_coord[from_simplex]
            for omit_idx in range(len(from_simplex)):
                mapped_simplex = from_simplex[:omit_idx] + from_simplex[(omit_idx + 1):]
                i = self.simplex_coord[mapped_simplex]
                sign = 1 if omit_idx % 2 == 0 else -1
                boundary_matrix[i, j] = sign

        return boundary_matrix.tocsr()

    def get_laplacian(self, q):
        if q >= self.dimension:
            raise ValueError(
                f"The value of q needs to be less than the dimension of the simplicial complex (dimension = {self.dimension}).")
        elif q < 0:
            raise ValueError(f"The value of q needs to be non-negative.")

        up_boundary = self.get_boundary(q + 1)
        up_laplacian = up_boundary.dot(up_boundary.T)
        laplacian = up_laplacian
        if q > 0:
            down_boundary = self.get_boundary(q)
            down_laplacian = down_boundary.T.dot(down_boundary)
            laplacian += down_laplacian
        return laplacian

    def get_simplices(self, n):
        return self.n_simplices[n]


class PersistentLaplacian:
    def __init__(self):
        pass
