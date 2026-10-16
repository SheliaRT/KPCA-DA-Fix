from typing import Union, Any, Optional, Tuple
from typing_extensions import Protocol
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import inspect

from .utils._ags import asymmetric_greedy_search

# Import KPCA_DA - adjust path based on your project structure
try:
    from kpcada import KPCA_DA
except ImportError:
    import sys
    import os
    # Try to import from Scripts folder
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'Scripts'))
    from kpcada import KPCA_DA


class ManifoldLearner(Protocol):
    def fit_transform(self: 'ManifoldLearner',
                      X: np.ndarray) -> np.ndarray: pass


class DomainAdaptiveImageTransformer:
    """Transform features to image matrices using KPCA-DA for domain adaptation
    
    This class takes in source and target datasets and converts them to CNN
    compatible 'image' matrices. The feature extractor uses KPCA-DA to align
    the target domain to the source domain before creating images.
    """

    def __init__(self, 
                 kpca_da_params: Optional[dict] = None,
                 discretization: str = 'bin',
                 pixels: Union[int, Tuple[int, int]] = (224, 224)) -> None:
        """Generate a DomainAdaptiveImageTransformer instance

        Args:
            kpca_da_params: dictionary of KPCA_DA parameters. If None, uses defaults:
                {
                    'n_components': 2,
                    'kernel': 'rbf',
                    'kpca_gamma': 0.2,
                    'mmd_gamma': 0.5,
                    'lambda_reg': 0.1,
                    'lr': 0.0001,
                    'epochs': 100,
                    'verbose': True
                }
            discretization: string of values ('bin', 'assignment'). Defines
                the method for discretizing dimensionally reduced data to pixel
                coordinates.
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
        """
        # Set default KPCA-DA parameters
        if kpca_da_params is None:
            kpca_da_params = {
                'n_components': 2,
                'kernel': 'rbf',
                'kpca_gamma': 0.2,
                'mmd_gamma': 0.5,
                'lambda_reg': 0.1,
                'lr': 0.0001,
                'epochs': 100,
                'verbose': True
            }
        
        self._kpca_da = KPCA_DA(**kpca_da_params)
        self._dm = self._parse_discretization(discretization)
        self._pixels = self._parse_pixels(pixels)
        
        # Storage for fitted transformations
        self._xrot_source = np.empty(0)
        self._xrot_target = np.empty(0)
        self._coords_source = np.empty(0)
        self._coords_target = np.empty(0)
        self._mbr_rot = np.empty(0)
        
        # Store genes for reference
        self._genes_source = None
        self._genes_target = None

    @staticmethod
    def _parse_pixels(pixels: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Check and correct pixel parameter

        Args:
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        return pixels

    @classmethod
    def _parse_discretization(cls, method: str):
        """Validate the discretization value passed to the
        constructor method and return correct function

        Args:
            method: string of value ('bin', 'assignment')

        Returns:
            function
        """
        if method == 'bin':
            func = cls.coordinate_binning
        elif method == 'assignment' or method == 'lsa':
            func = cls.coordinate_optimal_assignment
        elif method == 'ags':
            func = cls.coordinate_heuristic_assignment
        else:
            raise ValueError(f"discretization method '{method}' not valid")
        return func

    @classmethod
    def coordinate_binning(cls, position: np.ndarray,
                           px_size: Tuple[int, int]) -> np.ndarray:
        """Determine the pixel locations of each feature based on the overlap of
        feature position and pixel locations.

        Args:
            position: a 2d array of feature coordinates
            px_size: tuple with image dimensions

        Returns:
            a 2d array of feature to pixel mappings
        """
        scaled = cls.scale_coordinates(position, px_size)
        px_binned = np.floor(scaled).astype(int)
        # Need to move maximum values into the lower bin
        px_binned[:, 0][px_binned[:, 0] == px_size[0]] = px_size[0] - 1
        px_binned[:, 1][px_binned[:, 1] == px_size[1]] = px_size[1] - 1
        return px_binned

    @staticmethod
    def lsap_optimal_solution(cost_matrix):
        return linear_sum_assignment(cost_matrix)

    @staticmethod
    def lsap_heuristic_solution(cost_matrix):
        return asymmetric_greedy_search(cost_matrix,
                                        shuffle=True,
                                        minimize=True)

    @classmethod
    def coordinate_optimal_assignment(cls, position: np.ndarray,
                                      px_size: Tuple[int, int]) -> np.ndarray:
        """Determine the pixel location of each feature using a linear sum
        assignment problem solution on the exponential on the euclidean
        distances between the features and the pixels

        Args:
            position: a 2d array of feature coordinates
            px_size: tuple with image dimensions

        Returns:
            a 2d array of feature to pixel mappings
        """
        scaled = cls.scale_coordinates(position, px_size)
        px_centers = cls.calculate_pixel_centroids(px_size)

        # calculate distances
        k = np.prod(px_size)
        clustered = scaled.shape[0] > k
        if clustered:
            kmeans = KMeans(n_clusters=k).fit(scaled)
            cl_labels = kmeans.labels_
            cl_centers = kmeans.cluster_centers_
            dist = cdist(cl_centers, px_centers, metric='euclidean')
        else:
            dist = cdist(scaled, px_centers, metric='euclidean')
        # assignment of features/clusters to pixels
        lsa = cls.lsap_optimal_solution(dist)
        px_assigned = np.empty(scaled.shape, dtype=int)
        for i in range(scaled.shape[0]):
            if clustered:
                j = cl_labels[i]
            else:
                j = i
            ki = lsa[1][j]
            xi = ki % px_size[0]
            yi = ki // px_size[0]
            px_assigned[i] = [yi, xi]
        return px_assigned

    @classmethod
    def coordinate_heuristic_assignment(cls, position: np.ndarray,
                                        px_size: Tuple[int, int]) -> np.ndarray:

        scaled = cls.scale_coordinates(position, px_size)
        px_centers = cls.calculate_pixel_centroids(px_size)

        # calculate distances
        # AGS requires asymmetric assignment so k must be less than pixels
        k = np.prod(px_size) - 1
        clustered = scaled.shape[0] > k
        if clustered:
            kmeans = KMeans(n_clusters=k).fit(scaled)
            cl_labels = kmeans.labels_
            cl_centers = kmeans.cluster_centers_
            dist = cdist(cl_centers, px_centers, metric='euclidean')
        else:
            dist = cdist(scaled, px_centers, metric='euclidean')
        # assignment of features/clusters to pixels
        lsa = cls.lsap_heuristic_solution(dist)
        px_assigned = np.empty(scaled.shape, dtype=int)
        for i in range(scaled.shape[0]):
            if clustered:
                j = cl_labels[i]
            else:
                j = i
            ki = lsa[1][j]
            xi = ki % px_size[0]
            yi = ki // px_size[0]
            px_assigned[i] = [yi, xi]
        return px_assigned

    @staticmethod
    def calculate_pixel_centroids(px_size: Tuple[int, int]) -> np.ndarray:
        """Generate a 2d array of the centroid of each pixel

        Args:
            px_size: tuple with image dimensions

        Returns:
            a 2d array of pixel centroid locations
        """
        px_map = np.empty((np.prod(px_size), 2))
        for i in range(0, px_size[0]):
            for j in range(0, px_size[1]):
                px_map[i * px_size[0] + j] = [i, j]
        px_centroid = px_map + 0.5
        return px_centroid

    def fit(self, X_source: np.ndarray, X_target: np.ndarray,
            genes_source: np.ndarray, genes_target: np.ndarray,
            y_source: Optional[ArrayLike] = None,
            y_target: Optional[ArrayLike] = None,
            plot: bool = False):
        """Train the domain adaptive image transformer from source and target sets

        Args:
            X_source: {array-like, sparse matrix} of shape (n_samples_source, n_features)
            X_target: {array-like, sparse matrix} of shape (n_samples_target, n_features)
            genes_source: array of gene names for source dataset
            genes_target: array of gene names for target dataset
            y_source: Ignored. Present for continuity
            y_target: Ignored. Present for continuity
            plot: boolean of whether to produce a scatter plot showing the
                feature reduction, hull points, and minimum bounding rectangle

        Returns:
            self: object
        """
        # Store genes for reference
        self._genes_source = genes_source
        self._genes_target = genes_target
        
        # Apply KPCA-DA for domain adaptation
        # Note: KPCA_DA expects (genes x samples) format
        print("Applying KPCA-DA for domain adaptation...")
        Z_source, Z_target = self._kpca_da.fit_transform(
            X_source=X_source.T,  # Transpose to (genes x samples)
            X_target=X_target.T,  # Transpose to (genes x samples)
            genes_source=genes_source,
            genes_target=genes_target
        )
        
        # Z_source and Z_target are now (n_samples x n_components)
        print(f"Adapted embeddings - Source: {Z_source.shape}, Target: {Z_target.shape}")
        
        # Combine source and target for unified coordinate system
        # This ensures both datasets map to the same pixel space
        x_combined = np.vstack([Z_source, Z_target])
        
        # Get the convex hull for the combined points
        chvertices = ConvexHull(x_combined).vertices
        hull_points = x_combined[chvertices]
        
        # Determine the minimum bounding rectangle
        mbr, mbr_rot = self._minimum_bounding_rectangle(hull_points)
        self._mbr_rot = mbr_rot
        
        # Rotate both source and target using the same rotation matrix
        self._xrot_source = np.dot(mbr_rot, Z_source.T).T
        self._xrot_target = np.dot(mbr_rot, Z_target.T).T
        
        # Calculate coordinates for both datasets
        self._calculate_coords()
        
        # Plot rotation diagram if requested
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot original KPCA-DA space
            axes[0].scatter(Z_source[:, 0], Z_source[:, 1], s=30,
                          c='blue', alpha=0.6, label='Source')
            axes[0].scatter(Z_target[:, 0], Z_target[:, 1], s=30,
                          c='red', alpha=0.6, label='Target', marker='^')
            axes[0].fill(x_combined[chvertices, 0], x_combined[chvertices, 1],
                       edgecolor='g', fill=False, linewidth=2, label='Convex Hull')
            axes[0].fill(mbr[:, 0], mbr[:, 1], edgecolor='orange', 
                       fill=False, linewidth=2, label='MBR')
            axes[0].set_title('KPCA-DA Adapted Space', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Component 1')
            axes[0].set_ylabel('Component 2')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_aspect('equal', adjustable='box')
            
            # Plot rotated space
            xrot_combined = np.vstack([self._xrot_source, self._xrot_target])
            n_source = self._xrot_source.shape[0]
            axes[1].scatter(self._xrot_source[:, 0], self._xrot_source[:, 1], 
                          s=30, c='blue', alpha=0.6, label='Source')
            axes[1].scatter(self._xrot_target[:, 0], self._xrot_target[:, 1], 
                          s=30, c='red', alpha=0.6, label='Target', marker='^')
            axes[1].set_title('Rotated Space for Pixel Mapping', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Rotated Component 1')
            axes[1].set_ylabel('Rotated Component 2')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            plt.show()
        
        return self

    @property
    def pixels(self) -> Tuple[int, int]:
        """The image matrix dimensions

        Returns:
            tuple: the image matrix dimensions (height, width)
        """
        return self._pixels

    @pixels.setter
    def pixels(self, pixels: Union[int, Tuple[int, int]]) -> None:
        """Set the image matrix dimension

        Args:
            pixels: int or tuple with the dimensions (height, width)
            of the image matrix
        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        self._pixels = pixels
        # Recalculate coordinates if already fit
        if hasattr(self, '_coords_source'):
            self._calculate_coords()

    @staticmethod
    def scale_coordinates(coords: np.ndarray, dim_max: ArrayLike) -> np.ndarray:
        """Transforms a list of n-dimensional coordinates by scaling them
        between zero and the given dimensional maximum

        Args:
            coords: a 2d ndarray of coordinates
            dim_max: a list of maximum ranges for each dimension of coords

        Returns:
            a 2d ndarray of scaled coordinates
        """
        data_min = coords.min(axis=0)
        data_max = coords.max(axis=0)
        std = (coords - data_min) / (data_max - data_min)
        scaled = np.multiply(std, dim_max)
        return scaled

    def _calculate_coords(self) -> None:
        """Calculate the matrix coordinates of each feature based on the
        pixel dimensions for both source and target datasets.
        """
        # Scale and discretize source coordinates
        scaled_source = self.scale_coordinates(self._xrot_source, self._pixels)
        self._coords_source = self._dm(scaled_source, self._pixels)
        
        # Scale and discretize target coordinates
        scaled_target = self.scale_coordinates(self._xrot_target, self._pixels)
        self._coords_target = self._dm(scaled_target, self._pixels)

    def transform_source(self, X_source: np.ndarray, img_format: str = 'rgb',
                        empty_value: int = 0) -> np.ndarray:
        """Transform the source expression matrix into image matrices

        Args:
            X_source: {array-like} of shape (n_samples, n_features)
                Original expression data (e.g., X_train_norm). The method maps
                these expression values to pixel coordinates calculated from KPCA-DA.
            img_format: The format of the image matrix to return.
                'scalar' returns an array of shape (M, N). 'rgb' returns
                a numpy.ndarray of shape (M, N, 3) that is compatible with PIL.
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0.

        Returns:
            A numpy array of n_samples image matrices of dimensions set by
            the pixel parameter
        """
        return self._transform_helper(X_source, self._coords_source, 
                                      img_format, empty_value)

    def transform_target(self, X_target: np.ndarray, img_format: str = 'rgb',
                        empty_value: int = 0) -> np.ndarray:
        """Transform the target expression matrix into image matrices

        Args:
            X_target: {array-like} of shape (n_samples, n_features)
                Original expression data (e.g., X_adapt_norm). The method maps
                these expression values to pixel coordinates calculated from KPCA-DA.
            img_format: The format of the image matrix to return.
                'scalar' returns an array of shape (M, N). 'rgb' returns
                a numpy.ndarray of shape (M, N, 3) that is compatible with PIL.
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0.

        Returns:
            A numpy array of n_samples image matrices of dimensions set by
            the pixel parameter
        """
        return self._transform_helper(X_target, self._coords_target, 
                                      img_format, empty_value)

    def _transform_helper(self, X: np.ndarray, coords: np.ndarray,
                         img_format: str = 'rgb',
                         empty_value: int = 0) -> np.ndarray:
        """Helper method to transform data to images using given coordinates

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            coords: coordinate mappings for features
            img_format: 'scalar' or 'rgb'
            empty_value: value for unmapped pixels

        Returns:
            Array of image matrices
        """
        img_coords = pd.DataFrame(np.vstack((
            coords.T,
            X
        )).T).groupby([0, 1], as_index=False).mean()

        img_list = []
        blank_mat = np.zeros(self._pixels)
        if empty_value != 0:
            blank_mat[:] = empty_value
            
        for z in range(2, img_coords.shape[1]):
            img_matrix = blank_mat.copy()
            img_matrix[img_coords[0].astype(int),
                       img_coords[1].astype(int)] = img_coords[z]
            img_list.append(img_matrix)

        if img_format == 'rgb':
            img_matrices = np.array([self._mat_to_rgb(m) for m in img_list])
        elif img_format == 'scalar':
            img_matrices = np.stack(img_list)
        else:
            raise ValueError(f"'{img_format}' not accepted for img_format")

        return img_matrices

    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray,
                     genes_source: np.ndarray, genes_target: np.ndarray,
                     **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Train the transformer and return adapted feature matrices for both datasets

        Args:
            X_source: {array-like, sparse matrix} of shape (n_samples_source, n_features)
            X_target: {array-like, sparse matrix} of shape (n_samples_target, n_features)
            genes_source: array of gene names for source dataset
            genes_target: array of gene names for target dataset
            **kwargs: additional arguments (e.g., plot=True to visualize)

        Returns:
            Tuple of (Z_source, Z_target) - adapted feature matrices of shape 
            (n_samples, n_components) where n_components is typically 2
        """
        self.fit(X_source, X_target, genes_source, genes_target,
                plot=kwargs.get('plot', False))
        
        # Return the adapted feature matrices from KPCA-DA
        return self._kpca_da.Z_s_.T, self._kpca_da.Z_t_.T

    def coords_source(self) -> np.ndarray:
        """Get source feature coordinates

        Returns:
            ndarray: the pixel coordinates for source features
        """
        return self._coords_source.copy()

    def coords_target(self) -> np.ndarray:
        """Get target feature coordinates

        Returns:
            ndarray: the pixel coordinates for target features
        """
        return self._coords_target.copy()

    def feature_density_matrix_source(self) -> np.ndarray:
        """Generate image matrix with source feature counts per pixel

        Returns:
            img_matrix (ndarray): matrix with feature counts per pixel
        """
        fdmat = np.zeros(self._pixels)
        np.add.at(fdmat, tuple(self._coords_source.T), 1)
        return fdmat

    def feature_density_matrix_target(self) -> np.ndarray:
        """Generate image matrix with target feature counts per pixel

        Returns:
            img_matrix (ndarray): matrix with feature counts per pixel
        """
        fdmat = np.zeros(self._pixels)
        np.add.at(fdmat, tuple(self._coords_target.T), 1)
        return fdmat

    @property
    def kpca_da(self) -> KPCA_DA:
        """Access the underlying KPCA_DA model

        Returns:
            KPCA_DA: the fitted KPCA_DA model
        """
        return self._kpca_da

    @property
    def loss_history(self) -> list:
        """Get the KPCA-DA optimization loss history

        Returns:
            list: loss values per epoch
        """
        if self._kpca_da.loss_history_ is not None:
            return self._kpca_da.loss_history_
        return []

    @staticmethod
    def _minimum_bounding_rectangle(hull_points: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the smallest bounding rectangle for a set of points.

        Modified from JesseBuesking at https://stackoverflow.com/a/33619018
        Returns a set of points representing the corners of the bounding box.

        Args:
            hull_points : an nx2 matrix of hull coordinates

        Returns:
            (tuple): tuple containing
                coords (ndarray): coordinates of the corners of the rectangle
                rotmat (ndarray): rotation matrix to align edges of rectangle
                    to x and y
        """
        pi2 = np.pi / 2
        # calculate edge angles
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)
        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            -np.sin(angles),
            np.sin(angles),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))
        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)
        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)
        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)
        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        rotmat = rotations[best_idx]
        # generate coordinates
        coords = np.zeros((4, 2))
        coords[0] = np.dot([x1, y2], rotmat)
        coords[1] = np.dot([x2, y2], rotmat)
        coords[2] = np.dot([x2, y1], rotmat)
        coords[3] = np.dot([x1, y1], rotmat)

        return coords, rotmat

    @staticmethod
    def _mat_to_rgb(mat: np.ndarray) -> np.ndarray:
        """Convert image matrix to numpy rgb format

        Args:
            mat: {array-like} (M, N)

        Returns:
            An numpy.ndarray (M, N, 3) with original values repeated across
            RGB channels.
        """
        return np.repeat(mat[:, :, np.newaxis], 3, axis=2)
