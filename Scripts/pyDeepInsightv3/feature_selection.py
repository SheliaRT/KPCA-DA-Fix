from typing import Union, Optional, Dict, Sequence
from .image_transformer import ImageTransformer
from .gene_attribution import rank_gene_attributions

import numpy as np
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader

import pytorch_grad_cam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchcam.methods._utils import locate_candidate_layer


class CAMFeatureSelector:
    """Extract important features from a model using class activation mapping
    (CAM).

    This class takes a trained model and it's training data to identify
    important features to prediction of each class.
    """

    def __init__(self, model: nn.Module,
                 it: ImageTransformer,
                 target_layer: Union[str, nn.Module] = None,
                 cam_method: str = "GradCAM", *,
                 map_domain: Optional[str] = None,
                 feature_coords: Optional[np.ndarray] = None,
                 gene_ids: Optional[Sequence[str]] = None) -> None:
        """Generate a CAMFeatureSelector instance

        Args:
            model: trained pytorch.nn.Module CNN model
            it: ImageTransformer class used to create the images that trained
                the model
            target_layer: the target layer of the model, the name of the
                target layer, or "None" for the layer to be selected
                automatically
            cam_method: name of the CAM method from pytorch-grad-cam
            map_domain: ``"source"`` or ``"target"`` for a domain-adaptive
                transformer. Legacy image transformers with ``coords()`` are
                still accepted.
            feature_coords: explicit ordered coordinates, when already saved.
            gene_ids: gene identifiers in exactly the same order as coordinates.
        """

        if cam_method not in CAM_FUNCTIONS.keys():
            raise ValueError(
                f"Unknown cam_method {cam_method}. Valid methods are "
                f"{CAM_FUNCTIONS.keys()}"
            )
        self.cam_method = cam_method
        if feature_coords is not None:
            coordinates = np.asarray(feature_coords)
        elif map_domain is None and callable(getattr(it, "coords", None)):
            coordinates = np.asarray(it.coords())
        elif map_domain in {"source", "target"}:
            get_coords = getattr(it, f"coords_{map_domain}", None)
            if not callable(get_coords):
                raise ValueError(f"Transformer does not expose {map_domain} coordinates")
            coordinates = np.asarray(get_coords())
        else:
            raise ValueError(
                "Adaptive transformers require map_domain='source' or 'target', "
                "or explicit feature_coords"
            )
        if (coordinates.ndim != 2 or coordinates.shape[1] != 2
                or not np.isfinite(coordinates).all()
                or not np.equal(coordinates, np.floor(coordinates)).all()):
            raise ValueError("Feature coordinates must be finite integer (row, column) pairs")
        self.feature_coords = coordinates.astype(int)
        provided_gene_ids = None if gene_ids is None else np.asarray(gene_ids).astype(str)
        fitted_gene_ids = None
        if map_domain in {"source", "target"}:
            fitted_gene_ids = getattr(it, f"_genes_{map_domain}", None)
            if fitted_gene_ids is not None:
                fitted_gene_ids = np.asarray(fitted_gene_ids).astype(str)
        if (provided_gene_ids is not None and fitted_gene_ids is not None
                and not np.array_equal(provided_gene_ids, fitted_gene_ids)):
            raise ValueError("gene_ids do not match the fitted adaptive map's feature order")
        self.gene_ids = provided_gene_ids if provided_gene_ids is not None else fitted_gene_ids
        if self.gene_ids is not None and self.gene_ids.shape != (len(self.feature_coords),):
            raise ValueError("gene_ids must follow the exact order of feature_coords")
        if self.gene_ids is not None and len(set(self.gene_ids.tolist())) != len(self.gene_ids):
            raise ValueError("gene_ids must be unique after the fitted feature filter")
        self.map_domain = map_domain
        self.model = model
        self.target_layer = self._resolve_target_layer(target_layer)

    def rank_class_features(self, cams: Dict[int, np.ndarray],
                            top_k: int = 10) -> Dict[int, list[dict]]:
        """Rank gene IDs from mean class CAMs using the main-paper rule."""
        if self.gene_ids is None:
            raise ValueError("gene_ids in fitted feature order are required for gene ranking")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        return {
            class_id: rank_gene_attributions(
                cam, self.gene_ids, self.feature_coords, top_k=top_k,
                normalize_mean_map=True,
            )
            for class_id, cam in cams.items()
        }

    def _resolve_target_layer(self,
                              target_layer: Union[str, nn.Module] = None
                              ) -> Optional[nn.Module]:
        """Convert layer name to layer or identify candidate layer if none
        given.

        Args:
            target_layer: the target layer of the model, the name of the
                target layer, or "None" for the layer to be selected
                automatically
        """
        resolved_target = None
        submodule_dict = dict(self.model.named_modules())
        if isinstance(target_layer, nn.Module):
            resolved_target = target_layer
        elif isinstance(target_layer, str):
            if target_layer not in submodule_dict.keys():
                raise ValueError(
                    f"Unable to find submodule {target_layer} in the model")
            else:
                resolved_target = submodule_dict[target_layer]
        elif target_layer is None:
            target_layer = locate_candidate_layer(self.model)
            resolved_target = submodule_dict[target_layer]

        return resolved_target

    def compute_cam(self, X: Tensor, y: Tensor,
                    batch_size: int = 1, use_cuda: bool = False,
                    ) -> np.ndarray:
        """Compute class activation map (CAM) for each image in X of classes y.

        Args:
            X: Tensor of input images
            y: Tensor of input labels
            batch_size: Batch size (default 1)
            use_cuda: Whether to use cuda for calculating CAMs
        Return:
            A numpy array of CAMs
        """
        activations = np.empty((0, X.shape[-2], X.shape[-1]))
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        func = CAM_FUNCTIONS[self.cam_method]
        self.model.eval()
        for i, data in enumerate(dl):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            cam = func(model=self.model, target_layers=[self.target_layer])
            targets = [ClassifierOutputTarget(label) for label in labels]
            grayscale_cam = cam(input_tensor=inputs, targets=targets)
            activations = np.append(activations, grayscale_cam, axis=0)

        return activations

    @staticmethod
    def flatten_cam(cams: np.ndarray,
                    method: str = "mean") -> np.ndarray:
        """Flatten multiple CAMs into a single CAM.

        Args:
            cams: ndarray of CAMs
            method: Method to flatten CAMs. When used with feature selection
                threshold, max is essentially a union operation, and min is an
                intersect operation.
        Returns:
            A numpy array
        """
        if method == "mean":
            flat_cam = np.mean(cams, axis=0)
        elif method == "max":
            flat_cam = np.max(cams, axis=0)
        elif method == "min":
            flat_cam = np.min(cams, axis=0)
        else:
            raise ValueError(f"Unknown CAM flatten method {method}")
        return flat_cam

    @staticmethod
    def flatten_classes(labels: np.ndarray, cam: np.ndarray,
                        method: str = "mean") -> Dict[int, np.ndarray]:
        """Flatten CAMs across all classes

        Args:
            labels: class labels for cam array
            cam: ndarray of CAMs
            method: method to merge CAMs per class passed to .flatten_cam()
        Returns:
             A dictionary with class labels as keys and a single CAM as
             the value
        """
        cats = np.unique(labels)
        cat_cam = {}
        for cat in cats:
            cat_act = cam[np.where(labels == cat)[0], :, :]
            flat_act = CAMFeatureSelector.flatten_cam(
                cams=cat_act, method=method)
            cat_cam[cat] = flat_act
        return cat_cam

    def calculate_class_activations(self, X: Tensor, y: Tensor,
                                    batch_size: int = 1, flatten_method="mean"
                                    ) -> Dict[int, np.ndarray]:
        """Calculate CAM for each input then flatten for each class.

        Args:
            X: Tensor of input images
            y: Tensor of input labels
            batch_size: Batch size (default 1)
            flatten_method: Method to flatten CAMs for each class. 'max' is
                essentially a union operation, and 'min' is an
                intersect operation.
        Returns:
            A dictionary with classes as keys and the flattened CAM as values
        """
        use_cuda = X.is_cuda
        activations = self.compute_cam(X=X, y=y, batch_size=batch_size)
        y_cpu = y.detach().cpu().numpy()
        cat_cam = self.flatten_classes(y_cpu, activations,
                                       method=flatten_method)

        return cat_cam

    def select_class_features(self, cams: Dict[int, np.ndarray],
                              threshold: float = 0.6
                              ) -> tuple[Dict[int, np.ndarray],
                                         Dict[int, np.ndarray],
                                         Dict[int, np.ndarray]]:
        """Select mapped features with strict ``score > threshold``.

        Return per-class feature indices, scores and coordinates in the exact
        fitted feature order. If genes collide at one pixel, every gene at
        that pixel is returned with the shared pixel score.
        """
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        class_idx: Dict[int, np.ndarray] = {}
        class_thr: Dict[int, np.ndarray] = {}
        class_coords: Dict[int, np.ndarray] = {}
        for cat, cam in cams.items():
            cam = np.asarray(cam, dtype=float)
            if cam.ndim != 2 or not np.isfinite(cam).all():
                raise ValueError("Each class CAM must be a finite height x width array")
            low, high = float(cam.min()), float(cam.max())
            if high > low:
                cam = (cam - low) / (high - low)
            else:
                cam = np.zeros_like(cam)
            coords = self.feature_coords
            if len(coords) and (np.any(coords < 0)
                    or np.any(coords[:, 0] >= cam.shape[0])
                    or np.any(coords[:, 1] >= cam.shape[1])):
                raise ValueError("Feature coordinates fall outside the CAM dimensions")
            scores = cam[coords[:, 0], coords[:, 1]]
            indices = np.flatnonzero(scores > threshold)
            class_idx[cat] = indices
            class_thr[cat] = scores[indices]
            class_coords[cat] = coords[indices]
        return class_idx, class_thr, class_coords


CAM_FUNCTIONS = {
    "GradCAM": pytorch_grad_cam.GradCAM,
    "AblationCAM": pytorch_grad_cam.AblationCAM,
    "XGradCAM": pytorch_grad_cam.XGradCAM,
    "GradCAMPlusPlus": pytorch_grad_cam.GradCAMPlusPlus,
    "ScoreCAM": pytorch_grad_cam.ScoreCAM,
    "LayerCAM": pytorch_grad_cam.LayerCAM,
    "EigenCAM": pytorch_grad_cam.EigenCAM,
    "EigenGradCAM": pytorch_grad_cam.EigenGradCAM,
    "FullGrad": pytorch_grad_cam.FullGrad
}
