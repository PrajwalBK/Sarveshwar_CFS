"""
Advanced methods for calculating ground contact point from bounding box.

This module implements multiple computational approaches based on research
in monocular 3D object detection and ground plane projection.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import json
import os
from pathlib import Path


def calculate_image_coords_advanced(
    bbox: List[float],
    method: str = "learned_features"
) -> Tuple[float, float]:
    """
    Calculate image coordinates using advanced methods.
    
    Methods based on research:
    1. "bottom_center" - Standard bottom center point
    2. "bottom_intersection" - Intersection of bbox bottom with expected Y
    3. "learned_features" - Uses multiple bbox features with learned weights
    4. "perspective_corrected" - Accounts for perspective distortion
    5. "dimension_based" - Uses bbox dimensions to predict offset
    """
    if len(bbox) != 4:
        raise ValueError(f"Bbox must have 4 elements, got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    bottom_y = float(y2)
    
    if method == "bottom_center":
        return (center_x, bottom_y)
    
    elif method == "bottom_intersection":
        # This would need the expected Y coordinate, so not applicable here
        # But we can use a percentage from top
        y_contact = y1 + (bbox_height * 0.64)  # From calibration
        return (center_x, y_contact)
    
    elif method == "learned_features":
        # Use multiple features: center, bottom, edges, dimensions
        # This is a weighted combination approach
        # Weights would be learned from calibration data
        # For now, use a simple version
        weight_center = 0.3
        weight_right = 0.4
        weight_dimension = 0.3
        
        # Feature 1: Center X
        feature_center_x = center_x
        
        # Feature 2: Right edge with dimension-based offset
        # Larger bboxes might have different offsets
        if bbox_width > 800:
            dim_offset = bbox_width * 0.16
        else:
            dim_offset = bbox_width * 0.12
        feature_right_x = float(x2) + dim_offset
        
        # Feature 3: Dimension-based prediction
        # Use aspect ratio and size to predict
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        feature_dim_x = center_x + (bbox_width * 0.1 * aspect_ratio)
        
        # Weighted combination
        calc_x = (weight_center * feature_center_x + 
                 weight_right * feature_right_x + 
                 weight_dimension * feature_dim_x)
        
        # Y: Use calibrated percentage
        calc_y = y1 + (bbox_height * 0.64)
        
        return (calc_x, calc_y)
    
    elif method == "perspective_corrected":
        # Account for perspective: objects further away appear smaller
        # Use bbox area as proxy for distance
        bbox_area = bbox_width * bbox_height
        
        # Normalize area (assuming max area ~1000000 for close objects)
        normalized_area = min(bbox_area / 1000000.0, 1.0)
        
        # Further objects (smaller bbox) need larger offset
        distance_factor = 1.0 + (1.0 - normalized_area) * 0.5
        
        # X: Offset increases with distance
        x_offset = bbox_width * 0.16 * distance_factor
        calc_x = float(x2) + x_offset
        
        # Y: Percentage from top (calibrated)
        calc_y = y1 + (bbox_height * 0.64)
        
        return (calc_x, calc_y)
    
    elif method == "dimension_based":
        # Use bbox dimensions to predict the contact point
        # Based on the idea that contact point relates to object geometry
        
        # Calculate offset based on width and height
        width_factor = bbox_width / 1000.0  # Normalize
        height_factor = bbox_height / 1000.0
        
        # X offset: depends on width (wider objects might have different contact)
        x_offset_pct = 0.16 + (width_factor - 0.8) * 0.05  # Adjust based on width
        calc_x = float(x2) + (bbox_width * x_offset_pct)
        
        # Y: Use calibrated percentage
        calc_y = y1 + (bbox_height * 0.64)
        
        return (calc_x, calc_y)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def find_optimal_linear_combination(
    bboxes: List[List[float]],
    expected_coords: List[Tuple[float, float]]
) -> dict:
    """
    Find optimal linear combination of bbox features.
    
    Uses least squares to find weights for:
    - bbox center X
    - bbox right edge X
    - bbox left edge X
    - bbox width
    - bbox height
    - bbox area
    
    Returns optimal weights and error.
    """
    # Build feature matrix
    n_samples = len(bboxes)
    n_features = 6  # center_x, right_x, left_x, width, height, area
    
    X = np.zeros((n_samples, n_features))
    y_x = np.zeros(n_samples)
    y_y = np.zeros(n_samples)
    
    for i, (bbox, expected) in enumerate(zip(bboxes, expected_coords)):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2.0
        area = width * height
        
        # Features
        X[i, 0] = center_x
        X[i, 1] = float(x2)  # right edge
        X[i, 2] = float(x1)  # left edge
        X[i, 3] = width
        X[i, 4] = height
        X[i, 5] = area
        
        y_x[i] = expected[0]
        y_y[i] = expected[1]
    
    # Add bias term (intercept)
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    # Solve for X coordinates
    try:
        weights_x = np.linalg.lstsq(X_with_bias, y_x, rcond=None)[0]
        pred_x = X_with_bias @ weights_x
        error_x = np.mean(np.abs(pred_x - y_x))
    except:
        weights_x = None
        error_x = float('inf')
    
    # Solve for Y coordinates
    try:
        weights_y = np.linalg.lstsq(X_with_bias, y_y, rcond=None)[0]
        pred_y = X_with_bias @ weights_y
        error_y = np.mean(np.abs(pred_y - y_y))
    except:
        weights_y = None
        error_y = float('inf')
    
    total_error = np.sqrt(error_x**2 + error_y**2)
    
    return {
        'weights_x': weights_x,
        'weights_y': weights_y,
        'error_x': error_x,
        'error_y': error_y,
        'total_error': total_error,
        'features': ['bias', 'center_x', 'right_x', 'left_x', 'width', 'height', 'area']
    }


def calculate_with_linear_model(
    bbox: List[float],
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    features: List[str]
) -> Tuple[float, float]:
    """
    Calculate image coordinates using learned linear model.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2.0
    area = width * height
    
    # Build feature vector
    feature_vec = np.array([
        1.0,  # bias
        center_x,
        float(x2),  # right edge
        float(x1),  # left edge
        width,
        height,
        area
    ])
    
    calc_x = feature_vec @ weights_x
    calc_y = feature_vec @ weights_y
    
    return (float(calc_x), float(calc_y))


def load_bbox_coords_config(config_path: Optional[str] = None) -> Dict:
    """
    Load bbox-to-image-coords configuration from JSON file.
    
    Args:
        config_path: Path to config file, or None to use default
        
    Returns:
        Dict with method and weights
    """
    if config_path is None:
        # Default path: config/bbox_to_image_coords.json
        config_path = Path(__file__).parent.parent / "config" / "bbox_to_image_coords.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return {
            'method': 'adaptive_percentage',
            'linear_model': None
        }
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert weights to numpy arrays
    if 'linear_model' in config and config['linear_model']:
        config['linear_model']['weights_x'] = np.array(config['linear_model']['weights_x'])
        config['linear_model']['weights_y'] = np.array(config['linear_model']['weights_y'])
    
    return config


def calculate_image_coords_from_bbox_with_config(
    bbox: List[float],
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    debug: bool = False
) -> Tuple[float, float]:
    """
    Calculate image coordinates using configuration.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        config: Config dict (if None, will load from config_path)
        config_path: Path to config file (if config is None)
        debug: If True, print debug information
        
    Returns:
        Tuple of (x_img, y_img) image coordinates
    """
    if config is None:
        config = load_bbox_coords_config(config_path)
    
    method = config.get('method', 'adaptive_percentage')
    
    if method == 'linear_combination' and config.get('linear_model'):
        # Use linear model
        weights_x = config['linear_model']['weights_x']
        weights_y = config['linear_model']['weights_y']
        features = config['linear_model']['features']
        
        if debug:
            print(f"[bbox_to_image_coords] Using linear_combination method")
            print(f"  BBox: {bbox}")
        
        result = calculate_with_linear_model(bbox, weights_x, weights_y, features)
        
        if debug:
            print(f"  Calculated image coords: ({result[0]:.2f}, {result[1]:.2f})")
        
        return result
    else:
        # Fallback to other methods
        if debug:
            print(f"[bbox_to_image_coords] Using fallback method: {config.get('fallback_method', 'adaptive_percentage')}")
        from app.bbox_to_image_coords import calculate_image_coords_from_bbox
        fallback_method = config.get('fallback_method', 'adaptive_percentage')
        return calculate_image_coords_from_bbox(bbox, fallback_method)



