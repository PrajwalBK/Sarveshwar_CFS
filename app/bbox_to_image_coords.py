"""
Calculate correct image coordinates from bounding box.

This module provides functions to calculate the ground contact point
image coordinates from YOLO bounding box coordinates.
"""

from typing import Tuple, List, Optional
import numpy as np


def calculate_image_coords_from_bbox(
    bbox: List[float],
    method: str = "adaptive_percentage"
) -> Tuple[float, float]:
    """
    Calculate image coordinates (ground contact point) from bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        method: Calculation method:
            - "bottom_center": Center X, bottom Y (simple)
            - "right_bottom": Right edge X, bottom Y
            - "adaptive_percentage": Adaptive percentage-based offsets
            - "fixed_offset": Fixed pixel offsets from right-bottom
            - "from_left_percentage": Percentage from left edge
    
    Returns:
        Tuple of (x_img, y_img) image coordinates
    """
    if len(bbox) != 4:
        raise ValueError(f"Bbox must have 4 elements, got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    center_x = (x1 + x2) / 2.0
    
    if method == "bottom_center":
        # Simple: center X, bottom Y
        return (center_x, float(y2))
    
    elif method == "right_bottom":
        # Right edge, bottom
        return (float(x2), float(y2))
    
    elif method == "adaptive_percentage":
        # Adaptive percentage-based (like main_trt_demo.py)
        # X: Offset from right edge based on bbox width
        if bbox_width > 800:
            x_offset_percent = 0.15
        elif bbox_width > 600:
            x_offset_percent = 0.13
        else:
            x_offset_percent = 0.12
        
        ground_x = float(x2) + (bbox_width * x_offset_percent)
        
        # Y: Percentage from top based on bbox height
        if bbox_height > 900:
            y_percent = 0.39
        elif bbox_height > 700:
            y_percent = 0.385
        else:
            y_percent = 0.38
        
        ground_y = y1 + (bbox_height * y_percent)
        return (ground_x, ground_y)
    
    elif method == "fixed_offset":
        # Fixed pixel offsets (needs calibration)
        x_offset = 100  # Default, should be calibrated
        y_offset = -200  # Default, should be calibrated
        return (float(x2) + x_offset, float(y2) + y_offset)
    
    elif method == "from_left_percentage":
        # Percentage from left edge
        x_pct = 1.0  # 100% = right edge
        y_pct = 0.38  # 38% from top
        return (x1 + (bbox_width * x_pct), y1 + (bbox_height * y_pct))
    
    elif method.startswith("optimal_percentage:"):
        # Configurable percentage offsets: "optimal_percentage:x_pct:y_pct:reference"
        # Example: "optimal_percentage:0.21:0.38:center"
        # Reference can be: "right_edge", "left_edge", or "center"
        parts = method.split(":")
        if len(parts) < 3:
            raise ValueError(f"optimal_percentage method must be in format 'optimal_percentage:x_pct:y_pct[:reference]', got: {method}")
        x_offset_pct = float(parts[1])
        y_offset_pct = float(parts[2])
        reference = parts[3] if len(parts) > 3 else "right_edge"
        
        # Calculate X based on reference point
        if reference == "right_edge":
            ground_x = float(x2) + (bbox_width * x_offset_pct)
        elif reference == "left_edge":
            ground_x = float(x1) + (bbox_width * x_offset_pct)
        elif reference == "center":
            ground_x = center_x + (bbox_width * x_offset_pct)
        else:
            ground_x = float(x2) + (bbox_width * x_offset_pct)
        
        ground_y = y1 + (bbox_height * y_offset_pct)
        return (ground_x, ground_y)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calibrate_bbox_method(
    bboxes: List[List[float]],
    expected_coords: List[Tuple[float, float]],
    methods_to_test: Optional[List[str]] = None
) -> dict:
    """
    Calibrate the best bbox-to-image-coords calculation method.
    
    Args:
        bboxes: List of bboxes [[x1,y1,x2,y2], ...]
        expected_coords: List of expected (x, y) image coordinates
        methods_to_test: List of method names to test, or None for all
    
    Returns:
        Dict with best method name and statistics
    """
    if methods_to_test is None:
        methods_to_test = [
            "bottom_center",
            "right_bottom",
            "adaptive_percentage",
            "from_left_percentage"
        ]
    
    method_errors = {method: [] for method in methods_to_test}
    
    for bbox, expected in zip(bboxes, expected_coords):
        for method in methods_to_test:
            try:
                calc_coords = calculate_image_coords_from_bbox(bbox, method)
                error = np.sqrt(
                    (calc_coords[0] - expected[0])**2 +
                    (calc_coords[1] - expected[1])**2
                )
                method_errors[method].append(error)
            except Exception as e:
                method_errors[method].append(float('inf'))
    
    # Find best method
    avg_errors = {
        method: np.mean(errors) if errors else float('inf')
        for method, errors in method_errors.items()
    }
    
    best_method = min(avg_errors.items(), key=lambda x: x[1])
    
    return {
        'best_method': best_method[0],
        'avg_error': best_method[1],
        'all_errors': method_errors,
        'method_stats': {
            method: {
                'avg': np.mean(errors) if errors else float('inf'),
                'max': max(errors) if errors else float('inf'),
                'min': min(errors) if errors else float('inf'),
                'std': np.std(errors) if errors else float('inf')
            }
            for method, errors in method_errors.items()
        }
    }


def find_optimal_percentage_offsets(
    bboxes: List[List[float]],
    expected_coords: List[Tuple[float, float]],
    x_range: Tuple[float, float] = (0.0, 0.3),
    y_range: Tuple[float, float] = (0.3, 0.5),
    step: float = 0.01,
    reference_point: str = "right_edge"
) -> dict:
    """
    Find optimal percentage offsets for bbox-to-image-coords calculation.
    
    Tests different percentage offsets to find the best fit.
    
    Args:
        bboxes: List of bboxes
        expected_coords: List of expected image coordinates
        x_range: Range of X offset percentages to test
        y_range: Range of Y offset percentages to test (from top)
        step: Step size for testing
        reference_point: Reference point for X calculation:
            - "right_edge": Offset from right edge (x2)
            - "left_edge": Offset from left edge (x1)
            - "center": Offset from center
    
    Returns:
        Dict with optimal offsets and error statistics
    """
    best_error = float('inf')
    best_x_pct = 0.0
    best_y_pct = 0.38
    best_reference = reference_point
    
    x_pcts = np.arange(x_range[0], x_range[1] + step, step)
    y_pcts = np.arange(y_range[0], y_range[1] + step, step)
    
    all_errors = []
    
    for x_pct in x_pcts:
        for y_pct in y_pcts:
            errors = []
            for bbox, expected in zip(bboxes, expected_coords):
                x1, y1, x2, y2 = bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center_x = (x1 + x2) / 2.0
                
                # Calculate X based on reference point
                if reference_point == "right_edge":
                    calc_x = float(x2) + (bbox_width * x_pct)
                elif reference_point == "left_edge":
                    calc_x = float(x1) + (bbox_width * x_pct)
                elif reference_point == "center":
                    calc_x = center_x + (bbox_width * x_pct)
                else:
                    calc_x = float(x2) + (bbox_width * x_pct)
                
                calc_y = y1 + (bbox_height * y_pct)
                
                error = np.sqrt(
                    (calc_x - expected[0])**2 +
                    (calc_y - expected[1])**2
                )
                errors.append(error)
            
            avg_error = np.mean(errors)
            all_errors.append({
                'x_pct': x_pct,
                'y_pct': y_pct,
                'reference': reference_point,
                'avg_error': avg_error,
                'max_error': max(errors),
                'min_error': min(errors)
            })
            
            if avg_error < best_error:
                best_error = avg_error
                best_x_pct = x_pct
                best_y_pct = y_pct
    
    return {
        'optimal_x_pct': best_x_pct,
        'optimal_y_pct': best_y_pct,
        'optimal_reference': best_reference,
        'avg_error': best_error,
        'all_results': sorted(all_errors, key=lambda x: x['avg_error'])[:10]  # Top 10
    }



