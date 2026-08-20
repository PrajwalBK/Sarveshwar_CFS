"""
3D Camera Calibration Tool

Interactive tool for calibrating camera intrinsic and extrinsic parameters
for 3D projection. Uses ground-level reference points with GPS coordinates
to solve for camera pose (PnP problem).

Usage:
    python tools/calibrate_3d.py --image path/to/image.jpg --save config/calib/camera-01_3d.json
"""

import cv2
import numpy as np
import json
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path to import gps_utils
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import gps_utils
try:
    from app.gps_utils import gps_to_meters, validate_gps_coordinate
except ImportError:
    print("Warning: app.gps_utils not found. GPS coordinate validation disabled.")
    def validate_gps_coordinate(lat, lon):
        return True
    def gps_to_meters(lat, lon, ref_lat, ref_lon):
        # Simple approximation if gps_utils not available
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        lat_diff = lat - ref_lat
        lon_diff = lon - ref_lon
        x = lon_diff * 111320.0 * np.cos(np.radians(ref_lat))
        y = lat_diff * 111320.0
        return (x, y)


class Camera3DCalibrator:
    """
    Interactive 3D camera calibration tool.
    
    Collects image points and corresponding 3D world points (with GPS),
    then solves for camera intrinsic and extrinsic parameters.
    """
    
    def __init__(self, image_path: str, initial_intrinsics: Optional[np.ndarray] = None):
        """
        Initialize calibrator.
        
        Args:
            image_path: Path to calibration image
            initial_intrinsics: Optional initial intrinsic matrix estimate
        """
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.image_points = []  # List of (u, v) image coordinates
        self.world_points = []  # List of (X, Y, Z) world coordinates in meters
        self.gps_points = []    # List of (lat, lon) GPS coordinates
        self.ref_lat = None
        self.ref_lon = None
        
        # Get image size for initial intrinsics
        h, w = self.image.shape[:2]
        self.image_size = (w, h)
        
        # Initialize intrinsics (default estimate)
        if initial_intrinsics is not None:
            self.K = np.array(initial_intrinsics, dtype=np.float32)
        else:
            # Default: assume focal length is ~image width (common for cameras)
            fx = fy = float(w)  # Reasonable default
            cx = w / 2.0
            cy = h / 2.0
            self.K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        
        self.window_name = "3D Camera Calibration - Click 4+ ground points, press 'q' when done"
    
    def parse_gps_from_input(self, user_input: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse GPS coordinates from various input formats.
        
        Supports:
        - "lat, lon" or "lat,lon"
        - "lat lon" (space separated)
        - Google Maps URL with coordinates
        - Single coordinate value (prompts for second)
        
        Args:
            user_input: User input string
            
        Returns:
            Tuple of (lat, lon) or (None, None) if cannot parse
        """
        user_input = user_input.strip()
        
        # Try to extract from Google Maps URL
        url_pattern = r'[-+]?[0-9]*\.?[0-9]+'
        if 'maps.google' in user_input or 'google.com/maps' in user_input:
            coords = re.findall(url_pattern, user_input)
            if len(coords) >= 2:
                try:
                    lat = float(coords[0])
                    lon = float(coords[1])
                    if validate_gps_coordinate(lat, lon):
                        return (lat, lon)
                except ValueError:
                    pass
        
        # Try comma-separated
        if ',' in user_input:
            parts = user_input.split(',')
            if len(parts) >= 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    if validate_gps_coordinate(lat, lon):
                        return (lat, lon)
                except ValueError:
                    pass
        
        # Try space-separated
        parts = user_input.split()
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                if validate_gps_coordinate(lat, lon):
                    return (lat, lon)
            except ValueError:
                pass
        
        # Try single value
        try:
            lat = float(user_input)
            return (lat, None)
        except ValueError:
            pass
        
        return (None, None)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select image points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\nPoint {len(self.image_points) + 1}: Image ({x}, {y})")
            
            # Get GPS coordinates
            if self.ref_lat is None:
                # First point - ask for GPS
                print("  Enter GPS Latitude (degrees, e.g., 41.91164053): ", end='')
                lat_input = input().strip()
                lat, lon = self.parse_gps_from_input(lat_input)
                
                # If only latitude was parsed, prompt for longitude
                if lat is not None and lon is None:
                    print("  Enter GPS Longitude (degrees, e.g., -89.04468542): ", end='')
                    lon_input = input().strip()
                    # Try to parse as single number (longitude)
                    try:
                        lon = float(lon_input)
                    except ValueError:
                        # Try parsing as coordinate pair
                        parsed_lat, parsed_lon = self.parse_gps_from_input(lon_input)
                        if parsed_lon is not None:
                            lon = parsed_lon
                        elif parsed_lat is not None:
                            # User entered latitude again, use it as longitude (might be mistake, but try)
                            lon = parsed_lat
                
                if lat is not None and lon is not None:
                    if validate_gps_coordinate(lat, lon):
                        self.ref_lat = lat
                        self.ref_lon = lon
                        print(f"  Reference point set: ({lat}, {lon})")
                        print(f"  GPS ({lat:.8f}, {lon:.8f}) -> Local (0.00m, 0.00m)")
                    else:
                        print("  Invalid GPS coordinates! Latitude must be -90 to 90, Longitude -180 to 180")
                        return
                else:
                    print("  Failed to parse GPS coordinates!")
                    if lat is None:
                        print("  Could not parse latitude. Please enter a number.")
                    if lon is None:
                        print("  Could not parse longitude. Please enter a number.")
                    return
            else:
                # Subsequent points
                print(f"  Enter GPS Latitude (degrees, current ref: {self.ref_lat:.8f}): ", end='')
                lat_input = input().strip()
                lat, lon = self.parse_gps_from_input(lat_input)
                
                # If only latitude was parsed, prompt for longitude
                if lat is not None and lon is None:
                    print(f"  Enter GPS Longitude (degrees, current ref: {self.ref_lon:.8f}): ", end='')
                    lon_input = input().strip()
                    # Try to parse as single number (longitude)
                    try:
                        lon = float(lon_input)
                    except ValueError:
                        # Try parsing as coordinate pair
                        parsed_lat, parsed_lon = self.parse_gps_from_input(lon_input)
                        if parsed_lon is not None:
                            lon = parsed_lon
                        elif parsed_lat is not None:
                            lon = parsed_lat
                
                if lat is None or lon is None:
                    print("  Failed to parse GPS coordinates!")
                    if lat is None:
                        print("  Could not parse latitude. Please enter a number.")
                    if lon is None:
                        print("  Could not parse longitude. Please enter a number.")
                    return
                
                if not validate_gps_coordinate(lat, lon):
                    print("  Invalid GPS coordinates! Latitude must be -90 to 90, Longitude -180 to 180")
                    return
            
            # Convert GPS to world coordinates (meters)
            x_meters, y_meters = gps_to_meters(lat, lon, self.ref_lat, self.ref_lon)
            z_meters = 0.0  # Ground points are at Z=0
            
            # Store points
            self.image_points.append((float(x), float(y)))
            self.world_points.append((float(x_meters), float(y_meters), z_meters))
            self.gps_points.append((lat, lon))
            
            print(f"  GPS ({lat:.8f}, {lon:.8f}) -> Local ({x_meters:.2f}m, {y_meters:.2f}m)")
            
            # Draw point on image
            display_img = self.image.copy()
            for i, (px, py) in enumerate(self.image_points):
                cv2.circle(display_img, (int(px), int(py)), 5, (0, 255, 0), -1)
                cv2.putText(display_img, f"{i+1}", (int(px)+10, int(py)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(self.window_name, display_img)
    
    def calibrate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solve for camera pose using PnP algorithm.
        
        Returns:
            (K, R, t, reprojection_error) - Intrinsic matrix, rotation, translation, error
        """
        if len(self.image_points) < 4:
            raise ValueError(f"Need at least 4 points, got {len(self.image_points)}")
        
        # Convert to numpy arrays
        image_pts = np.array(self.image_points, dtype=np.float32)
        world_pts = np.array(self.world_points, dtype=np.float32)
        
        # Assume no distortion for now (can be refined later)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Try to refine intrinsics if we have enough points (6+)
        # Use solvePnP with refinement flags
        use_refine = len(self.image_points) >= 6
        
        if use_refine:
            # Use solvePnP with refinement to get better intrinsics estimate
            # First solve with initial intrinsics
            success, rvec, tvec = cv2.solvePnP(
                world_pts,
                image_pts,
                self.K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Try to refine using solvePnPRefineLM (Levenberg-Marquardt optimization)
                try:
                    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
                        world_pts,
                        image_pts,
                        self.K,
                        dist_coeffs,
                        rvec,
                        tvec
                    )
                    rvec = rvec_refined
                    tvec = tvec_refined
                except:
                    # Refinement failed, use original
                    pass
        
        # Solve PnP (Perspective-n-Point) problem
        # This finds camera pose (R, t) given 3D-2D correspondences
        # Note: solvePnP with refinement flags if available
        flags = cv2.SOLVEPNP_ITERATIVE
        if use_refine and hasattr(cv2, 'SOLVEPNP_ITERATIVE'):
            # Try refinement with more iterations
            flags = cv2.SOLVEPNP_ITERATIVE
        
        success, rvec, tvec = cv2.solvePnP(
            world_pts,  # 3D world points
            image_pts,  # 2D image points
            self.K,     # Camera intrinsic matrix
            dist_coeffs,
            flags=flags
        )
        
        if not success:
            raise RuntimeError("Failed to solve PnP problem")
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Calculate reprojection error
        reprojected_points, _ = cv2.projectPoints(
            world_pts, rvec, tvec, self.K, dist_coeffs
        )
        reprojected_points = reprojected_points.reshape(-1, 2)
        
        errors = np.sqrt(np.sum((image_pts - reprojected_points) ** 2, axis=1))
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        
        # Warn if error is high
        if mean_error > 10.0:
            print(f"\n⚠️  WARNING: High reprojection error ({mean_error:.2f} pixels)")
            print(f"   This suggests the camera intrinsics may need adjustment.")
            print(f"   Suggestions:")
            print(f"   - Try providing better initial intrinsics (--fx, --fy, --cx, --cy)")
            print(f"   - Use more calibration points (6-8 recommended)")
            print(f"   - Verify GPS coordinates are accurate (within 1-2 meters)")
            print(f"   - Ensure all points are truly on the same plane (ground level)")
        
        return (self.K, R, tvec, mean_error)
    
    def run(self):
        """Run interactive calibration."""
        print("=" * 80)
        print("3D Camera Calibration")
        print("=" * 80)
        print()
        print("Instructions:")
        print("1. Click on 4+ ground-level landmark points in the image")
        print("2. Enter corresponding GPS coordinates (latitude, longitude)")
        print("   Tip: Use Google Maps to get GPS coordinates for each point")
        print("3. Press 'q' when done (minimum 4 points required)")
        print()
        print("NOTE: The first GPS point will be used as the reference origin (0,0)")
        print("      All other points will be converted relative to this reference.")
        print()
        print("Best Practices:")
        print("- Use points on the ground plane (same elevation)")
        print("- Spread points across the image (not clustered)")
        print("- Choose points that are clearly visible and permanent")
        print()
        
        # Display image
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.image)
        
        print("Click on points in the image window...")
        print("Press 'q' in the image window when done.\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if len(self.image_points) < 4:
                    print(f"\nError: Need at least 4 points, currently have {len(self.image_points)}")
                    print("Continue clicking points or press 'q' again to exit.")
                    continue
                break
        
        cv2.destroyAllWindows()
        
        if len(self.image_points) < 4:
            raise ValueError(f"Calibration failed: Need at least 4 points, got {len(self.image_points)}")
        
        print(f"\nCalibrating with {len(self.image_points)} points...")
        
        # Solve for camera pose
        K, R, t, reproj_error = self.calibrate()
        
        print(f"\nCalibration complete!")
        print(f"Reprojection error: {reproj_error:.4f} pixels")
        
        # Provide quality assessment
        if reproj_error < 1.0:
            print("✅ Excellent calibration quality")
        elif reproj_error < 5.0:
            print("✅ Good calibration quality")
        elif reproj_error < 10.0:
            print("⚠️  Acceptable calibration quality")
        else:
            print("⚠️  Low calibration quality - consider recalibrating")
            print("   High error may affect GPS coordinate accuracy")
        
        print(f"GPS Reference: ({self.ref_lat:.8f}, {self.ref_lon:.8f})")
        
        return {
            'intrinsic_matrix': K.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': t.flatten().tolist(),
            'distortion_coefficients': [0.0, 0.0, 0.0, 0.0],  # No distortion for now
            'image_points': self.image_points,
            'world_points': self.world_points,
            'gps_points': self.gps_points,
            'image_path': str(self.image_path),
            'image_size': {'width': self.image_size[0], 'height': self.image_size[1]},
            'gps_reference': {
                'lat': self.ref_lat,
                'lon': self.ref_lon,
                'description': 'Reference GPS point corresponding to origin (0,0,0)'
            },
            'reprojection_error': reproj_error
        }


def main():
    parser = argparse.ArgumentParser(
        description='3D Camera Calibration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate with image file
  python tools/calibrate_3d.py --image calibration_image.jpg --save config/calib/camera-01_3d.json
  
  # Calibrate with custom initial intrinsics
  python tools/calibrate_3d.py --image img.jpg --save calib.json --fx 1000 --fy 1000
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to calibration image'
    )
    
    parser.add_argument(
        '--save', '-s',
        type=str,
        required=True,
        help='Path to save calibration JSON file'
    )
    
    parser.add_argument(
        '--fx',
        type=float,
        default=None,
        help='Initial focal length X (pixels). If not provided, estimated from image width.'
    )
    
    parser.add_argument(
        '--fy',
        type=float,
        default=None,
        help='Initial focal length Y (pixels). If not provided, estimated from image width.'
    )
    
    parser.add_argument(
        '--cx',
        type=float,
        default=None,
        help='Initial principal point X (pixels). Default: image width / 2'
    )
    
    parser.add_argument(
        '--cy',
        type=float,
        default=None,
        help='Initial principal point Y (pixels). Default: image height / 2'
    )
    
    args = parser.parse_args()
    
    # Build initial intrinsics if provided
    initial_intrinsics = None
    if args.fx or args.fy or args.cx or args.cy:
        # Load image to get size
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Could not load image: {args.image}")
            return
        h, w = img.shape[:2]
        
        fx = args.fx if args.fx else float(w)
        fy = args.fy if args.fy else float(w)
        cx = args.cx if args.cx else w / 2.0
        cy = args.cy if args.cy else h / 2.0
        
        initial_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    # Run calibration
    try:
        calibrator = Camera3DCalibrator(args.image, initial_intrinsics)
        calib_data = calibrator.run()
        
        # Save calibration
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"\nCalibration saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
