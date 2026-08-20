"""
Homography Calibration Tool

Interactive tool for calibrating camera homography (image -> GPS coordinates).
Click 4+ landmark points on an image and enter corresponding GPS coordinates.

Supports both camera frames and Google Maps satellite images.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import sys
import os
import re

# Add parent directory to path to import gps_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.gps_utils import gps_to_meters, validate_gps_coordinate


class HomographyCalibrator:
    """
    Interactive homography calibration tool.
    """
    
    def __init__(self, image_path: str, use_gps: bool = True, is_satellite: bool = False):
        """
        Initialize calibrator.
        
        Args:
            image_path: Path to calibration image
            use_gps: If True, accept GPS coordinates (lat/lon). If False, use local meters.
            is_satellite: If True, image is from Google Maps satellite view (enables GPS URL parsing)
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.use_gps = use_gps
        self.is_satellite = is_satellite
        
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.image_points = []
        self.world_points = []  # Will store GPS coords if use_gps=True, else meters
        self.gps_points = []  # Always store GPS coords for reference
        self.ref_lat = None
        self.ref_lon = None
        
        if is_satellite:
            self.window_name = "Homography Calibration (Google Maps Satellite) - Click 4+ points, press 'q' when done"
        else:
        self.window_name = "Homography Calibration - Click 4+ points, press 'q' when done"
    
    def parse_gps_from_input(self, user_input: str) -> tuple:
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
            Tuple of (lat, lon) or (lat, None) if only one value, or None if cannot parse
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
                    return (lat, lon)
                except ValueError:
                    pass
        
        # Try space-separated
        parts = user_input.split()
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                return (lat, lon)
            except ValueError:
                pass
        
        # Single value - return as latitude, will prompt for longitude
        try:
            lat = float(user_input)
            return (lat, None)
        except ValueError:
            pass
        
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for clicking points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_points.append([x, y])
            print(f"Point {len(self.image_points)}: Image ({x}, {y})")
            
            # Prompt for coordinates
            try:
                if self.use_gps:
                    if self.is_satellite:
                        print("  Enter GPS coordinates:")
                        print("    - Format: 'lat, lon' (e.g., '41.91164053, -89.04468542')")
                        print("    - Or paste Google Maps URL with coordinates")
                        print("    - Or enter latitude, then longitude separately")
                        user_input = input(f"  GPS (lat, lon or URL): ").strip()
                        
                        parsed = self.parse_gps_from_input(user_input)
                        if parsed and parsed[1] is not None:
                            lat, lon = parsed
                        elif parsed and parsed[1] is None:
                            # Got latitude, prompt for longitude
                            lat = parsed[0]
                            lon = float(input(f"  Enter GPS Longitude (degrees): "))
                        else:
                            # Try as single latitude value
                            lat = float(user_input)
                            lon = float(input(f"  Enter GPS Longitude (degrees): "))
                    else:
                    lat = float(input(f"  Enter GPS Latitude (degrees, e.g., 40.7128): "))
                    lon = float(input(f"  Enter GPS Longitude (degrees, e.g., -74.0060): "))
                    
                    if not validate_gps_coordinate(lat, lon):
                        print("  Invalid GPS coordinates! Latitude must be -90 to 90, Longitude -180 to 180")
                        self.image_points.pop()
                        return
                    
                    # Store GPS coordinates
                    self.gps_points.append([lat, lon])
                    
                    # Set reference point from first GPS coordinate
                    if self.ref_lat is None:
                        self.ref_lat = lat
                        self.ref_lon = lon
                        print(f"  Reference point set: ({lat}, {lon})")
                    
                    # Convert GPS to local meters for homography calculation
                    x_meters, y_meters = gps_to_meters(lat, lon, self.ref_lat, self.ref_lon)
                    self.world_points.append([x_meters, y_meters])
                    print(f"  GPS ({lat:.6f}, {lon:.6f}) -> Local ({x_meters:.2f}m, {y_meters:.2f}m)")
                else:
                    world_x = float(input(f"  Enter world X (meters): "))
                    world_y = float(input(f"  Enter world Y (meters): "))
                    self.world_points.append([world_x, world_y])
                    self.gps_points.append([None, None])  # No GPS data
                    print(f"  World ({world_x}, {world_y})")
            except ValueError:
                print("  Invalid input, skipping point")
                self.image_points.pop()
                return
            
            # Draw point on image
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                self.image,
                f"{len(self.image_points)}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow(self.window_name, self.image)
    
    def calibrate(self) -> tuple:
        """
        Run interactive calibration.
        
        Returns:
            Tuple of (homography_matrix, rmse)
        """
        print("\n=== Homography Calibration ===")
        if self.is_satellite:
            print("Google Maps Satellite Image Mode")
            print("=" * 50)
        if self.use_gps:
            print("Instructions:")
            print("1. Click 4+ landmark points on the image")
            print("2. Enter corresponding GPS coordinates (latitude, longitude)")
            if self.is_satellite:
                print("   - Right-click on Google Maps to get coordinates")
                print("   - Or paste Google Maps URL")
                print("   - Format: 'lat, lon' or separate prompts")
            else:
            print("   Example: Latitude: 40.7128, Longitude: -74.0060")
            print("   Tip: Use Google Maps to get GPS coordinates for each point")
            print("3. Press 'q' when done (minimum 4 points required)")
            print()
            print("NOTE: The first GPS point will be used as the reference origin (0,0)")
            print("      All other points will be converted relative to this reference.")
            print()
        else:
            print("Instructions:")
            print("1. Click 4+ landmark points on the image")
            print("2. Enter corresponding world coordinates (in meters)")
            print("3. Press 'q' when done (minimum 4 points required)")
            print()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display image
        cv2.imshow(self.window_name, self.image)
        
        # Wait for 'q' key
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Check minimum points
        if len(self.image_points) < 4:
            raise ValueError("Need at least 4 points for homography calibration")
        
        # Convert to numpy arrays
        img_pts = np.array(self.image_points, dtype=np.float32)
        world_pts = np.array(self.world_points, dtype=np.float32)
        
        # Compute homography
        H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            raise ValueError("Failed to compute homography")
        
        # Compute RMSE
        projected = cv2.perspectiveTransform(
            img_pts.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2)
        
        errors = np.sqrt(np.sum((projected - world_pts) ** 2, axis=1))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        return H, rmse
    
    def save(self, output_path: str, H: np.ndarray, rmse: float):
        """
        Save calibration to JSON file.
        
        Args:
            output_path: Output JSON file path
            H: Homography matrix (3x3)
            rmse: Root mean square error (meters)
        """
        # Get image dimensions
        img_height, img_width = self.image.shape[:2]
        
        calib_data = {
            'H': H.tolist(),
            'rmse': float(rmse),
            'image_points': self.image_points,
            'world_points': self.world_points,
            'image_path': str(self.image_path),
            'image_size': {
                'width': int(img_width),
                'height': int(img_height)
            },
            'use_gps': self.use_gps,
            'is_satellite_image': self.is_satellite
        }
        
        # Add GPS reference if using GPS
        if self.use_gps and self.ref_lat is not None and self.ref_lon is not None:
            calib_data['gps_reference'] = {
                'lat': float(self.ref_lat),
                'lon': float(self.ref_lon),
                'description': 'Reference GPS point corresponding to homography origin (0,0)'
            }
            calib_data['gps_points'] = self.gps_points
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"\nCalibration saved to: {output_path}")
        print(f"RMSE: {rmse:.3f} meters")
        if self.use_gps and self.ref_lat is not None:
            print(f"GPS Reference: ({self.ref_lat:.6f}, {self.ref_lon:.6f})")
        if rmse > 1.5:
            print("Warning: RMSE > 1.5m, consider recalibrating with more points")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Homography Calibration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate from camera frame:
  python tools/calibrate_h.py --image frame.jpg --save config/calib/camera-01_h.json
  
  # Calibrate from Google Maps satellite image:
  python tools/calibrate_h.py --image satellite_map.png --save config/calib/camera-01_h.json --satellite
  
  # Using local meters (legacy mode):
  python tools/calibrate_h.py --image frame.jpg --save config/calib/camera-01_h.json --use-meters
        """
    )
    parser.add_argument('--image', required=True, help='Path to calibration image')
    parser.add_argument('--save', required=True, help='Output JSON file path (e.g., config/calib/camera-id_h.json)')
    parser.add_argument('--use-meters', action='store_true', help='Use local meters instead of GPS coordinates (legacy mode)')
    parser.add_argument('--satellite', action='store_true', help='Image is from Google Maps satellite view (enables GPS URL parsing)')
    
    args = parser.parse_args()
    
    try:
        use_gps = not args.use_meters  # Default to GPS mode
        calibrator = HomographyCalibrator(
            args.image, 
            use_gps=use_gps,
            is_satellite=args.satellite
        )
        H, rmse = calibrator.calibrate()
        calibrator.save(args.save, H, rmse)
        
        print("\nCalibration complete!")
        if use_gps:
            print("\nGPS coordinates will be output for all detections.")
            print("You can copy-paste these coordinates directly into Google Maps!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())


