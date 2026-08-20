"""
ByteTrack Multi-Object Tracker Wrapper

This module provides a wrapper for ByteTrack tracking with Kalman filtering.
"""

from typing import List, Dict, Tuple
import numpy as np


class KalmanFilter:
    """Simple Kalman filter for bounding box tracking."""
    
    def __init__(self):
        # State: [cx, cy, s, r, vx, vy, vs, vr] where (cx,cy) is center, s is area, r is aspect ratio
        self.state = np.zeros(8, dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
        
        # Motion model (constant velocity)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # s += vs
        self.F[3, 7] = 1.0  # r += vr
        
        # Measurement model (observe center, area, aspect ratio)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0  # observe cx
        self.H[1, 1] = 1.0  # observe cy
        self.H[2, 2] = 1.0  # observe s
        self.H[3, 3] = 1.0  # observe r
        
        # Process noise - increased to allow for more movement prediction
        # This helps when trailers move across the frame
        self.Q = np.eye(8, dtype=np.float32) * 0.2  # Increased from 0.1 to allow more movement
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1.0
    
    def init(self, bbox: List[float]):
        """Initialize filter with bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)
        
        self.state = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
    
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Validate state after prediction - clamp to reasonable values
        self.state[2] = max(self.state[2], 1.0)  # Area must be positive
        self.state[3] = max(min(self.state[3], 100.0), 0.01)  # Aspect ratio in reasonable range
        # Ensure state values are finite
        self.state = np.where(np.isfinite(self.state), self.state, 0.0)
    
    def update(self, bbox: List[float]):
        """Update with measurement [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        
        # Validate input bbox
        if not all(np.isfinite(v) for v in bbox):
            return  # Skip update if bbox contains NaN/Inf
        
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)
        
        # Validate computed values
        if not all(np.isfinite([cx, cy, s, r])) or s <= 0 or r <= 0:
            return  # Skip update if computed values are invalid
        
        z = np.array([cx, cy, s, r], dtype=np.float32)
        
        # Kalman update
        try:
            y = z - self.H @ self.state  # Innovation
            S = self.H @ self.covariance @ self.H.T + self.R  # Innovation covariance
            K = self.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
            
            self.state = self.state + K @ y
            self.covariance = (np.eye(8) - K @ self.H) @ self.covariance
            
            # Validate state after update - clamp to reasonable values
            self.state[2] = max(self.state[2], 1.0)  # Area must be positive
            self.state[3] = max(min(self.state[3], 100.0), 0.01)  # Aspect ratio in reasonable range
            # Ensure state values are finite
            self.state = np.where(np.isfinite(self.state), self.state, 0.0)
        except (np.linalg.LinAlgError, ValueError):
            # If Kalman update fails, skip it
            pass
    
    def get_bbox(self) -> List[float]:
        """Get predicted bounding box [x1, y1, x2, y2]."""
        cx, cy, s, r = self.state[0], self.state[1], self.state[2], self.state[3]
        
        # Validate and clamp state values to prevent NaN
        # Ensure area is positive and aspect ratio is reasonable
        s = max(s, 1.0)  # Minimum area of 1 pixel
        r = max(r, 0.01)  # Minimum aspect ratio (avoid division by very small values)
        r = min(r, 100.0)  # Maximum aspect ratio (prevent extreme values)
        
        # Check for NaN or Inf in center coordinates
        if not np.isfinite(cx) or not np.isfinite(cy):
            # Return a default bbox if center is invalid
            return [0.0, 0.0, 100.0, 100.0]
        
        # Compute height and width with validation
        try:
            h = np.sqrt(s / (r + 1e-6))
            w = s / (h + 1e-6)
            
            # Validate computed dimensions
            if not np.isfinite(h) or not np.isfinite(w) or h <= 0 or w <= 0:
                # Fallback: use reasonable defaults based on area
                h = np.sqrt(s)
                w = h  # Square fallback
        except (ValueError, ZeroDivisionError):
            # Fallback if computation fails
            h = np.sqrt(s)
            w = h
        
        # Ensure dimensions are finite and positive
        h = max(h, 1.0) if np.isfinite(h) else 100.0
        w = max(w, 1.0) if np.isfinite(w) else 100.0
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Final validation: ensure all values are finite
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        if any(not np.isfinite(v) for v in bbox):
            # Return a default bbox if any value is invalid
            return [0.0, 0.0, 100.0, 100.0]
        
        return bbox


class ByteTrackWrapper:
    """
    ByteTrack multi-object tracker with Kalman filtering.
    
    Maintains track identities across frames using IoU matching and Kalman filtering.
    """
    
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.20, track_buffer: int = 30, match_thresh: float = 0.8):
        """
        Initialize ByteTrack tracker.
        
        Args:
            frame_rate: Video frame rate
            track_thresh: Detection confidence threshold for tracking (lowered to 0.20 for better detection)
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
        """
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.next_track_id = 1
        self.tracks = {}  # track_id -> {bbox, conf, cls, kf, frame_count, state}
        self.lost_tracks = {}  # Lost tracks that might be recovered
        self.frame_count = 0
    
    def _bbox_to_xyxy(self, bbox: List[float]) -> List[float]:
        """Ensure bbox is in [x1, y1, x2, y2] format."""
        if len(bbox) == 4:
            return bbox
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    
    def update(self, detections: List[Dict], frame: np.ndarray = None) -> List[Dict]:
        """
        Update tracker with new detections (ByteTrack algorithm).
        
        Args:
            detections: List of detection dicts with keys: bbox, cls, conf
            frame: Optional frame for visualization
            
        Returns:
            List of track dicts with keys: track_id, bbox, cls, conf
        """
        self.frame_count += 1
        
        # Predict all active tracks
        # For lost tracks, predict multiple steps to account for detection gaps
        # (e.g., if detection runs every 5 frames, predict 5 steps ahead)
        for track_id, track in list(self.tracks.items()):
            if track['state'] == 'tracked':
                track['kf'].predict()
                track['bbox'] = track['kf'].get_bbox()
            elif track['state'] == 'lost':
                # For lost tracks, predict multiple steps to catch up with detection gaps
                # This helps when detection runs every N frames
                frames_lost = self.frame_count - track['frame_count']
                # Predict ahead by the number of frames lost (up to 10 steps for better prediction)
                # This accounts for detection gaps (detect_every_n=5) and movement
                for _ in range(min(frames_lost, 10)):  # Predict up to 10 steps ahead
                    track['kf'].predict()
                track['bbox'] = track['kf'].get_bbox()
        
        # Separate detections into high and low confidence
        high_conf_dets = [d for d in detections if d['conf'] >= self.track_thresh]
        low_conf_dets = [d for d in detections if d['conf'] < self.track_thresh]
        
        # First association: match high confidence detections with tracked objects
        tracked_tracks = {tid: t for tid, t in self.tracks.items() if t['state'] == 'tracked'}
        matched_pairs, unmatched_tracks, unmatched_dets = self._associate_detections_to_tracks(
            high_conf_dets, tracked_tracks
        )
        
        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            det = high_conf_dets[det_idx]
            track = self.tracks[track_id]
            track['kf'].update(det['bbox'])
            track['bbox'] = det['bbox']  # Use detection bbox
            track['conf'] = det['conf']
            track['cls'] = det['cls']
            track['frame_count'] = self.frame_count
            track['state'] = 'tracked'
        
        # Second association: match remaining detections with lost tracks
        # Use more lenient matching for lost tracks (they may have moved significantly)
        lost_tracks = {tid: t for tid, t in self.tracks.items() if t['state'] == 'lost'}
        
        # For lost tracks, use VERY lenient matching to maximize reconnection
        # This is critical for maintaining track identity when trailers move significantly
        original_match_thresh = self.match_thresh
        self.match_thresh = 0.15  # Very low threshold for lost tracks (allows reconnection even with low IoU)
        
        matched_pairs_lost, unmatched_lost, unmatched_dets_rem = self._associate_detections_to_tracks(
            [high_conf_dets[i] for i in unmatched_dets], lost_tracks
        )
        
        # Restore original threshold
        self.match_thresh = original_match_thresh
        
        # Reactivate lost tracks
        for det_idx_rem, track_id in matched_pairs_lost:
            det = [high_conf_dets[i] for i in unmatched_dets][det_idx_rem]
            track = self.tracks[track_id]
            track['kf'].update(det['bbox'])
            track['bbox'] = det['bbox']
            track['conf'] = det['conf']
            track['cls'] = det['cls']
            track['frame_count'] = self.frame_count
            track['state'] = 'tracked'
        
        # Create new tracks for unmatched high confidence detections
        final_unmatched_dets = [unmatched_dets[i] for i in range(len(unmatched_dets)) 
                               if i not in [p[0] for p in matched_pairs_lost]]
        for det_idx in final_unmatched_dets:
            det = high_conf_dets[det_idx]
            track_id = self.next_track_id
            self.next_track_id += 1
            
            kf = KalmanFilter()
            kf.init(det['bbox'])
            
            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'conf': det['conf'],
                'cls': det['cls'],
                'kf': kf,
                'frame_count': self.frame_count,
                'state': 'tracked'
            }
        
        # Mark unmatched tracks as lost
        for track_id in unmatched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id]['state'] = 'lost'
        
        # Remove tracks that have been lost for too long
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track['state'] == 'lost' and (self.frame_count - track['frame_count']) > self.track_buffer:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return active tracks (both tracked and lost tracks within buffer)
        # This ensures detections remain stable even when detector misses frames
        active_tracks = []
        for track_id, track in self.tracks.items():
            # Include both 'tracked' and 'lost' tracks (lost tracks are still within buffer)
            # Lost tracks use predicted positions from Kalman filter
            if track['state'] in ['tracked', 'lost']:
                active_tracks.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],  # Uses predicted position for lost tracks
                    'cls': track['cls'],
                    'conf': track['conf'],
                    'state': track['state']  # Include state for visualization
                })
        
        return active_tracks
    
    def _compute_center_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute distance between centers of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        cx1 = (x1_1 + x2_1) / 2.0
        cy1 = (y1_1 + y2_1) / 2.0
        cx2 = (x1_2 + x2_2) / 2.0
        cy2 = (y1_2 + y2_2) / 2.0
        
        dx = cx2 - cx1
        dy = cy2 - cy1
        distance = np.sqrt(dx * dx + dy * dy)
        
        return distance
    
    def _compute_bbox_size(self, bbox: List[float]) -> float:
        """Compute average size (diagonal) of bounding box."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return np.sqrt(w * w + h * h)
    
    def _associate_detections_to_tracks(self, detections: List[Dict], tracks: Dict) -> Tuple[List, List, List]:
        """
        Associate detections to tracks using IoU matching with position-based fallback.
        
        Enhanced matching strategy:
        1. Primary: IoU matching (works when trailer is in similar position)
        2. Fallback: Position-based matching (works when trailer moved significantly)
        
        Returns:
            (matched_pairs, unmatched_tracks, unmatched_dets)
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(tracks.keys()), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
        distance_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
        track_ids = list(tracks.keys())
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = tracks[track_id]
                iou_matrix[i, j] = self._compute_iou(det['bbox'], track['bbox'])
                # Compute center distance for fallback matching
                distance_matrix[i, j] = self._compute_center_distance(det['bbox'], track['bbox'])
        
        # Greedy matching with two strategies
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(tracks.keys())
        
        # Strategy 1: IoU-based matching (primary)
        matches_iou = []
        for i in range(len(detections)):
            for j in range(len(tracks)):
                if iou_matrix[i, j] > self.match_thresh:
                    matches_iou.append((iou_matrix[i, j], i, j, 'iou'))
        
        matches_iou.sort(reverse=True, key=lambda x: x[0])
        
        used_dets = set()
        used_tracks = set()
        
        # Match using IoU first
        for _, i, j, _ in matches_iou:
            if i not in used_dets and j not in used_tracks:
                matched_pairs.append((i, track_ids[j]))
                used_dets.add(i)
                used_tracks.add(j)
        
        # Strategy 2: Position-based fallback for unmatched detections/tracks
        # This helps when trailer moved significantly (low IoU but reasonable distance)
        # Only use for lost tracks or when IoU is very low
        remaining_dets = [i for i in range(len(detections)) if i not in used_dets]
        remaining_track_indices = [j for j in range(len(tracks)) if track_ids[j] not in used_tracks]
        
        if remaining_dets and remaining_track_indices:
            # For lost tracks, use more lenient position-based matching
            # Calculate relative distance threshold based on bbox size
            matches_position = []
            for i in remaining_dets:
                det = detections[i]
                det_size = self._compute_bbox_size(det['bbox'])
                
                for j in remaining_track_indices:
                    track_id = track_ids[j]
                    track = tracks[track_id]
                    distance = distance_matrix[i, j]
                    
                    # Use relative distance threshold: within 2x the bbox size
                    # This allows matching even when trailer moved significantly
                    max_distance = det_size * 2.0
                    
                    # Also check IoU - if it's above a lower threshold, consider it
                    iou = iou_matrix[i, j]
                    
                    # Match if:
                    # 1. Distance is reasonable (within 2x bbox size) AND
                    # 2. IoU is above a lower threshold (very lenient for lost tracks)
                    # Note: When matching lost tracks, all tracks in the dict should be lost,
                    # but we check state for safety and to handle edge cases
                    is_lost_track = track.get('state', 'tracked') == 'lost'
                    
                    # For lost tracks, use VERY lenient matching to allow reconnection after movement
                    if is_lost_track:
                        # Extremely lenient: within 3x bbox size (allows significant movement) and IoU >= 0.02
                        # This ensures trailers that move across the frame can be reconnected
                        max_distance_lost = det_size * 3.0  # Increased from 1.5x to 3x for very lenient matching
                        if distance <= max_distance_lost and iou >= 0.02:  # Very low IoU threshold
                            # Score: prefer higher IoU and lower distance, but prioritize distance for lost tracks
                            score = iou * 0.4 + (1.0 - min(distance / max_distance_lost, 1.0)) * 0.6
                            matches_position.append((score, i, j, 'position'))
                    else:
                        # For active tracks, use standard matching
                        if distance <= max_distance and iou >= 0.20:
                            # Score: prefer higher IoU and lower distance
                            score = iou * 0.7 + (1.0 - min(distance / max_distance, 1.0)) * 0.3
                            matches_position.append((score, i, j, 'position'))
            
            matches_position.sort(reverse=True, key=lambda x: x[0])
            
            # Match using position-based fallback
            for _, i, j, _ in matches_position:
                track_id = track_ids[j]
                if i not in used_dets and track_id not in used_tracks:
                    matched_pairs.append((i, track_id))
                    used_dets.add(i)
                    used_tracks.add(track_id)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [track_ids[j] for j in range(len(tracks)) if track_ids[j] not in used_tracks]
        
        return matched_pairs, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

