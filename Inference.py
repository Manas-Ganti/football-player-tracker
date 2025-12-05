from team_classify import *
from live_stats import *
from speed_track import * 

import cv2
import numpy as np
import torch
import supervision as sv
from typing import List, Dict, Generator, Iterable, TypeVar
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import umap
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
import time



class SoccerInference:
    """Main inference pipeline for soccer player detection, tracking, and analysis"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 player_class_id: int = 0, ball_class_id: int = 32):
        """
        Initialize the soccer inference system
        
        Args:
            model_path (str): Path to YOLOv8 model
            confidence_threshold (float): Detection confidence threshold
            player_class_id (int): Class ID for players in YOLO model
            ball_class_id (int): Class ID for ball in YOLO model
        """
        self.confidence_threshold = confidence_threshold
        self.player_class_id = player_class_id
        self.ball_class_id = ball_class_id
        
        # Detect device
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("✓ Using MPS (Apple Silicon GPU) acceleration")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("✓ Using CUDA GPU acceleration")
        else:
            self.device = 'cpu'
            print("✗ Using CPU")
        
        # Load YOLOv8 detection model
        print(f"Loading detection model: {model_path}")
        self.detection_model = YOLO(model_path)
        self.detection_model.conf = confidence_threshold
        
        # Initialize DeepSort tracker
        print("Initializing DeepSort tracker...")
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_iou_distance=0.7,
            embedder="mobilenet",
            half=True
        )
        
        # Initialize team classifier (will be fitted on first N frames)
        print("Initializing SigLIP team classifier...")
        self.team_classifier = TeamClassifier(device=self.device, batch_size=16)
        
        # Speed tracker
        self.speed_tracker = None
        
        # Stats display
        self.stats_display = None
        
        # Team assignment storage
        self.team_assignments = {}
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        
    def collect_player_crops(self, video_path: str, num_frames: int = 50,
                            frame_stride: int = 5) -> List[np.ndarray]:
        """
        Collect player crops from video for team classifier training
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to sample
            frame_stride (int): Sample every Nth frame
            
        Returns:
            List[np.ndarray]: List of player crop images
        """
        print(f"\nCollecting player crops from {num_frames} frames for team classification...")
        
        cap = cv2.VideoCapture(video_path)
        crops = []
        frame_count = 0
        frames_processed = 0
        
        while frames_processed < num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames based on stride
            if frame_count % frame_stride != 0:
                continue
            
            # Detect players
            results = self.detection_model.predict(frame, device=self.device, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only collect player crops
                    if cls == self.player_class_id and conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Extract upper body (jersey area)
                        height = y2 - y1
                        jersey_height = int(height * 0.6)
                        crop = frame[y1:y1 + jersey_height, x1:x2]
                        
                        if crop.size > 0:
                            crops.append(crop)
            
            frames_processed += 1
            
            if frames_processed % 10 == 0:
                print(f"Processed {frames_processed}/{num_frames} frames, collected {len(crops)} crops")
        
        cap.release()
        print(f"✓ Collected {len(crops)} player crops")
        return crops
    
    def process_video(self, video_path: str, output_path: str = None,
                     display: bool = True, fit_team_classifier: bool = True):
        """
        Process video with detection, tracking, team classification, and speed tracking
        
        Args:
            video_path (str): Input video path
            output_path (str): Output video path (optional)
            display (bool): Whether to display video during processing
            fit_team_classifier (bool): Whether to fit team classifier on video
        """
        # Step 1: Fit team classifier on collected crops
        if fit_team_classifier:
            crops = self.collect_player_crops(video_path, num_frames=50, frame_stride=5)
            if len(crops) > 0:
                print("\nTraining team classifier...")
                self.team_classifier.fit(crops)
                print("✓ Team classifier trained")
            else:
                print("⚠ Warning: No player crops collected, team classification will fail")
                return
        
        # Step 2: Process full video
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize trackers
        self.speed_tracker = SpeedTracker(fps)
        self.speed_tracker.calibrate_scale(width, height)
        self.stats_display = LiveStatsDisplay(width=400, height=height)
        
        # Setup video writer
        writer = None
        if output_path:
            combined_width = width + 400
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))
        
        print(f"\n{'='*60}")
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print(f"{'='*60}")
        print("Press 'q' to quit, 's' to save screenshot\n")
        
        frame_count = 0
        
        while cap.isOpened():
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1. Detect all bounding boxes
            detections = self.detect_players(frame)
            
            # 2. Track players with DeepSort
            tracked_players = self.track_players(frame, detections)
            
            # 3. Classify teams using SigLIP
            team_assignments = self.classify_teams(frame, tracked_players)
            
            # 4. Calculate speeds
            speeds = self.calculate_speeds(tracked_players)
            
            # Visualize
            annotated_frame = self.visualize_frame(
                frame, tracked_players, team_assignments, speeds
            )
            
            # Calculate processing FPS
            frame_time = time.time() - frame_start_time
            processing_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(processing_fps)
            avg_fps = np.mean(self.fps_history)
            
            # Create stats panel
            stats_panel = self.stats_display.create_stats_panel(
                team_assignments, speeds, self.speed_tracker,
                frame_count, fps, avg_fps
            )
            
            # Combine frame and stats
            combined_frame = np.hstack([annotated_frame, stats_panel])
            
            if writer:
                writer.write(combined_frame)
            
            if display:
                display_frame = combined_frame
                if combined_frame.shape[1] > 1920:
                    scale = 1920 / combined_frame.shape[1]
                    new_width = int(combined_frame.shape[1] * scale)
                    new_height = int(combined_frame.shape[0] * scale)
                    display_frame = cv2.resize(combined_frame, (new_width, new_height))
                
                cv2.imshow('Soccer Tracking - Live Stats', display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✗ Quit requested by user")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_frame_{frame_count}.png"
                    cv2.imwrite(screenshot_path, combined_frame)
                    print(f"✓ Screenshot saved: {screenshot_path}")
            
            if frame_count % 30 == 0:
                progress = frame_count * 100 // total_frames
                print(f"Progress: {frame_count}/{total_frames} ({progress}%) | "
                      f"Processing: {avg_fps:.1f} FPS")
        
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✓ Processing complete!")
        print("="*60)
        
        return self.generate_stats()
    
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """Detect all players and ball in frame"""
        results = self.detection_model.predict(frame, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf >= self.confidence_threshold:
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class': cls
                    })
        
        return detections
    
    def track_players(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Track players using DeepSort"""
        # Convert to DeepSort format
        deepsort_detections = []
        for det in detections:
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            deepsort_detections.append((
                [bbox[0], bbox[1], width, height],
                det['confidence'],
                det['class']
            ))
        
        # Update tracker
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        # Convert to output format
        tracked_players = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            ltrb = track.to_ltrb()
            tracked_players.append({
                'track_id': track.track_id,
                'bbox': ltrb,
                'class': track.get_det_class()
            })
        
        return tracked_players
    
    def classify_teams(self, frame: np.ndarray, tracked_players: List[Dict]) -> Dict[int, int]:
        """Classify players into teams using SigLIP"""
        if not self.team_classifier.is_fitted:
            return {}
        
        # Collect crops for players only
        crops = []
        track_ids = []
        
        for player in tracked_players:
            if player['class'] != self.player_class_id:
                continue
            
            track_id = player['track_id']
            bbox = player['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract jersey region
            height = y2 - y1
            jersey_height = int(height * 0.6)
            crop = frame[y1:y1 + jersey_height, x1:x2]
            
            if crop.size > 0:
                crops.append(crop)
                track_ids.append(track_id)
        
        # Predict teams
        if len(crops) > 0:
            team_labels = self.team_classifier.predict(crops)
            
            for track_id, team_label in zip(track_ids, team_labels):
                self.team_assignments[track_id] = int(team_label)
        
        return self.team_assignments
    
    def calculate_speeds(self, tracked_players: List[Dict]) -> Dict[int, float]:
        """Calculate speeds for all players"""
        speeds = {}
        
        for player in tracked_players:
            if player['class'] == self.ball_class_id:
                continue
            
            track_id = player['track_id']
            bbox = player['bbox']
            speed = self.speed_tracker.update_position(track_id, bbox)
            speeds[track_id] = speed
        
        return speeds
    
    def visualize_frame(self, frame: np.ndarray, tracked_players: List[Dict],
                       team_assignments: Dict[int, int], 
                       speeds: Dict[int, float]) -> np.ndarray:
        """Draw annotations on frame"""
        team_colors = {
            0: (255, 100, 100),
            1: (100, 100, 255),
        }
        
        for player in tracked_players:
            track_id = player['track_id']
            bbox = player['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Determine color and label
            if player['class'] == self.ball_class_id:
                color = (0, 255, 0)
                label = "BALL"
            else:
                team = team_assignments.get(track_id, 0)
                color = team_colors.get(team, (255, 255, 255))
                speed = speeds.get(track_id, 0)
                label = f"ID:{track_id} T{team} {speed:.1f}km/h"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return frame
    
    def generate_stats(self) -> Dict:
        """Generate final statistics"""
        stats = {
            'total_players_tracked': len(self.speed_tracker.speeds),
            'max_speeds': dict(self.speed_tracker.max_speeds),
            'total_distances': dict(self.speed_tracker.total_distance),
            'team_distribution': dict(self.team_assignments),
            'avg_processing_fps': float(np.mean(self.fps_history)) if self.fps_history else 0
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    
    # Configuration
    MODEL_PATH = "models/cv.pt"  # Path to your trained model
    VIDEO_PATH = "videos/0bfacc_0.mp4"  # Input video
    OUTPUT_PATH = "videos/output_annotated.mp4"  # Output video
    print("HERE")
    # Initialize inference pipeline
    inference = SoccerInference(
        model_path=MODEL_PATH,
        confidence_threshold=0.5,
        player_class_id=0,  # Adjust based on your model
        ball_class_id=32    # Adjust based on your model
    )
    
    # Process video
    stats = inference.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        display=True,
        fit_team_classifier=True
    )
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total players tracked: {stats['total_players_tracked']}")
    print(f"Average processing FPS: {stats['avg_processing_fps']:.1f}")
    
    print("\nTeam Distribution:")
    team_0_count = sum(1 for t in stats['team_distribution'].values() if t == 0)
    team_1_count = sum(1 for t in stats['team_distribution'].values() if t == 1)
    print(f"  Team 0: {team_0_count} players")
    print(f"  Team 1: {team_1_count} players")
    
    print("\nTop 5 Max Speeds:")
    sorted_speeds = sorted(stats['max_speeds'].items(), key=lambda x: x[1], reverse=True)
    for i, (track_id, speed) in enumerate(sorted_speeds[:5], 1):
        team = stats['team_distribution'].get(track_id, 'Unknown')
        distance = stats['total_distances'].get(track_id, 0)
        print(f"  {i}. Player {track_id} (Team {team}): {speed:.2f} km/h | {distance:.0f}m covered")
    
    print("\nTop 5 Distance Covered:")
    sorted_distances = sorted(stats['total_distances'].items(), key=lambda x: x[1], reverse=True)
    for i, (track_id, distance) in enumerate(sorted_distances[:5], 1):
        team = stats['team_distribution'].get(track_id, 'Unknown')
        max_speed = stats['max_speeds'].get(track_id, 0)
        print(f"  {i}. Player {track_id} (Team {team}): {distance:.0f}m | Max: {max_speed:.2f} km/h")
    
    print("\n" + "="*60)