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
from speed_track import *


class LiveStatsDisplay:
    """Creates a live stats panel for display"""
    
    def __init__(self, width: int = 400, height: int = 800):
        self.width = width
        self.height = height
        self.team_colors = {
            0: (255, 100, 100),  # Light blue
            1: (100, 100, 255),  # Light red
        }
    
    def create_stats_panel(self, team_assignments: Dict[int, int], 
                          speeds: Dict[int, float], 
                          speed_tracker: SpeedTracker,
                          frame_count: int, fps: float, 
                          processing_fps: float) -> np.ndarray:
        """Create a visual stats panel"""
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark background
        
        y_offset = 20
        
        # Title
        cv2.putText(panel, "LIVE STATS", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y_offset += 40
        
        # Performance metrics
        cv2.putText(panel, f"Processing: {processing_fps:.1f} FPS", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        y_offset += 25
        
        cv2.putText(panel, f"Frame: {frame_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        y_offset += 35
        
        # Separator
        cv2.line(panel, (20, y_offset), (self.width - 20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        # Team statistics
        team_0_count = sum(1 for t in team_assignments.values() if t == 0)
        team_1_count = sum(1 for t in team_assignments.values() if t == 1)
        
        cv2.putText(panel, "TEAMS", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        # Team 0
        cv2.rectangle(panel, (20, y_offset - 15), (40, y_offset + 5), 
                     self.team_colors[0], -1)
        cv2.putText(panel, f"Team 0: {team_0_count} players", (50, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 30
        
        # Team 1
        cv2.rectangle(panel, (20, y_offset - 15), (40, y_offset + 5), 
                     self.team_colors[1], -1)
        cv2.putText(panel, f"Team 1: {team_1_count} players", (50, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 35
        
        # Separator
        cv2.line(panel, (20, y_offset), (self.width - 20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        # Top speeds
        cv2.putText(panel, "TOP SPEEDS", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        # Sort players by current speed
        speed_items = [(tid, spd) for tid, spd in speeds.items() if spd > 0.5]
        speed_items.sort(key=lambda x: x[1], reverse=True)
        
        for i, (track_id, speed) in enumerate(speed_items[:8]):
            team = team_assignments.get(track_id, 0)
            color = self.team_colors[team]
            
            cv2.rectangle(panel, (20, y_offset - 15), (35, y_offset + 5), color, -1)
            
            cv2.putText(panel, f"#{track_id}", (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(panel, f"{speed:.1f} km/h", (100, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Speed bar
            bar_width = int((speed / 35.0) * 150)
            cv2.rectangle(panel, (220, y_offset - 10), (220 + bar_width, y_offset), 
                         color, -1)
            
            y_offset += 25
        
        y_offset += 15
        
        # Separator
        cv2.line(panel, (20, y_offset), (self.width - 20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        # Maximum speeds achieved
        cv2.putText(panel, "MAX SPEEDS", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        max_speed_items = [(tid, spd) for tid, spd in speed_tracker.max_speeds.items()]
        max_speed_items.sort(key=lambda x: x[1], reverse=True)
        
        for i, (track_id, max_speed) in enumerate(max_speed_items[:5]):
            team = team_assignments.get(track_id, 0)
            color = self.team_colors[team]
            
            cv2.rectangle(panel, (20, y_offset - 15), (35, y_offset + 5), color, -1)
            
            cv2.putText(panel, f"#{track_id}", (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(panel, f"{max_speed:.1f} km/h", (100, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1)
            
            y_offset += 25
        
        y_offset += 15
        
        # Separator
        cv2.line(panel, (20, y_offset), (self.width - 20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        # Distance covered
        cv2.putText(panel, "DISTANCE COVERED", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        distance_items = [(tid, dist) for tid, dist in speed_tracker.total_distance.items()]
        distance_items.sort(key=lambda x: x[1], reverse=True)
        
        for i, (track_id, distance) in enumerate(distance_items[:5]):
            team = team_assignments.get(track_id, 0)
            color = self.team_colors[team]
            
            cv2.rectangle(panel, (20, y_offset - 15), (35, y_offset + 5), color, -1)
            
            cv2.putText(panel, f"#{track_id}", (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(panel, f"{distance:.0f} m", (100, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            y_offset += 25
        
        return panel
