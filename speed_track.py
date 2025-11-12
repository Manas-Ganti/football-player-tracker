
class SpeedTracker:
    """Tracks and calculates player speeds"""
    
    def __init__(self, fps: float, field_width_meters: float = 105, 
                 field_height_meters: float = 68):
        """
        Initialize speed tracker
        
        Args:
            fps (float): Video frame rate
            field_width_meters (float): Real-world field width in meters
            field_height_meters (float): Real-world field height in meters
        """
        self.fps = fps
        self.positions = defaultdict(lambda: deque(maxlen=10))
        self.speeds = {}
        self.max_speeds = defaultdict(float)
        self.total_distance = defaultdict(float)
        self.field_width_meters = field_width_meters
        self.field_height_meters = field_height_meters
        self.pixels_to_meters = None
    
    def calibrate_scale(self, frame_width: int, frame_height: int):
        """Calibrate pixel to meter conversion"""
        self.pixels_to_meters = self.field_width_meters / frame_width
    
    def update_position(self, track_id: int, bbox: List[float]) -> float:
        """
        Update position and calculate speed
        
        Args:
            track_id (int): Player tracking ID
            bbox (List[float]): Bounding box [x1, y1, x2, y2]
            
        Returns:
            float: Current speed in km/h
        """
        # Get center of bounding box
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        
        # Calculate distance traveled since last position
        if len(self.positions[track_id]) > 0:
            prev_pos = self.positions[track_id][-1]
            dx = x_center - prev_pos[0]
            dy = y_center - prev_pos[1]
            distance_pixels = np.sqrt(dx**2 + dy**2)
            
            if self.pixels_to_meters is None:
                distance_meters = distance_pixels * 0.05  # rough estimate
            else:
                distance_meters = distance_pixels * self.pixels_to_meters
            
            self.total_distance[track_id] += distance_meters
        
        self.positions[track_id].append((x_center, y_center))
        
        # Need at least 2 positions to calculate speed
        if len(self.positions[track_id]) < 2:
            self.speeds[track_id] = 0.0
            return 0.0
        
        # Calculate displacement over last few frames
        positions = list(self.positions[track_id])
        frames_back = min(int(self.fps * 0.5), len(positions) - 1)
        
        pos_current = positions[-1]
        pos_previous = positions[-frames_back - 1]
        
        # Calculate pixel distance
        dx = pos_current[0] - pos_previous[0]
        dy = pos_current[1] - pos_previous[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        if self.pixels_to_meters is None:
            distance_meters = distance_pixels * 0.05
        else:
            distance_meters = distance_pixels * self.pixels_to_meters
        
        # Calculate speed (m/s)
        time_elapsed = frames_back / self.fps
        speed_ms = distance_meters / time_elapsed if time_elapsed > 0 else 0
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        # Smooth speed with exponential moving average
        if track_id in self.speeds:
            speed_kmh = 0.7 * self.speeds[track_id] + 0.3 * speed_kmh
        
        self.speeds[track_id] = speed_kmh
        
        # Update max speed
        if speed_kmh > self.max_speeds[track_id]:
            self.max_speeds[track_id] = speed_kmh
        
        return speed_kmh
