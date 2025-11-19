import cv2 
import numpy as np
import torch
from loguru import logger 
from ultralytics import YOLO
import supervision as sv
import tqdm 
from team_classify import *
# from live_stats import *
# from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = 'videos/121364_0.mp4' 
model_path = 'models/best-3.pt'

vid = cv2.VideoCapture(video_path)
crops = []
fps = vid.get(cv2.CAP_PROP_FPS)
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))


output_path = 'videos/output_tracking.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


model = YOLO(model_path)
logger.info('Loaded YOLO model from {}'.format(model_path))
frame_generator = sv.get_video_frames_generator(
    source_path=video_path, stride=50)

''' Training SigLip Umap KNN model '''

''' '0:ball', '1:goalkeeper', '2:player', 'referee' '''
logger.info('Training SigLip Umap KNN model on every 50 frames to classify players into teams')

for frame in tqdm(frame_generator):
    result = model.predict(frame , device='mps',conf=0.5)[0]
    detections = sv.Detections.from_ultralytics(result)
    players_detections = detections[detections.class_id == 2]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops

team_classifier = TeamClassifier(device="mps")

team_classifier.fit(crops)

logger.info('Training done')

vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Initialize tracker
tracker = sv.ByteTrack()

# Player tracking data: {tracker_id: {'positions': [(x, y, frame_num)], 'team': team_id}}
player_data = {}

# Pixel to meter conversion (adjust based on your field calibration)
PIXELS_PER_METER = 10.0  # Approximate - adjust based on known field dimensions

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
# trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

frame_count = 0


total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc="Processing") as pbar:
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        result = model.predict(frame, conf=0.3, device='mps')[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter out ball detections and apply NMS
        all_detections = detections[detections.class_id != 0]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        
        # Update tracker (DON'T reset or recreate)
        all_detections = tracker.update_with_detections(detections=all_detections)
        
        # Get player detections
        players_detections = all_detections[all_detections.class_id == 2]
        
        # Classify teams for players
        if len(players_detections) > 0:
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            team_ids = team_classifier.predict(player_crops)
            
            # Process each player
            for i, (xyxy, tracker_id, team_id) in enumerate(zip(
                players_detections.xyxy, 
                players_detections.tracker_id, 
                team_ids
            )):
                # Calculate center position
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                
                # Initialize player data if new
                if tracker_id not in player_data:
                    player_data[tracker_id] = {
                        'positions': [],
                        'team': team_id,
                        'speeds': []
                    }
                
                # Store position and frame
                player_data[tracker_id]['positions'].append((x_center, y_center, frame_count))
                player_data[tracker_id]['team'] = team_id
                
                # Calculate speed if we have previous position
                if len(player_data[tracker_id]['positions']) > 1:
                    prev_pos = player_data[tracker_id]['positions'][-2]
                    curr_pos = player_data[tracker_id]['positions'][-1]
                    
                    # Calculate distance in pixels
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    distance_pixels = np.sqrt(dx**2 + dy**2)
                    
                    # Convert to meters
                    distance_meters = distance_pixels / PIXELS_PER_METER
                    
                    # Calculate time difference
                    frame_diff = curr_pos[2] - prev_pos[2]
                    time_diff = frame_diff / fps
                    
                    # Calculate speed in m/s and convert to km/h
                    if time_diff > 0:
                        speed_ms = distance_meters / time_diff
                        speed_kmh = speed_ms * 3.6
                        player_data[tracker_id]['speeds'].append(speed_kmh)
            
            # Create simple labels with only team and current speed
            labels = []
            for tracker_id, team_id in zip(players_detections.tracker_id, team_ids):
                # Get current speed
                if tracker_id in player_data and len(player_data[tracker_id]['speeds']) > 0:
                    current_speed = player_data[tracker_id]['speeds'][-1]
                    label = f"Team {team_id} | {current_speed:.1f} km/h"
                else:
                    label = f"Team {team_id}"
                
                labels.append(label)
            
            # Annotate frame (removed trace_annotator)
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), 
                detections=players_detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, 
                detections=players_detections, 
                labels=labels
            )
        else:
            annotated_frame = frame.copy()
        
        # Write frame
        out.write(annotated_frame)
        
        # Display frame (optional)
        cv2.imshow('Soccer Tracking', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        pbar.update(1)
vid.release()
out.release()
cv2.destroyAllWindows()

# Calculate and display statistics
logger.info("\n" + "="*60)
logger.info("PLAYER STATISTICS")
logger.info("="*60)

for tracker_id, data in player_data.items():
    if len(data['speeds']) > 0:
        avg_speed = np.mean(data['speeds'])
        max_speed = np.max(data['speeds'])
        team = data['team']
        
        logger.info(f"\nPlayer ID: {tracker_id} (Team {team})")
        logger.info(f"  Average Speed: {avg_speed:.2f} km/h")
        logger.info(f"  Max Speed: {max_speed:.2f} km/h")
