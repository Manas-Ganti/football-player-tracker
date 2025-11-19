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
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

frame_count = 0


while True:
    ret,frame = vid.read()
    if not ret:
        break

    tracker = sv.ByteTrack()
    tracker.reset()
    
    result = model.predict(frame, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    all_detections = detections[detections.class_id != 0]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections = tracker.update_with_detections(detections=all_detections)
    
    players_detections = all_detections[all_detections.class_id == 2]
     #Classify teams for players
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
            logger.info(f'Frame {frame_count} processed')
        # Create labels with team and speed info
        labels = []
        for tracker_id, team_id in zip(players_detections.tracker_id, team_ids):
            team_name = f"Team {team_id}"
            
            # Get current speed
            if tracker_id in player_data and len(player_data[tracker_id]['speeds']) > 0:
                current_speed = player_data[tracker_id]['speeds'][-1]
                avg_speed = np.mean(player_data[tracker_id]['speeds'])
                label = f"ID:{tracker_id} {team_name}\nSpeed:{current_speed:.1f}km/h\nAvg:{avg_speed:.1f}km/h"
            else:
                label = f"ID:{tracker_id} {team_name}"
            
            labels.append(label)
        
        # Set colors based on team
        colors = []
        for team_id in team_ids:
            if team_id == 0:
                colors.append(sv.Color.RED)
            elif team_id == 1:
                colors.append(sv.Color.BLUE)
            else:
                colors.append(sv.Color.GREEN)
        
        # Annotate frame
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), 
            detections=players_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=players_detections, 
            labels=labels
        )
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=players_detections
        )
    else:
        annotated_frame = frame.copy()
    
    # Write frame
    out.write(annotated_frame)
    
    # Display frame (optional - comment out for faster processing)
    cv2.imshow('Soccer Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # pbar.update(1)
