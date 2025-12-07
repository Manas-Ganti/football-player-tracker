import cv2 
import numpy as np
import torch
from loguru import logger 
from ultralytics import YOLO
import supervision as sv
import tqdm 
from team_classify import *
from sklearn.cluster import KMeans
from homography import *


# function to draw circles under players and the player current speed
def draw(frame, detections, labels):
    for (xyxy, label) in zip(detections.xyxy, labels):

        x1, y1, x2, y2 = xyxy

        # Compute player foot position (midpoint bottom of bounding box)
        cx = int((x1 + x2) / 2)
        cy = int(y2)


        # Draw the circle under the player
        cv2.ellipse(frame, (cx, cy), (25, 15), 0, -30, 210, (0, 255, 0), 2)

        team_id = int(label.split()[1])

        if team_id == 1:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(
            frame,
            label,
            (cx - 20, cy - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    return frame

# function to draw stats panel
def stats(frame, player_data, h, panel_width):

    # Create a solid background panel (dark gray)
    panel = np.full((h, panel_width, 3), (30, 30, 30), dtype=np.uint8)

    y_offset = 50
    line_spacing = 23

    cv2.putText(panel, "PLAYER STATISTICS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Loop through players and print stats
    for player_id, pdata in player_data.items():
        if player_id <= 20:
            team = pdata['team']
            
            # Latest speed (if available)
            if len(pdata['speeds']) > 0:
                speed = np.max(pdata['speeds'])
                speed_text = f"{speed:.1f} km/h"
                mspeed = np.mean(pdata['speeds'])
                mspeed_text = f"{mspeed:.1f} km/h"
            else:
                speed_text = "--.- km/h"
                mspeed_text = "--.- km/h"

            # Choose color by team
            color = (255, 0, 0) if team == 1 else (0, 0, 255)

            text = f"#{player_id} | T{team}"

            cv2.putText(panel, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_spacing

            text = f"    Max Speed: {speed_text} | Avg Speed: {mspeed_text}"

            cv2.putText(panel, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_spacing

    # Combine panel with original frame
    combined = np.hstack((frame, panel))
    return combined

# get model from robo flow for getting key points from pitch
ROBOFLOW_API_KEY = "GU1FNPGRswx8gDRZNXII"
PITCH_MODEL_ID = "football-field-detection-f07vi/15"
PITCH_MODEL = get_model(PITCH_MODEL_ID, ROBOFLOW_API_KEY)

# get the video and model
video_path = 'videos/0bfacc_0.mp4' 
model_path = 'models/cv.pt'


# open video file to read frames
vid = cv2.VideoCapture(video_path)
crops = [] # to hold image crops
# get video fps, width, and height
fps = vid.get(cv2.CAP_PROP_FPS)
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_height = frame_height
panelW = int(frame_width*0.3)
out_width = frame_width+panelW


# print(frame_width) 1920
# print(frame_height) 1080

# creates an object to write out frames to a video file
output_path = 'videos/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

# load in detection model
model = YOLO(model_path)
logger.info('Loaded YOLO model from {}'.format(model_path))
# get an object that returns sampled frame from video, sends only every 50th frame
frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=50)

''' Training SigLip Umap KNN model '''

''' '0:ball', '1:goalkeeper', '2:player', 'referee' '''
logger.info('Training SigLip Umap KNN model on every 50 frames to classify players into teams')

# iterate through frames in the frame_generator, add a tqdm adds progress bar
for frame in tqdm(frame_generator):
    # run prediction on frame using yolo model
    result = model.predict(frame , device='cpu',conf=0.5)[0]
    # transform to supervision objects
    detections = sv.Detections.from_ultralytics(result)
    # take only the player detection by filtering class id = 2
    players_detections = detections[detections.class_id == 2]
    # for each bounding box (xyxy) in player_detections, crop out an image
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    # add the player crops to crops list
    crops += players_crops

# create a team classifier object
team_classifier = TeamClassifier(device="cpu")

# train the team classifier on the crops
team_classifier.fit(crops)
logger.info('Training done')

# reset the original video back to the start so we can go frame by frame and annotate
vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Initialize tracker to track detections across frames
tracker = sv.ByteTrack()

# Player tracking data: {tracker_id: {'positions': [(x, y, frame_num)], 'team': team_id, 'speeds': []}}
player_data = {}

# Pixel to meter conversion (adjust based on your field calibration)
PIXELS_PER_METER = 50.0

# Annotators to draw bounding boxes and data
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

# trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
frame_count = 0
total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# create virtual pitch to translate points onto
CONFIG = SoccerPitchConfiguration()


# create a progress bar
with tqdm(total=total_frames, desc="Processing") as pbar:
    while True:
        ret, frame = vid.read() # pull frame from video
        if not ret:
            break
        
        # increase the frame count
        frame_count += 1
        
        # Run detection using yolo model
        result = model.predict(frame, conf=0.3, device='cpu')[0]
        # convert to super vision
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter out ball detections and apply NMS
        players_detections = detections[detections.class_id == 2]
        players_detections = players_detections.with_nms(threshold=0.5, class_agnostic=True)
        
        # Update tracker (DON'T reset or recreate)
        players_detections = tracker.update_with_detections(detections=players_detections)
        
        # run pitch key points detection and get homograpgy matrix
        result_h = PITCH_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result_h)

        # filter out unwanted points
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter] # pitch reference

        # points transformer object to build homography matrix
        view_transformer = transformPoints(
            source = frame_reference_points,
            target = pitch_reference_points
        )

        # Classify teams for players
        if len(players_detections) > 0:
            # get cropped detections
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]


            # assign team_ids to detected players
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
                
                points = np.array([[x_center, y_center]])

                pitch_xy = view_transformer.transform(points)

                x_center = pitch_xy[0][0]
                y_center = pitch_xy[0][1]

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
                if len(player_data[tracker_id]['positions']) > int(fps/4) and (frame_count % int(fps/4) == 0):
                    # in the form x, y, frame
                    prev_pos = player_data[tracker_id]['positions'][-int(fps/4)]
                    curr_pos = player_data[tracker_id]['positions'][-1]
                    
                    # Calculate distance in pixels
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    distance_centimeters = np.sqrt(dx**2 + dy**2)
                    
                    # Convert to meters
                    distance_meters = distance_centimeters / 100
                    
                    # Calculate time difference
                    frame_diff = curr_pos[2] - prev_pos[2]
                    time_diff = frame_diff / fps
                    
                    # Calculate speed in m/s and convert to km/h
                    if time_diff > 0:
                        speed_ms = distance_meters / time_diff
                        speed_kmh = speed_ms * 3.6
                        player_data[tracker_id]['speeds'].append(speed_kmh)
            
        if len(players_detections) > 0:
            # Create simple labels with only team and current speed
            labels = []
            for tracker_id, team_id in zip(players_detections.tracker_id, team_ids):
                # Get current speed
                if tracker_id in player_data and len(player_data[tracker_id]['speeds']) > 0:
                    current_speed = player_data[tracker_id]['speeds'][-1]
                    label = f"Team {team_id} #{tracker_id} | {current_speed:.1f} km/h"
                else:
                    label = f"Team {team_id} #{tracker_id}"


                
                labels.append(label)
            
            # Annotate frame (removed trace_annotator)
            # annotated_frame = box_annotator.annotate(
            #     scene=frame.copy(), 
            #     detections=players_detections
            # )
            annotated_frame = draw(frame.copy(), players_detections, labels)
            
            

            # annotated_frame = label_annotator.annotate(
            #     scene=annotated_frame, 
            #     detections=players_detections, 
            #     labels=labels
            # )
        else:
            annotated_frame = frame.copy()
        

        outFrame = stats(annotated_frame, player_data, out_height, panelW)

        # Write frame
        out.write(outFrame)
        
        # Display frame (optional)
        cv2.imshow('Soccer Tracking', outFrame)
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




