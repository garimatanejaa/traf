from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    erode = cv2.erode(blur, np.ones((3, 3)))
    dilated = cv2.dilate(erode, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closing

# Helper function to detect cars using Haar Cascade classifier
def detect_cars(frame):
    processed = preprocess_image(frame)
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
    cars = car_cascade.detectMultiScale(processed, 1.1, 1)
    return cars

# Helper function to calculate car density
def calculate_car_density(frame, cars):
    frame_area = frame.shape[0] * frame.shape[1]
    car_area = sum([w*h for (x,y,w,h) in cars])
    density = car_area / frame_area
    return density

# Helper function to determine green light duration based on density
def determine_light_timing(density):
    if density < 0.1:
        return 30
    elif density < 0.3:
        return 60
    else:
        return 90

# Helper function to determine if the road is congested
def is_congested(density):
    return density > 0.5

# Function to process the video and analyze the traffic
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Debug log
    print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
    
    # Define the codec and create VideoWriter object (use a compatible codec)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Switching to XVID codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    results = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing all frames.")
            break
        
        # Detect cars and calculate metrics
        cars = detect_cars(frame)
        density = calculate_car_density(frame, cars)
        green_light_duration = determine_light_timing(density)
        congestion_status = is_congested(density)

        # Debug log for frame analysis
        print(f"Frame {frame_count}: Density={density:.2f}, GreenLight={green_light_duration}s, Congested={congestion_status}")
        
        # Draw rectangles around detected cars and overlay text
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Car Density: {density:.2f}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Green Light Duration: {green_light_duration}s", (10, 70), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Congested: {'Yes' if congestion_status else 'No'}", (10, 110), font, 0.7, (255, 255, 255), 2)

        out.write(frame)
        
        # Append frame analysis result
        results.append({
            'frame': frame_count,
            'density': density,
            'green_light_duration': green_light_duration,
            'congested': congestion_status
        })
        frame_count += 1

    cap.release()
    out.release()
    return results

# Route to handle video file upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_filepath)
        
        # Generate a unique identifier for this analysis
        analysis_id = str(uuid.uuid4())
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], f"{analysis_id}_output.mp4")
        
        # Process the video
        print(f"Processing video: {input_filepath}")
        results = process_video(input_filepath, output_filepath)
        print(f"Video processing complete. Output saved at: {output_filepath}")
        
        # Clean up the uploaded file
        os.remove(input_filepath)
        
        return jsonify({
            'results': results,
            'video_id': analysis_id
        }), 200
    return jsonify({'error': 'Invalid file type'}), 400

# Route to retrieve processed video file
@app.route('/video/<video_id>', methods=['GET'])
def get_video(video_id):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{video_id}_output.mp4")
    if os.path.exists(video_path):
        print(f"Serving video: {video_path}")
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'Video not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
