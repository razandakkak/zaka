from flask import Flask, request, render_template, redirect, url_for, send_file
import os
from PIL import Image
import subprocess
from ultralytics import YOLO
import shlex


app = Flask(__name__)

# Define paths for gender model and video directories
base_dir = os.getcwd()  # Get the base directory
gender_model_path = os.path.join(base_dir, 'GenderModel_YOLOv10.pt')
male_videos = os.path.join(base_dir, 'male')
female_videos = os.path.join(base_dir, 'female')
video_to_actor_map = {
    os.path.join(female_videos, "Maguy.mp4"): os.path.join(female_videos, "julia.png"),
    os.path.join(female_videos, "May ez dine.mp4"): os.path.join(female_videos, "MayEzDine.png"),
    os.path.join(female_videos, "Nadine Njem.mp4"): os.path.join(female_videos, "Nadine.png"),
    os.path.join(female_videos, "Samia al jazaeri.mp4"): os.path.join(female_videos, "samia.png"),
    os.path.join(female_videos, "Shokran.mp4"): os.path.join(female_videos, "Torfa.png"),
    os.path.join(male_videos, "Ahmad Helmi.mp4"): os.path.join(male_videos, "XLARGE.png"),
    os.path.join(male_videos, "Joude.mp4"): os.path.join(male_videos, "Day3aDay3a.png"),
    os.path.join(male_videos, "Moetassem al nahar.mp4"): os.path.join(male_videos, "Moetassem.png"),
    os.path.join(male_videos, "Taym AL Hassan.mp4"): os.path.join(male_videos, "TaymAlHassan.png"),
    os.path.join(male_videos, "Mohamad Henedi.mp4"): os.path.join(male_videos, "julia.png")
}

# Load gender model
gender_model = YOLO(gender_model_path)

def detect_gender(user_face_path):
    image = Image.open(user_face_path)
    results = gender_model.predict(image, verbose=False)
    confidence_threshold = 0.5
    detected_gender = None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = box.conf.item()
            if confidence >= confidence_threshold:
                detected_gender = 'female' if class_id == 1 else 'male'
    
    return detected_gender

def clean_up(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            
def run_simswap(user_face_path, selected_video, actor_face):
    original_dir = os.getcwd()  # Save the original working directory
    simswap_dir = os.path.join(original_dir, 'SimSwap')  # SimSwap directory
    os.chdir(simswap_dir)  # Change to SimSwap directory
    
    try:
        # Define paths
        checkpoint_dir = os.path.abspath("./checkpoints")  # Checkpoints directory
        output_dir = os.path.abspath(os.path.join(simswap_dir, 'output'))  # Output directory
        temp_path = os.path.abspath('./temp_results')  # Temp directory

        # Create directories if they don't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        output_video_path = os.path.join(output_dir, os.path.basename(selected_video))  # Output video path

        # Command to run SimSwap
        command = f"""
        python test_video_swapspecific.py --no_simswaplogo --crop_size 224 --use_mask --pic_specific_path "{actor_face}" --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path "{user_face_path}" --video_path "{selected_video}" --checkpoints_dir "{checkpoint_dir}" --output_path "{output_video_path}" --temp_path "{temp_path}"
        """
        print(f"Running SimSwap with command: {command}")

        # Run SimSwap
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"SimSwap process completed successfully. Output: {stdout}")
            return send_file(os.path.abspath(output_video_path), as_attachment=True)
        else:
            print(f"SimSwap failed with error: {stderr}")
            raise Exception(f"SimSwap failed: {stderr}")

    finally:
        os.chdir(original_dir)  # Always revert to the original directory
        clean_up([user_face_path, temp_path])  # Clean up files after execution

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    upload_dir = os.path.join(base_dir, 'uploads')  # Ensure the upload directory is absolute
    if not os.path.exists(upload_dir):
        os.mkdir(upload_dir)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Detect gender
    detected_gender = detect_gender(file_path)
    if not detected_gender:
        return "Could not detect gender. Please try another image."

    # Display video options based on gender
    if detected_gender == 'male':
        video_options = [f for f in os.listdir(male_videos) if f.endswith('.mp4')]
    else:
        video_options = [f for f in os.listdir(female_videos) if f.endswith('.mp4')]
    
    return render_template('select_video.html', videos=video_options, gender=detected_gender, face_image=file_path)


@app.route('/run_simswap', methods=['POST'])
def run_simswap_route():
    selected_video = request.form['video']
    gender = request.form['gender']
    user_face_path = request.form['face_image']

    # Prepend the correct prefix based on gender
    selected_video = os.path.join(male_videos if gender == 'male' else female_videos, selected_video)

    # Determine actor's face
    actor_face = video_to_actor_map.get(selected_video)
    if actor_face is None:
        return "Selected video does not have a corresponding actor face.", 500

    # Run SimSwap
    try:
        return run_simswap(user_face_path, selected_video, actor_face)
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    app.run(debug=True)