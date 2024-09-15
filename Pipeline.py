!pip install -r requirements.txt
import torch
import torchvision
import moviepy
import os
from ultralytics import YOLO
from PIL import Image
import subprocess
from google.colab import files
uploaded = files.upload()
user_face_filename = list(uploaded.keys())[0]
user_face_path = os.path.join(os.getcwd(), user_face_filename)

gender_model_path = './GenderModel_YOLOv10.pt'
gender_model = YOLO(gender_model_path)

def DetectGender(user_face_path):
    image = Image.open(user_face_path)
    # Perform prediction
    results = gender_model.predict(image, verbose=False)
    confidence_threshold = 0.5
    prediction_made = False
    detected_gender = None

    # Iterate through results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = box.conf.item()
            # Output based on confidence threshold
            if confidence >= confidence_threshold:
                prediction_made = True  # A valid prediction is made
                if class_id == 1:  # Female with confidence > 0.5
                    detected_gender = 'female'
                elif class_id == 0:  # Male with confidence > 0.5
                    detected_gender = 'male'

    if not prediction_made:
        print("No confident prediction found. Please upload another image.")
        return None
    else:
        print(f"Detected gender: {detected_gender}")
        return detected_gender
print(DetectGender(user_face_path))

def show_videos(gender):
    if gender == 'male':
        video_options = os.listdir(male_videos)
    else:
        video_options = os.listdir(female_videos)

    selected_video = user_select_video(video_options, gender)
    return selected_video


def user_select_video(video_options, gender):
    if gender == 'male':
        folder_path = male_videos
    else:
        folder_path = female_videos

    # Filter out image files by only allowing video files (assuming .mp4 is the video extension)
    video_files = [video for video in video_options if video.endswith('.mp4')]

    for idx, video in enumerate(video_files):
        print(f"{idx+1}: {video}")
    choice = int(input("Select a video by number: ")) - 1
    selected_video = os.path.join(folder_path, video_files[choice])
    return selected_video


def run_simswap_from_selection(user_face_path, selected_video, checkpoint_dir):
    # Find the actor's face associated with the selected video
    actor_face = video_to_actor_map[selected_video]

    output_video_path = "/content/output_video.mp4"

    command = f"""
    python test_video_swapspecific.py --no_simswaplogo --crop_size 224 --use_mask \
    --pic_specific_path "{actor_face}" \
    --name people \
    --Arc_path arcface_model/arcface_checkpoint.tar \
    --pic_a_path "{user_face_path}" \
    --video_path "{selected_video}" \
    --checkpoints_dir "{checkpoint_dir}" \
    --output_path "{output_video_path}" \
    --temp_path ./temp_results
    """

    # Execute the command in a subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"SimSwap completed successfully! Output video saved to {output_video_path}")
    else:
        print(f"Error occurred during SimSwap execution: {stderr}")


male_videos = "./male"
female_videos = "./female"

video_to_actor_map = {
    "./female/Maguy.mp4": "./female/julia.png",
    "./female/May ez dine.mp4": "./female/MayEzDine.png",
    "./female/Nadine Njem.mp4": "./female/Nadine.png",
    "./female/Samia al jazaeri.mp4": "./female/samia.png",
    "./female/Shokran.mp4": "./female/Torfa.png",
    "./male/Ahmad Helmi.mp4": "./male/XLARGE.png",
    "./male/Joude.mp4": "./male/Day3aDay3a.png",
    "./male/Moetassem al nahar.mp4": "./male/Moetassem.png",
    "./male/Taym AL Hassan.mp4": "./male/TaymAlHassan.png",
    "./male/Mohamad Henedi.mp4": "./male/julia.png"}

%cd SimSwap
gender = DetectGender(user_face_path)
selected_video = show_videos(gender)

# Now, run SimSwap with the selected video and corresponding actor's face
run_simswap_from_selection(user_face_path, selected_video, "./checkpoints")
