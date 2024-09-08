# pip install ultralytics pillow matplotlib
# pip install opencv-python

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Path to Gender model
model = YOLO('./GenderModel_YOLOv10.pt')

# Load and open an image 
img_path = input("Please enter the path to the image: ")
try:
    image = Image.open(img_path)
except FileNotFoundError:
    print("Image not found. Please check the path and try again.")
    exit()

# Perform prediction
results = model.predict(image, verbose=False) #  suppresses the detailed logging

# Set confidence threshold
confidence_threshold = 0.5
prediction_made = False
# Iterate through results
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        confidence = box.conf.item()

        # Output based on confidence threshold
        if confidence >= confidence_threshold:
            prediction_made = True  # A valid prediction is made
            if class_id == 1:  # Female with confidence > 0.5
                print('female')
            elif class_id == 0:  # Male with confidence > 0.5
                print('male')

# If no prediction was made with sufficient confidence
if not prediction_made:
    print("No confident prediction found. Please upload another image.")
