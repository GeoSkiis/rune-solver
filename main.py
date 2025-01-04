from fastapi import FastAPI, File, UploadFile, HTTPException #type: ignore
import uvicorn
import cv2
import tensorflow as tf
import numpy as np
import os
import datetime
from PIL import Image
import torchvision.transforms as T
import torchvision
import torch
import random
from torchvision.models.detection import maskrcnn_resnet50_fpn
from dotenv import load_dotenv
import os


def detect_arrows(img_path):
    results = []
    img = cv2.imread(img_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            centroid = (cx, cy)
        else:
            continue

        directions = ["up", "down", "left", "right"]
        final_line = {}

        for direction in directions:
            if direction == "down":
                dist = 0
                while cy - dist >= 0:
                    if all(binary[cy + dist, cx + i] == 255 for i in range(-5, 5)):
                        dist += 1
                    elif all(binary[cy + dist, cx + i] == 0 for i in range(-5, 5)):
                        final_line["down"] = sum(binary[cy + dist - 2, cx + i] == 255 for i in range(-5, 5))
                        break
                    else:
                        dist += 1
            elif direction == "up":
                dist = 0
                while cy + dist < img.shape[0]:
                    if all(binary[cy - dist, cx + i] == 255 for i in range(-5, 5)):
                        dist += 1
                    elif all(binary[cy - dist, cx + i] == 0 for i in range(-5, 5)):
                        final_line["up"] = sum(binary[cy - dist + 2, cx + i] == 255 for i in range(-5, 5))  
                        break
                    else:
                        dist += 1
            elif direction == "left":
                dist = 0
                while cx - dist >= 0:
                    if all(binary[cy + i, cx - dist] == 255 for i in range(-5, 5)):
                        dist += 1
                    elif all(binary[cy + i, cx - dist] == 0 for i in range(-5, 5)):
                        final_line["left"] = sum(binary[cy + i, cx - dist + 2] == 255 for i in range(-5, 5))
                        break
                    else:
                        dist += 1
            elif direction == "right":
                dist = 0
                while cx + dist < img.shape[1]:
                    if all(binary[cy+i, cx + dist] == 255 for i in range(-5, 5)):
                        dist += 1
                    elif all(binary[cy+i, cx + dist] == 0 for i in range(-5, 5)):
                        final_line["right"] = sum(binary[cy + i, cx + dist - 2] == 255 for i in range(-5, 5))
                        break
                    else:
                        dist += 1

        results.append({
            "centroid": centroid,
            "last_line": final_line,
        })

    results.sort(key=lambda x: x["centroid"][0])
    inverse_directions = []
    for result in results:
        print(result, flush=True)
        max_direction = max(result["last_line"], key=result["last_line"].get)

        # Determine the inverse direction
        inverse_direction = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }[max_direction]
        inverse_directions.append(inverse_direction)

    return inverse_directions

def crop_image_new(frame):
    height, width, _ = frame.shape

    # Calculate the crop dimensions based on percentage
    left = int(width * 0.20)
    right = int(width * 0.20)
    top = int(height * 0.20)
    bottom = int(height * 0.40)

    # Define the crop area
    x = left
    y = top
    w = width - left - right
    h = height - top - bottom

    cropped_frame = frame[y:y+h, x:x+w]

    # Save the cropped frame (for reference)
    cv2.imwrite('crop_rough.png', cropped_frame)

    # return cropped_frame


# Load the trained model
def load_arrow_model(model_path, num_classes):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # Load a pre-trained Mask R-CNN model with no weights
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Get the number of input features for the box classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the box predictor head
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
    
    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    # Load the checkpoint and extract only the model's state_dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state dict
    
    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def filter_color_new(image_path, output_path):
    img = Image.open(image_path).convert("RGB")

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        prediction = arrow_model(img_tensor)[0]
    
    masks = prediction['masks'] > 0.4

    confidence_threshold = 0.2
    valid_indices = [i for i, score in enumerate(prediction['scores'].cpu().numpy()) if score > confidence_threshold]
    masks = masks[valid_indices].squeeze(1).cpu().numpy()
    
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

    img_array = np.array(img)
    img_array[combined_mask == 0] = 0

    output_img = Image.fromarray(img_array)
    output_img.save(output_path)
    print(f"Saved image with background removed to {output_path}")


def merge_detection(image):
    """
    Run two inferences: one on the upright image, and one on the image rotated 90 degrees.
    Only considers vertical arrows and merges the results of the two inferences together.
    (Vertical arrows in the rotated image are actually horizontal arrows).
    :param model:   The model object to use.
    :param image:   The input image.
    :return:        A list of four arrow directions.
    """
    crop_image_new(image)
    try:
        filter_color_new("crop_rough.png", "filtered.png")
        # Run inference on the filtered image
        return detect_arrows("filtered.png")
    except Exception as e:
        print("no arrows detected, returning empty array", flush=True)
        return []

model_path = "mask_rcnn_arrows_epoch_3.pth"
# Number of classes (background + arrow class)
num_classes = 2  # 1 background + 1 arrow

arrow_model = load_arrow_model(model_path, num_classes)

count = 0

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Adjust this to specific HTTP methods if needed
    allow_headers=["*"],
)

load_dotenv()

@app.post("/")
async def execute_code(file: UploadFile = File(...)):
    global count
    print(count)
    count += 1
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    file_path = "received_frame.png"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    frame = cv2.imread(file_path)
    cv2.imwrite("raw_frames/"+str(count)+'_image.png', frame)

    result = merge_detection(frame)
    print(result)
    return result

if __name__ == "__main__":
    host = str(os.getenv('HOST', '0.0.0.0'))
    port = int(os.getenv('PORT', 8100))
    uvicorn.run(app, host=host, port=port)
