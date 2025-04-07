import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import serial
import csv
import time
from datetime import datetime
import requests

def capture_ip_camera_frame(ip_camera_url, save_path):
    """
    Captures a frame from an IP camera and saves it as an image using OpenCV.

    :param ip_camera_url: URL of the IP camera (e.g., "http://<IP>:<port>/shot.jpg").
    :param save_path: Path to save the captured image.
    :return: True if successful, False otherwise.
    """
    try:
        response = requests.get(ip_camera_url, stream=True)
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                cv2.imwrite(save_path, image)
                print(f"Image saved at: {save_path}")
                return True
            else:
                print("Failed to decode image from the URL.")
                return False
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error capturing image: {e}")
        return False

def get_next_csv_row(csv_file_path, tracker_file_path):
    """
    Get the next row of data from the CSV file based on the tracker file.
    """
    try:
        # Load CSV data
        with open(csv_file_path, 'r') as file:
            reader = list(csv.DictReader(file))

        # Get the last used row index from the tracker file
        try:
            with open(tracker_file_path, 'r') as tracker_file:
                content = tracker_file.read().strip()
                last_index = int(content) if content.isdigit() else -1
        except FileNotFoundError:
            last_index = -1  # Start with the first row if tracker doesn't exist

        # Calculate the next index
        next_index = (last_index + 1) % len(reader)

        # Save the new index back to the tracker file
        with open(tracker_file_path, 'w') as tracker_file:
            tracker_file.write(str(next_index))

        # Return the next row
        return reader[next_index]

    except Exception as e:
        print(f"Error accessing CSV or tracker file: {e}")
        return None

def load_model(model_path, num_classes):
    """
    Load a Mask R-CNN model.
    """
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_and_segment(model, image_path, confidence_threshold=0.5):
    """
    Perform prediction and generate binary mask.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    # Convert BGR to RGB and to tensor
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = F.to_tensor(image_rgb).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        predictions = model(input_image)

    # Extract masks and scores
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']

    # Filter by confidence threshold
    mask_indices = torch.where(scores > confidence_threshold)[0]

    # Create binary mask
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for idx in mask_indices:
        mask = masks[idx, 0].numpy()
        binary_mask = np.logical_or(binary_mask, mask > 0.5).astype(np.uint8)

    # Convert binary mask to 0-255 for visualization
    binary_mask_255 = binary_mask * 255

    # Calculate the number of white pixels
    white_pixel_count = np.sum(binary_mask_255 == 255)

    # Total pixels in the image
    total_pixels = binary_mask_255.size

    return binary_mask_255, white_pixel_count, total_pixels

def get_distance_from_arduino(arduino_port="/dev/ttyACM0", baud_rate=9600):
    """
    Get distance data from Arduino at the moment of capture.
    """
    try:
        ser = serial.Serial(arduino_port, baud_rate, timeout=2)
        time.sleep(2)  # Allow time for the connection to stabilize
        distance_data = ser.readline().decode("utf-8").strip()
        ser.close()
        return float(distance_data)  # Convert to float for comparison
    except Exception as e:
        print(f"Error reading from Arduino: {e}")
        return None

if __name__ == "__main__":
    # Configuration
    ip_camera_url = "http://192.168.98.181:8080/shot.jpg"
    image_path = r"C:\Users\harsh\OneDrive\Desktop\real_time_monitoring_model\champ.jpg"
    model_path = r"C:\Users\harsh\OneDrive\Desktop\real_time_monitoring_model\models\folder5H_rcnn_weights.pth"
    output_path = r"C:\Users\harsh\OneDrive\Desktop\real_time_monitoring_model\binary_mask.jpg"
    csv_file_path = r"C:\Users\harsh\OneDrive\Desktop\real_time_monitoring_model\height.csv"
    tracker_file_path = r"C:\Users\harsh\OneDrive\Desktop\real_time_monitoring_model\want.txt"
    num_classes = 2
    confidence_threshold = 0.5
    arduino_port = "COM15"
    baud_rate = 9600

    # Get the next row from the CSV
    csv_row = get_next_csv_row(csv_file_path, tracker_file_path)
    if csv_row is None:
        print("No valid row found in the CSV. Exiting...")
        exit()

    # Extract CSV data
    csv_height = float(csv_row["height"])
    csv_pixels = int(csv_row["pixels"])
    house_id = csv_row["house"]

    # Capture image from IP camera
    if capture_ip_camera_frame(ip_camera_url, image_path):
        print("Image capture successful!")

        # Get distance data from Arduino
        measured_height = 150 - get_distance_from_arduino(arduino_port, baud_rate)
        if measured_height:
            # Load model
            model = load_model(model_path, num_classes)

            # Perform prediction
            binary_mask, white_pixel_count, total_pixels = predict_and_segment(
                model, image_path, confidence_threshold
            )

            # Save the binary mask image
            cv2.imwrite(output_path, binary_mask)
            print(f"Binary mask saved at: {output_path}")

            # Compare measured height with CSV data
            height_difference = abs(measured_height - csv_height)

            # Compare calculated white pixels with CSV data
            pixel_difference = abs(white_pixel_count - csv_pixels)

            # Display the results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nTimestamp: {timestamp}")
            print(f"Measured Height: {measured_height:.2f} cm")
            print(f"CSV Height (House {house_id}): {csv_height:.2f} cm")
            print(f"Height Difference: {height_difference:.2f} cm")
            print(f"Calculated Pixels: {white_pixel_count}")
            print(f"CSV Pixels (House {house_id}): {csv_pixels}")
            print(f"Pixel Difference: {pixel_difference}")
        else:
            print("Failed to get distance data from Arduino.")
    else:
        print("Image capture failed.")
