# illigal-construction-detection-part-2-
# 🛰️ Real-Time Monitoring of Illegal Construction

This project aims to detect unauthorized or illegal building constructions by monitoring the **area** and **height** of buildings in real-time using drones, deep learning, and sensor-based systems.

---

## 🎯 Objective

To develop a drone-based surveillance system that:
- Detects the **actual area** of a building through aerial image segmentation.
- Measures the **height** of the building using onboard sensors.
- Compares these values with legally allotted data.
- Flags any structure that exceeds its permitted dimensions.

---

## 🔧 How It Works

1. **Drone Navigation**:
   - A drone follows a pre-defined path to fly over buildings.

2. **Image Capture**:
   - A **smartphone** is used as an **IP webcam** mounted on the drone.
   - The Raspberry Pi 4 captures an image from the camera (via URL) and saves it as `champ.jpg`.

3. **Area Detection (Segmentation)**:
   - A **Mask R-CNN** model trained on a custom dataset segments the house/building from the image.
   - The number of white (segmented) pixels is counted in the binary mask.
   - The real-world area is calculated using a pixel-to-square-meter ratio.

   📌 **Model training and dataset**: [View here](https://github.com/Harshbeast/ILLIGAL-CONSTRUCTION-DETECTOIN-MCD-)

4. **Height Detection**:
   - An **ultrasonic sensor** is mounted on the drone and connected to an Arduino Uno.
   - The sensor measures the distance from the drone to the rooftop.
   - Height = Drone altitude (150 cm for this test) – distance from rooftop.

5. **Validation**:
   - Data from a reference CSV (`height.csv`) includes the allotted height and area for each house.
   - The system compares real-time values with this reference.
   - If there is a significant deviation, the structure is **flagged**.

---

## 🧪 Experimental Setup

- **Drone + Raspberry Pi 4** for onboard processing and camera interface.
- **Smartphone** configured as an **IP camera** (e.g., `http://192.168.98.181:8080/shot.jpg`).
- **Ultrasonic sensor** for height measurement (TOF sensor planned for future versions).
- **Arduino Uno** for reading sensor values and sending them to the main script.
- **Mask R-CNN** trained on a custom dataset for segmentation.

---

## 📂 File Structure

```plaintext
real_time_monitoring/
├── champ.jpg                 # Captured aerial image
├── binary_mask.jpg           # Segmented output mask
├── height.csv                # Reference legal data (height, pixels)
├── want.txt                  # Tracker file for row selection
├── models/
│   └── folder5H_rcnn_weights.pth  # Trained model weights

CSV Format
house,height,pixels
A1,130,14256
A2,150,16580
...
output example:
Image saved at: champ.jpg
Model loaded successfully.
Binary mask saved at: binary_mask.jpg

Timestamp: 2025-04-07 12:45:32
Measured Height: 132.54 cm
CSV Height (House A1): 130.00 cm
Height Difference: 2.54 cm
Calculated Pixels: 14420
CSV Pixels (House A1): 14256
Pixel Difference: 164

