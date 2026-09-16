# Task 3: Real-Time Echocardiogram Video Analysis

## Objective

This task performs real-time processing of an echocardiogram ultrasound video using OpenCV.

The purpose is to enhance low-contrast ultrasound frames and compare the raw video with the fully enhanced output.

---

## Dataset

The ultrasound video is taken from the **Stanford EchoNet-Dynamic Dataset**.

Only the short `.mp4` clip used for testing is included in the `data/` folder.

---

## Folder Structure

```text
Task_3_Echo_Analysis/
├── data/
│   └── echo.mp4
├── output/
│   ├── processed_echo.mp4
│   └── comparison.png
├── realtime_echo.ipynb
└── README.md
```

---

## Processing Pipeline

Each video frame is processed using the following steps:

1. Video Capture
2. Grayscale Conversion
3. Histogram Equalization
4. JET Color Mapping
5. Color Balance Adjustment
6. Logarithmic Transformation
7. Power-Law Transformation
8. Side-by-Side Comparison

### Video Capture

The video is read using:

```python
cv2.VideoCapture()
```

Frames are continuously extracted using a `while` loop until the video ends.

### Histogram Equalization

Histogram equalization is applied using:

```python
cv2.equalizeHist()
```

This improves the contrast of low-contrast ultrasound frames.

### JET Color Mapping

The equalized grayscale frame is converted into a false-color heatmap using:

```python
cv2.applyColorMap(equalized, cv2.COLORMAP_JET)
```

### Color Balance

A Gray-World color balancing method is applied to balance the Red, Green, and Blue channels.

### Logarithmic Transformation

The logarithmic transformation is:

```text
s = c log(1 + r)
```

where:

```text
c = 255 / log(256)
```

This helps reveal details in darker regions of the heart chambers.

### Power-Law Transformation

The power-law transformation is:

```text
s = 255(r / 255)^gamma
```

A gamma value of:

```text
gamma = 1.4
```

is used to reduce excessively bright ultrasound backscatter.

---

## Output

The raw ultrasound frame and fully enhanced frame are concatenated horizontally:

```text
Raw Ultrasound | Enhanced Ultrasound
```

The processed video is saved as:

```text
output/processed_echo.mp4
```

A sample comparison image is saved as:

```text
output/comparison.png
```

---

## Running the Notebook

This implementation is designed for **Google Colab**.

1. Open `realtime_echo.ipynb` in Google Colab.
2. Run the cells in order.
3. Upload the ultrasound `.mp4` file when prompted.
4. The uploaded video is saved as `data/echo.mp4`.
5. Run the processing cells.
6. View the final processed video directly inside Colab.

---

## Required Libraries

```text
opencv-python
numpy
matplotlib
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Google Colab
