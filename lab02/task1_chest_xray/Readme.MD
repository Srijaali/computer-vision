Task 1 — Diagnostic Enhancement of Chest X-Rays

Objective

Enhance an underexposed grayscale chest X-ray using basic image-processing techniques to improve the visibility of intensity and structural variations.

Processing Pipeline

Load the X-ray as a grayscale image.
Apply Histogram Equalization for contrast enhancement.
Apply COLORMAP_JET for false-color visualization.
Perform Gray World color balancing.
Apply intensity thresholding to isolate high-intensity regions.
Apply logarithmic transformation to enhance darker regions.
Apply gamma correction with γ < 1 to improve midtones.
Key Equations

Log transformation:

s = c log(1 + r)

Gamma transformation:

s = c r^γ

Output

Processed images and comparison figures are stored in the output/ directory.

Run

Open and execute:

xray_enhancement.ipynb