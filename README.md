# 🧠 Smart Image Enhancement System

A complete Digital Image Processing (DIP) project that enhances low-quality images using multiple advanced techniques.

---

## 🚀 Features

- 📥 Image Acquisition & Visualization
- 🔍 Sampling & Quantization
- 🔄 Geometric Transformations (Rotation, Translation, Shearing)
- 🎨 Intensity Transformations (Negative, Log, Gamma)
- 📊 Histogram Equalization (Manual + OpenCV)
- ⚡ Final Smart Enhancement Pipeline

---

## 🔥 Enhancement Pipeline

The final system applies:

1. **Gamma Correction** → Brightens dark images  
2. **CLAHE** → Improves local contrast  
3. **Sharpening (Unsharp Mask)** → Enhances edges  
4. **Denoising** → Reduces noise while preserving details  

---

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Streamlit (UI)

---

## 📂 Project Structure



## ⚙️ Setup Instructions

Before running the project, follow these steps:

---

### 📁 1. Create Project Structure

Manually create the following folders on your PC:

DIP-Image-Enhancement-System/
├── code/
│ └── main.py
├── images/
│ ├── input/ # Place your input image here
│ └── output/
└── results/
### 📦 2. Install Required Libraries

Run the following commands in your terminal:
bash:
pip install opencv-python matplotlib numpy Pillow
pip install streamlit

▶️ 3. Run the Project

Navigate to the code folder and run:

streamlit run main.py
📌 Note

Make sure your input image is placed inside:

images/input/
Supported formats: .jpg, .png


# 💡 SMALL IMPROVEMENTS (OPTIONAL BUT PRO 🔥)

Instead of:
bash
pip install opencv-python matplotlib numpy Pillow
pip install streamlit
pip install opencv-python matplotlib numpy Pillow streamlit
