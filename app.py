import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import base64
from streamlit_option_menu import option_menu
import torch
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import random
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, confusion_matrix
import timm
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

# Set the page layout to wide mode
st.set_page_config(layout="wide", page_title="Breast Cancer Classifier")

# Define the Resnet50 classifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['benign','malignant']
minority_classes = ['malignant']

def Resnet50():
    model = timm.create_model('efficientnet_b4', pretrained=True)

# ตรวจสอบจำนวน input features ของ Fully Connected Layer
    num_ftrs = model.classifier.in_features

# แทนที่ Fully Connected Layer เพื่อให้รองรับ 2 คลาส
    model.classifier = torch.nn.Linear(num_ftrs, 2)


    model = model.to(device)
    return model

from torch.utils.data import Dataset
class_labels = ['benign','malignant']

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['Image_Path']
        label = class_labels.index(self.dataframe.iloc[idx]['Label'])  # Convert label to index
        image = Image.open(image_path)  # .convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

import requests

def load_classification_model():
    try:
        # Create the model architecture first (e.g., using ResNet50 or VGG16)
        classification_model = Resnet50()  # Make sure Vgg16 or ResNet50 is correctly defined in your code

        # Download model weights from the URL
        url = "https://github.com/fernchunyar/Final_Proj/releases/download/v.03/12MarFern+Weight15Fuzzy30epoch_efficientnet_b4.pth"
        response = requests.get(url)
        
        # Save the model weights to the local file
        with open('12MarFern+Weight15Fuzzy30epoch_efficientnet_b4.pth', 'wb') as f:
            f.write(response.content)

        # Load the saved weights into the model
        classification_model.load_state_dict(torch.load('12MarFern+Weight15Fuzzy30epoch_efficientnet_b4.pth', map_location=device, weights_only=False))

        # Set the model to evaluation mode
        classification_model.eval()

        return classification_model

    except Exception as e:
        st.error(f"Failed to load classification model: {e}")
        return None

from torchvision import transforms
import torch
import numpy as np
from PIL import Image

# Define the minority classes in your dataset
class_names = ['benign', 'malignant']
minority_classes = ['malignant']

# Define custom data transformations for minority classes
minority_class_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.9),  # Apply with 90% probability
    transforms.RandomRotation(15, expand=False, center=None),
    # Uncomment if you want to add color jittering
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Define data transformations for train, validation, and test sets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        # Apply custom augmentations to minority classes
        transforms.RandomApply([minority_class_transforms], p=0.5) if any(cls in minority_classes for cls in class_names) else transforms.RandomApply([], p=0.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def classify_image(image, model, device):
    try:
        # Ensure the image has three channels (RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply the standard transformations (resize, crop, normalize)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Transform the image and prepare it for model input
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy().squeeze() * 100  # Convert to percentages

        # Map predictions to labels
        labels = {0: "Benign", 1: "Malignant"}
        probabilities = {labels[i]: round(probs[i], 2) for i in range(len(labels))}
        predicted_class = np.argmax(probs)  # Get the class with the highest probability

        return labels[predicted_class], probabilities

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return None, None

# Create a guideline page
def guideline_page():
    st.title("Guidelines for Using the Breast Cancer Classification App")

    # Create a more modern language selection dropdown with Thai as the default
    lang_button = st.selectbox("Languages:", ("Thai", "English"))  # Start with Thai

    if lang_button == "English":
        st.markdown(
            """
            This app classifies breast ultrasound images into two categories:
            **Benign** or **Malignant**. Follow these steps to use the app:
            """
        )
        # Display steps in English
        st.markdown(
            """
            ### Steps:
            1. Go to the **Classification** page using the navigation bar.
            2. Upload a breast ultrasound image using the **Upload Image** button.
            3. Wait for the app to process the image and display the results:
               - **Classification Result**: Shows whether the image is Benign or Malignant.
               - **Predicted Probabilities**: Displays the likelihood of each class.
            4. Repeat the process to classify another image.
            """
        )

        # Add an example image for visual aid
        st.image("intro_img.png", caption="Example Output", use_container_width=True)

        st.markdown(
            """
            ### Notes:
            - Ensure that the image is a valid breast ultrasound scan in `.jpg`, `.jpeg`, or `.png` format.
            - The app may take a few seconds to process your image.
            - For best results, please upload clear and high-quality images.
            """
        )

        st.success("You are now ready to proceed to the **Classification** page!")
    
    elif lang_button == "Thai":
        st.markdown(
            """
            แอปนี้จำแนกภาพอัลตราซาวด์เต้านมเป็นสองประเภท คือ
            **Benign (ไม่เป็นมะเร็ง)** หรือ **Malignant (เป็นมะเร็ง)** โดยมีขั้นตอนการใช้งานแอปดังนี้:
            """
        )
        # Display steps in Thai
        st.markdown(
            """
            ### ขั้นตอน:
            1. ไปที่หน้า **Classification** ผ่านแถบเมนู
            2. อัปโหลดภาพอัลตราซาวด์เต้านม โดยคลิกที่ปุ่ม **Upload Image**
            3. รอให้แอปประมวลผลภาพและแสดงผลลัพธ์:
               - **Classification Result**: แสดงว่าเป็น Benign หรือ Malignant
               - **Predicted Probabilities**: แสดงความน่าจะเป็นของแต่ละประเภท
            4. สามารถคลิกปุ่มอีกครั้งเพื่อเปลี่ยนภาพได้
            """
        )

        # Add an example image for visual aid
        st.image("intro_img.png", caption="ตัวอย่างผลลัพธ์", use_container_width=True)

        st.markdown(
            """
            ### หมายเหตุ:
            - ตรวจสอบให้แน่ใจว่าภาพอยู่ในรูปแบบ `.jpg`, `.jpeg`, หรือ `.png`
            - แอปอาจใช้เวลาสักครู่ในการประมวลผลภาพ
            - เพื่อผลลัพธ์ที่ดีที่สุด ควรอัปโหลดภาพที่ชัดเจนและมีคุณภาพสูง
            """
        )

        st.success("ตอนนี้คุณพร้อมที่จะไปที่หน้า **Classification** แล้ว!")


        
# Create a classification page
def classification_page():
    # Display logos and header
    with open("swulogo.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
     # Display logos and header
    with open("logo_img.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="350" style="margin-bottom: 10px;">
    </div>
    <div style="text-align: center;">
        <h1 style="font-size: 48px; font-weight: bold;">
            Breast Cancer Ultrasound Classification
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown(
        """
        <h3 style="font-size: 24px;">
            Upload an image for classification.
        </h3>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Load the model and device
        classification_model = load_classification_model()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if classification_model:
            with st.spinner("Processing..."):
                # Pass the device to the classify_image function
                result, probabilities = classify_image(image, classification_model, device)

                if result:
                    # Two-column layout to display the image and classification result
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(image, caption="Uploaded Image", use_container_width=True)

                    with col2:
                        # Set color based on classification result
                        if result == "Benign":
                            color = "green"
                        else:
                            color = "red"
                        
                        st.markdown(
                            f"""
                            <h1 style="font-size: 30px; color:{color};">
                                Classification Result: {result}
                            </h1>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.write("### Predicted Probabilities:")
                        for label, prob in probabilities.items():
                            st.markdown(f"**{label}:** {prob:.2f}%")


# Sidebar menu logic with option menu
with st.sidebar:

    # Render the option menu
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Guideline", "Classification"],  # required
        icons=["book", "bar-chart"],  # optional icons
        menu_icon="house",  # optional menu icon
        default_index=0,  # default selection
    )


# Display the page based on the sidebar selection
if selected == "Guideline":
    guideline_page()
elif selected == "Classification":
    classification_page() 
