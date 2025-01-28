import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import base64
from streamlit_option_menu import option_menu
import torch
import torch.nn as nn
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

# Define the VGG classifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['benign','malignant']
minority_classes = ['malignant']

# Transfer Learning by fineTuning the pretrained Resnet101 Model
# Load Resnet101 pretrained Model
# If pretrained is not working, you can also use weights instead.
# def ResNet101():
#     Resnet101 = models.resnet101(pretrained=True)
#     Resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)  # Update for torchvision >= 0.13
#     for param in Resnet101.parameters():
#         param.requires_grad = True
#     in_features = Resnet101.fc.in_features
#     Resnet101.fc = nn.Linear(in_features, len(class_names))
#     return Resnet101  # Ensure the model is returned

def Vgg16():
    model = timm.create_model('resnet50', pretrained=True)

    # Adjust Fully Connected Layer (FC Layer)
    num_ftrs = model.fc.in_features  # Number of input features of FC layer
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Change to the number of classes you want (2 classes)

    # Move the model to device
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
        classification_model = Vgg16()  # Make sure Vgg16 or ResNet50 is correctly defined in your code

        # Download model weights from the URL
        url = "https://github.com/fernchunyar/Final_Proj/releases/download/v1.0/USonlyResnet50NoEarlyHaveTest.pth"
        response = requests.get(url)
        
        # Save the model weights to the local file
        with open('USonlyResnet50NoEarlyHaveTest.pth', 'wb') as f:
            f.write(response.content)

        # Load the saved weights into the model
        classification_model.load_state_dict(torch.load('USonlyResnet50NoEarlyHaveTest.pth', map_location=device))

        # Set the model to evaluation mode
        classification_model.eval()

        return classification_model

    except Exception as e:
        st.error(f"Failed to load classification model: {e}")
        return None

def classify_image(image, model):
    try:
        # Ensure the image has three channels (RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Define transformations for the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 as required by the model
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for pretrained models
        ])

        # Apply transformations to the image
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy().squeeze()  # Convert to probabilities

        # Normalize probabilities to ensure they sum to 100%
        probs_percent = probs * 100
        probs_percent = np.round(probs_percent / probs_percent.sum() * 100, 2)

        # Adjust for rounding differences to ensure exact 100%
        diff = 100.00 - np.sum(probs_percent)
        probs_percent[np.argmax(probs_percent)] += diff

        # Map predictions to labels
        labels = {0: "Benign", 1: "Malignant"}
        probabilities = {labels[i]: probs_percent[i] for i in range(len(labels))}
        _, predicted_class = torch.max(output, 1)

        return labels[predicted_class.item()], probabilities

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        return None, None

# Create a guideline page
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
            - The app uses a pre-trained model and may take a few seconds to process your image.
            - For best results, upload clear and high-quality images.
            """
        )

        st.success("You are now ready to proceed to the **Classification** page!")
    
    elif lang_button == "Thai":
        st.markdown(
            """
            แอปนี้จำแนกภาพอัลตราซาวด์เต้านมเป็นสองประเภท:
            **Benign (ไม่เป็นมะเร็ง)** หรือ **Malignant (เป็นมะเร็ง)** โดยมีขั้นตอนการใช้งานแอปดังนี้:
            """
        )
        # Display steps in Thai
        st.markdown(
            """
            ### ขั้นตอน:
            1. ไปที่หน้า **Classification** ผ่านแถบเมนู
            2. อัปโหลดภาพอัลตราซาวด์เต้านมโดยใช้ปุ่ม **Upload Image**
            3. รอให้แอปประมวลผลภาพและแสดงผลลัพธ์:
               - **Classification Result**: แสดงว่าเป็น Benign หรือ Malignant
               - **Predicted Probabilities**: แสดงความน่าจะเป็นของแต่ละประเภท
            4. ทำซ้ำเพื่อจัดประเภทภาพอื่น
            """
        )

        # Add an example image for visual aid
        st.image("intro_img.png", caption="ตัวอย่างผลลัพธ์", use_container_width=True)

        st.markdown(
            """
            ### หมายเหตุ:
            - ตรวจสอบให้แน่ใจว่าภาพเป็นภาพอัลตราซาวด์เต้านมที่ถูกต้องในรูปแบบ `.jpg`, `.jpeg`, หรือ `.png`
            - แอปใช้โมเดลที่ผ่านการฝึกฝนแล้ว อาจใช้เวลาสักครู่ในการประมวลผลภาพของคุณ
            - สำหรับผลลัพธ์ที่ดีที่สุด ควรอัปโหลดภาพที่ชัดเจนและมีคุณภาพสูง
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

    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the uploaded image in the left column
        image = Image.open(uploaded_file)
        classification_model = load_classification_model()

        if classification_model:
            with st.spinner("Processing..."):
                result, probabilities = classify_image(image, classification_model)

                # Two-column layout
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                with col2:
                    if result == "Benign":
                        color = "green"
                    else:
                        color = "red"
                    st.markdown(
                         f"""
                        <h1 style="font-size: 30px;">
                            Classification Result:<strong style="font-size: 30px; margin-bottom: 10px; color:{color};">
                            {result}
                        </strong> 
                        </h1>
                        
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("### Predicted Probabilities:")
                    for label, prob in probabilities.items():
                        st.markdown(f"**{label}:** {prob:.2f}%")
                    # total_prob = sum(probabilities.values())
                    # st.write(f"### Total Probability: {total_prob:.2f}%")

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
