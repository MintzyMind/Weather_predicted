import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

# Set title
st.title('Weather Prediction')

# Set Header
st.header('Please upload a picture')

# Model Path
model_path = 'mobilenetv3_large_100_checkpoint_fold1.pt'

# Check if model file exists
if not os.path.isfile(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Display image & Prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['Cloudy', 'Maybe rain', 'Sunny']
    
    if st.button('Predict'):
        try:
            # Add your prediction logic here
            probli = pred_class(model, image, class_name)
            
            st.write("## Prediction Result")
            max_index = np.argmax(probli[0])
            
            for i in range(len(class_name)):
                color = "blue" if i == max_index else None
                st.write(f"## <span style='color:{color}'>{class_name[i]}: {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
