import streamlit as st
import torch
from PIL import Image
import numpy as np
from prediction import pred_class
import os


# Set title 
st.title('Weather Prediction')

# Set Header 
st.header('Please upload a picture')

# Load Model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = 'mobilenetv3_large_100_checkpoint_fold0.pt'

if not os.path.isfile(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()  # Stop the Streamlit app if the model file is not found

try:
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set the model to evaluation mode
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the Streamlit app if the model cannot be loaded


    
# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Check the type of image
    if not isinstance(image, Image.Image):
        st.error("Uploaded file is not a valid image.")
        st.stop()
    
    # Prediction button
    if st.button('Prediction'):
        try:
            probli, class_names = pred_class(model, image, class_name)
            st.write("## Prediction Result")
            
            class_name = ['Sunny', 'Cloudy', 'Maybe rain']
            
            max_index = np.argmax(probli.numpy())  # Convert tensor to numpy for argmax
            for i in range(len(class_name)):
                color = "blue" if i == max_index else None
                st.write(f"## <span style='color:{color}'>{class_name[i]} : {probli[i].item()*100:.2f}%</span>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

