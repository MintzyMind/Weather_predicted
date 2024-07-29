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
model_path = 'mobilenetv3_large_100_checkpoint_fold0.pt'
if not os.path.isfile(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = torch.load(model_path, map_location=device)
    
# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['Sunny', 'Cloudy', 'Maybe rain']

    if st.button('Predict'):
        # Preprocess the image if required
        # image_tensor = preprocess_image(image) # Example of preprocessing
        
        # Prediction class
        probli = pred_class(model, image, class_name)

        st.write("## Prediction Result")
        
        # Get the index of the maximum value in probli[0]
        max_index = np.argmax(probli[0])

        # Iterate over the class_name and probli lists
        for i in range(len(class_name)):
            # Set the color to blue if it's the maximum value, otherwise use the default color
            color = "blue" if i == max_index else None
            # Use Markdown for styling
            st.markdown(f"### <span style='color:{color}'>{class_name[i]} : {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)
