import streamlit as st
import torch
from PIL import Image
from prediction import pred_class  # Ensure this function is defined and works correctly
import numpy as np
from torchvision import transforms

# Set title 
st.title('Weather Prediction')

# Set Header 
st.header('Please upload a picture')

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Use raw string or forward slashes for the file path
model_path = r"C:\Users\ACER PREDATOR\Downloads\mobilenetv3_large_100_checkpoint_fold1.pt"
model = torch.load(model_path, map_location=device)
model.eval()  # Set the model to evaluation mode

# Define the image transformations (if required by your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size required by your model
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize if required
])

# Display image & Prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['Cloudy', 'May_Be_Rain', 'Sunny' ]

    if st.button('Predict'):
        # Apply transformations to the image
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Prediction
        probli = pred_class(model, image_tensor, class_name)  # Ensure pred_class handles tensor input

        st.write("## Prediction Result")
        # Get the index of the maximum value in probli[0]
        max_index = np.argmax(probli[0])

        # Iterate over the class_name and probli lists
        for i in range(len(class_name)):
            # Set the color to blue if it's the maximum value, otherwise use the default color
            color = "blue" if i == max_index else None
            st.write(f"## <span style='color:{color}'>{class_name[i]} : {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)
