import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model("model_vgg19.h5")

# Define class labels
class_dict = {
    0: 'Tomato Bacterial spot',
    1: 'Tomato Early blight',
    2: 'Tomato Late blight',
    3: 'Tomato Leaf Mold',
    4: 'Tomato Septoria leaf spot',
    5: 'Tomato Spider mites Two-spotted spider mite',
    6: 'Tomato Target Spot',
    7: 'Tomato Tomato Yellow Leaf Curl Virus',
    8: 'Tomato Tomato mosaic virus',
    9: 'Tomato healthy'
}

# Function to preprocess the image
def prepare_image(image):
    img_array = np.array(image) / 255.0
    img_array = tf.image.resize(img_array, [128, 128])
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_disease(image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return class_dict[predicted_class], confidence

def main():
    # Set page title and background color
    st.set_page_config(page_title="Tomato Disease Prediction", page_icon="🍅", layout="wide", initial_sidebar_state="expanded")

    # Set background color and padding for the entire page
    st.markdown(
        """
        <style>
        body {
            background-color: #f5deb3; /* Cream color */
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set title and header
    st.title("Tomato Disease Prediction")
    st.subheader("Please upload the Tomato leaf image or video frame:")

    # Upload image or video frame
    uploaded_file = st.file_uploader("Upload Image or Video Frame", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image for prediction
        processed_image = prepare_image(image)

        if st.button("Predict"):
            # Make prediction
            with st.spinner('Predicting...'):
                prediction, confidence = predict_disease(processed_image)
                st.success(f"Prediction: {prediction}, Confidence: {confidence:.2f}")

        # Button to clear uploaded image
        if st.button("Clear Image"):
            st.image("", caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()
