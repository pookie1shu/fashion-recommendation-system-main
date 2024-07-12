import streamlit as st
import os
from PIL import Image
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load pre-trained features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit application title
st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try: 
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        columns = [col1, col2, col3, col4, col5]
        
        for col, index in zip(columns, indices[0][:5]):
            image_path = filenames[index]
            try:
                # Debugging: Print the image path
                print(f"Trying to load image: {image_path}")
                if os.path.exists(image_path):
                    col.image(image_path)
                else:
                    col.text(f"File does not exist: {image_path}")
            except Exception as e:
                col.text(f"Error loading image: {e}")
    else:
        st.header("Some error occurred in file upload")
