import pickle
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st

def get_similar_images(sample_image_path, feature_list_path, filenames_path, num_neighbors=5):
    """
    This function finds similar images to a given sample image using a pre-trained ResNet50 model
    and a k-Nearest Neighbors (KNN) search.

    Args:
        sample_image_path (str): Path to the sample image.
        feature_list_path (str): Path to the pickled list of image features.
        filenames_path (str): Path to the pickled list of corresponding image filenames.
        num_neighbors (int, optional): The number of nearest neighbors to return. Defaults to 5.

    Returns:
        list: A list of filenames of the most similar images to the sample image.
    """

    # Load pre-computed features and filenames
    try:
        with open(feature_list_path, 'rb') as f:
            feature_list = pickle.load(f)
        with open(filenames_path, 'rb') as f:
            filenames = pickle.load(f)
    except FileNotFoundError:
        st.error("Error: Feature list or filenames file not found.")
        return []

    # Create the ResNet50 model for feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freeze the base model (optional, if pre-trained weights are sufficient)
    # base_model.trainable = False

    # Create a sequential model with the frozen ResNet50 and GlobalMaxPooling2D
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    # Load the sample image and preprocess it
    try:
        img = image.load_img(sample_image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
    except FileNotFoundError:
        st.error("Error: Sample image file not found.")
        return []

    # Extract features from the sample image using the model
    result = model.predict(preprocessed_img).flatten()

    # Normalize the extracted features
    normalized_result = result / norm(result)

    # Create and fit the Nearest Neighbors model
    neighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    # Find similar images based on the normalized features
    distances, indices = neighbors.kneighbors([normalized_result])

    # Return filenames of the most similar images
    return [filenames[i] for i in indices[0]]

# Streamlit app
st.title('Fashion Recommender System')
sample_image_path = st.text_input('Enter the path to the sample image:')
if sample_image_path:
    similar_images = get_similar_images(sample_image_path, "embeddings.pkl", "filenames.pkl")
    if similar_images:
        st.write("Similar images:", similar_images)
        for img_path in similar_images:
            st.image(img_path, caption=img_path)
