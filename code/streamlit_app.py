# streamlit_app.py

import streamlit as st
import numpy as np
import pickle
import faiss
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import random

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants
BATCH_SIZE = 64
NUM_CLASSES = 10
DIM_REDUCTION = 256
N_NEIGHBORS = 5


# Load CIFAR-10 Test Images
@st.cache_data
def load_cifar10_test():
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.flatten()
    return x_test, y_test


# Load Models and Index
@st.cache_resource
def load_models():
    # Load feature extraction model
    feature_model = load_model("C:\\Users\\anjan\\OneDrive\\Desktop\\ML\\Hw3\\code\\feature_model.keras")

    # Load PCA
    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    # Load FAISS index
    index = faiss.read_index("faiss_index.idx")

    # Load normalized training features
    train_features_norm = np.load("train_features_norm.npy")

    return feature_model, pca, index, train_features_norm


# Load CIFAR-10 Training Images
@st.cache_data
def load_cifar10_train():
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("uint8")
    y_train = y_train.flatten()
    return x_train, y_train


# Retrieve similar images using FAISS
def retrieve_similar_images(query_feature, index, top_k=N_NEIGHBORS):
    query_feature = query_feature.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_feature)
    distances, indices = index.search(query_feature, top_k)
    return indices[0], distances[0]


# Display images
def display_images(query_image, retrieved_images, distances, width=None):
    st.image(
        query_image,
        caption="Query Image",
        use_container_width=True if width is None else False,
        width=width,
    )

    cols = st.columns(N_NEIGHBORS)
    for i in range(N_NEIGHBORS):
        with cols[i]:
            st.image(
                retrieved_images[i],
                caption=f"Result {i+1}\nScore: {distances[i]:.4f}",
                use_container_width=True if width is None else False,
                width=width,
            )


# Main Streamlit App
def main():
    st.title("CIFAR-10 Image Retrieval System")
    st.write("""
    This application allows you to retrieve similar images from the CIFAR-10 dataset.
    You can either select a random image from the test set or upload your own image.
    """)

    # Load data and models
    x_test, y_test = load_cifar10_test()
    feature_model, pca, index, train_features_norm = load_models()
    x_train, y_train = load_cifar10_train()

    # Sidebar options
    st.sidebar.header("Options")
    option = st.sidebar.selectbox(
        "Choose an option", ("Select Random Test Image", "Upload an Image")
    )

    if option == "Select Random Test Image":
        # Select a random image from the test set
        query_idx = random.randint(0, x_test.shape[0] - 1)
        query_image = x_test[query_idx]
        st.write(f"Selected Test Image Index: {query_idx}")
    else:
        # Upload an image
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            query_image = query_image.resize((32, 32))
            query_image = np.array(query_image)
        else:
            st.warning("Please upload an image.")
            return

    if "query_image" in locals():
        st.subheader("Query Image")
        st.image(query_image, width=200)

        # Preprocess the query image
        query_image_processed = preprocess_input(
            np.expand_dims(query_image, axis=0).astype("float32")
        )

        # Extract features
        query_feature = feature_model.predict(query_image_processed)

        # Apply PCA
        query_feature_pca = pca.transform(query_feature)

        # Normalize
        query_feature_norm = query_feature_pca / np.linalg.norm(
            query_feature_pca, axis=1, keepdims=True
        )

        # Retrieve similar images
        similar_indices, similar_distances = retrieve_similar_images(
            query_feature_norm, index, top_k=N_NEIGHBORS
        )

        # Get retrieved images
        retrieved_images = x_train[similar_indices]

        st.subheader("Retrieved Similar Images")
        display_images(query_image, retrieved_images, similar_distances)


if __name__ == "__main__":
    main()
