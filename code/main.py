# main.py

import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
import faiss

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


BATCH_SIZE = 64
NUM_CLASSES = 10
DIM_REDUCTION = 256
N_NEIGHBORS = 5


print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")


print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
)


print("Building ResNet50 model for feature extraction...")
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
feature_model = Model(inputs=base_model.input, outputs=x)


for layer in base_model.layers:
    layer.trainable = False


feature_model.compile(
    optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy"
)


print("Preparing training data with augmentation...")
train_generator = datagen.flow(
    x_train,
    tf.keras.utils.to_categorical(y_train, NUM_CLASSES),
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
)


print("Extracting features from training data...")
train_features = feature_model.predict(
    train_generator, steps=int(np.ceil(x_train.shape[0] / BATCH_SIZE)), verbose=1
)
train_labels = y_train


print("Extracting features from test data...")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow(
    x_test,
    tf.keras.utils.to_categorical(y_test, NUM_CLASSES),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_features = feature_model.predict(
    test_generator, steps=int(np.ceil(x_test.shape[0] / BATCH_SIZE)), verbose=1
)
test_labels = y_test


print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=DIM_REDUCTION, random_state=SEED)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)


print("Normalizing feature vectors...")
train_features_norm = train_features_pca / np.linalg.norm(
    train_features_pca, axis=1, keepdims=True
)
test_features_norm = test_features_pca / np.linalg.norm(
    test_features_pca, axis=1, keepdims=True
)


with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)


print("Building FAISS index for efficient similarity search...")
index = faiss.IndexFlatIP(DIM_REDUCTION)
faiss.normalize_L2(train_features_norm)
index.add(train_features_norm)
print(f"FAISS index contains {index.ntotal} vectors.")


faiss.write_index(index, "faiss_index.idx")


def retrieve_similar_images(query_feature, index, top_k=N_NEIGHBORS):
    query_feature = query_feature.reshape(1, -1)
    faiss.normalize_L2(query_feature)
    distances, indices = index.search(query_feature, top_k)
    return indices[0], distances[0]


def calculate_average_precision(relevant_scores):
    """Calculate Average Precision for a single query"""
    if not any(relevant_scores):
        return 0.0

    running_sum = 0.0
    num_relevant = 0

    for i, is_relevant in enumerate(relevant_scores, 1):
        if is_relevant:
            num_relevant += 1
            running_sum += num_relevant / i

    return running_sum / num_relevant if num_relevant > 0 else 0.0


def evaluate_retrieval(
    test_features, test_labels, train_labels, index, top_k=N_NEIGHBORS
):
    precisions = []
    recalls = []
    aps = []

    for i in tqdm(range(test_features.shape[0]), desc="Evaluating retrieval"):
        query = test_features[i]
        query_label = test_labels[i]
        indices, _ = retrieve_similar_images(query, index, top_k)
        retrieved_labels = train_labels[indices]

        relevant = retrieved_labels == query_label

        relevant_items_total = np.sum(train_labels == query_label)

        if relevant_items_total > 0:
            precision = np.sum(relevant) / len(relevant)
            recall = np.sum(relevant) / relevant_items_total
            ap = calculate_average_precision(relevant)

            precisions.append(precision)
            recalls.append(recall)
            aps.append(ap)

    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    mean_ap = np.mean(aps) if aps else 0.0

    return mean_precision, mean_recall, mean_ap


def display_retrieval(query_idx, indices, distances, x_train, x_test):
    plt.figure(figsize=(15, 3))

    plt.subplot(1, N_NEIGHBORS + 1, 1)
    plt.imshow(x_test[query_idx])
    plt.title("Query Image")
    plt.axis("off")

    for i, (idx, dist) in enumerate(zip(indices, distances)):
        plt.subplot(1, N_NEIGHBORS + 1, i + 2)
        plt.imshow(x_train[idx])
        plt.title(f"Result {i+1}\nScore: {dist:.4f}")
        plt.axis("off")

    plt.savefig("retrieval_results.png")
    plt.close()


print("\nEvaluating retrieval performance...")
mean_precision, mean_recall, mean_ap = evaluate_retrieval(
    test_features_norm, test_labels, train_labels, index, top_k=N_NEIGHBORS
)

print("\n=== Final Evaluation Metrics ===")
print(f"Mean Precision@{N_NEIGHBORS}: {mean_precision:.4f}")
print(f"Mean Recall@{N_NEIGHBORS}: {mean_recall:.4f}")
print(f"Mean Average Precision: {mean_ap:.4f}")
print(
    f"Accuracy: {mean_precision:.4f}"
)  # In this case, precision can be interpreted as accuracy
print(
    f"F1-Score: {2 * (mean_precision * mean_recall) / (mean_precision + mean_recall):.4f}"
)
print("================================\n")

print("Performing example retrieval...")
example_idx = random.randint(0, x_test.shape[0] - 1)
query_feature = test_features_norm[example_idx]
similar_indices, similar_distances = retrieve_similar_images(
    query_feature, index, top_k=N_NEIGHBORS
)
display_retrieval(example_idx, similar_indices, similar_distances, x_train, x_test)


print("Saving models and indices...")

feature_model.save("feature_model.keras")


np.save("train_features_norm.npy", train_features_norm)
np.save("test_features_norm.npy", test_features_norm)

print("All components saved successfully.")
