import random
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import datetime
import streamlit as st
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import cv2
import time 
from PIL import Image 


#Configuration
CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'epochs': 50,
    'initial_learning_rate': 1e-4,
    'fine_tune_epochs': 10,
    'unfreeze_ratio': 0.75,
    'l1_regularization': 0.002,
    'l2_regularization': 0.002,
    'dropout_rate': 0.2,
    'dense_units': 64,
    'rotation_steps': [0, 1, 3],
    'mixup_alpha': 0.0,
    'early_stopping_patience': 7,
}

#Uses mixed precision for performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

#Focal loss for imbalanced data
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

#Data Loading and Processing
def load_isic_dataset_paths(image_dir: str, metadata_path: str):
    metadata = pd.read_csv(metadata_path)
    st.write(f"Loaded metadata with {len(metadata)} records")

    image_paths, labels = [], []
    for _, row in metadata.iterrows():
        image_id = row['isic_id']
        for ext in ['.jpg', '.jpeg', '.png']:
            p = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(p):
                image_paths.append(p)
                lbl = 1 if str(row['benign_malignant']).lower() == 'malignant' else 0
                labels.append(lbl)
                break
    labels = to_categorical(np.array(labels), num_classes=2)
    return image_paths, labels, metadata

def process_image(file_path: tf.Tensor, label: tf.Tensor):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [CONFIG['image_size'], CONFIG['image_size']])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    k = random.choice(CONFIG['rotation_steps'])
    image = tf.image.rot90(image, k=k)
    return image, label

#Visualization of the dataset
def visualize_samples(dataset, num_samples=5):
    for images, labels in dataset.take(1):
        indices = random.sample(range(images.shape[0]), min(num_samples, images.shape[0]))
        fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
        for i, idx in enumerate(indices):
            axes[i].imshow(images[idx])
            axes[i].set_title('Malignant' if np.argmax(labels[idx]) == 1 else 'Benign')
            axes[i].axis('off')
        st.pyplot(fig)

def plot_class_distribution(metadata):
    dist = metadata['benign_malignant'].str.lower().value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=dist.index, y=dist.values, palette='viridis', ax=ax)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    st.pyplot(fig)

#Initialize the model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

#Function to get the last convolutional layer's name
def get_last_conv_layer_name(model):
    #Loops through the layers in reverse order to find the last convolutional layer
    for layer in reversed(model.layers):
        if 'conv' in layer.name:  #Checks if 'conv' is part of the layer's name
            return layer.name
    return None  #Returns None if no convolutional layer is found

#Calls the function to get the last convolutional layer's name
last_name = get_last_conv_layer_name(model)


# Model cache to prevent the model from reloading or retraining
@st.cache_resource
def get_trained_model(_train_ds, _val_ds, do_train=True):
    if do_train and _train_ds and _val_ds:
        return train_model(_train_ds, _val_ds)

    else:
        model = create_advanced_model()
        try:
            model.load_weights('best_skin_cancer_model.keras')
            model.compile(
                optimizer=optimizers.Adam(learning_rate=CONFIG['initial_learning_rate']),
                loss=focal_loss(),
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            return model
        except Exception as e:
            st.warning("Failed to load pretrained weights. You may need to train the model first.")
            return None



#Model's Architecture
def create_advanced_model() -> tf.keras.Model:
    inputs = layers.Input(shape=(CONFIG['image_size'], CONFIG['image_size'], 3))
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1)
    ])
    x = data_augmentation(inputs)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(CONFIG['image_size'], CONFIG['image_size'], 3)
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        CONFIG['dense_units'], activation='relu',
        kernel_regularizer=regularizers.l1_l2(CONFIG['l1_regularization'], CONFIG['l2_regularization'])
    )(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    outputs = layers.Dense(2, activation='softmax', dtype='float32')(x)

    return models.Model(inputs, outputs)

#Dataset Builder
def build_dataset(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(CONFIG['batch_size'])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

#Training Function
def train_model(train_ds, val_ds):
   
    model = create_advanced_model()

    #Calculates the number of steps per epoch
    steps = tf.data.experimental.cardinality(train_ds).numpy()

    #Learning rate schedule
    boundaries = [steps * 10, steps * 30]
    values = [CONFIG['initial_learning_rate'], CONFIG['initial_learning_rate'] / 10, CONFIG['initial_learning_rate'] / 100]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    #Adam optimizer and model compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    #Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['early_stopping_patience'], restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint('best_skin_cancer_model.keras', monitor='val_loss', save_best_only=True)
    ]

    #Displays training message in Streamlit
    st.info(" Training the model... This may take a while.")
    
    start_time = time.time()
    
    #Starts training
    history = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG['epochs'], callbacks=callbacks, verbose=1)
    
    #Saving the final model after training
    model.save('final_skin_cancer_model.keras')  #Saves the final model after training

    #Notifies the user that the training is done
    st.success(f" Initial training done in {int(time.time() - start_time)} seconds.")

    #Notifies about the model save
    st.success(" Final model has been saved as 'final_skin_cancer_model.keras'.")

    #Fine-tuning model
    base = model.get_layer(index=2)
    base.trainable = True
    total = len(base.layers)
    freeze_until = int((1 - CONFIG['unfreeze_ratio']) * total)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss=focal_loss(), metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    st.info(" Fine-tuning the model...")
    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG['epochs'] + CONFIG['fine_tune_epochs'], initial_epoch=history.epoch[-1] + 1, callbacks=callbacks, verbose=1)
    st.success(" Fine-tuning  is now completed.")


    #Plotting
    if history_fine and history_fine.history:
        acc = history.history.get('accuracy', []) + history_fine.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', []) + history_fine.history.get('val_accuracy', [])
        loss = history.history.get('loss', []) + history_fine.history.get('loss', [])
        val_loss = history.history.get('val_loss', []) + history_fine.history.get('val_loss', [])
    else:
        acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])


    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(acc, label='train_acc'); axs[0].plot(val_acc, label='val_acc'); axs[0].legend(); axs[0].set_title('Accuracy')
    axs[1].plot(loss, label='train_loss'); axs[1].plot(val_loss, label='val_loss'); axs[1].legend(); axs[1].set_title('Loss')
    st.pyplot(fig)

    return model

#Grad - CAM heatmap function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    #Ensures the image is batch dimensioned
    if img_array.ndim == 3:
        img_array = img_array[None, ...]  #Adds the batch dimension
    img_array = tf.cast(img_array, tf.float32)  #Ensurea the image is float32 

    #Gets the last conv layer if not passed
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
    conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    #Converts to numpy and ensure it's 2D for OpenCV resize
    heatmap = np.array(heatmap)
    heatmap = tf.squeeze(heatmap)  #Removes the extra dimensions if needed
    
    #Ensures valid shape and dtype for OpenCV
    if len(heatmap.shape) == 2:
        heatmap = np.uint8(255 * heatmap)  #Normalizes to 0-255 for visualization
        heatmap_resized = cv2.resize(heatmap, (CONFIG['image_size'], CONFIG['image_size']))
    else:
        print("Error: Heatmap is not 2D. Shape of heatmap:", heatmap.shape)
    
    return heatmap_resized


# Streamlit app Interface
def app():
    st.title(" Skin Cancer Classification Diagnostic Tool")
    st.markdown("""
        This application uses a **Machine Learning Model** to classify skin lesion images into **Benign** or **Malignant** categories.
    """)

    IMAGE_DIR = 'ISIC-images'
    META_PATH = 'TrainingMetadata.csv'

    image_paths, labels, metadata = load_isic_dataset_paths(IMAGE_DIR, META_PATH)
    if not image_paths:
        st.error("No images found in the directory. Please check the path.")
        return

    plot_class_distribution(metadata)
    st.write(f" Total images in the dataset: {len(image_paths)}")

    train_p, test_p, train_l, test_l = train_test_split(
        image_paths, labels, test_size=0.2, stratify=np.argmax(labels, axis=1), random_state=42
    )

    train_ds = build_dataset(train_p, train_l, training=True)
    val_ds = build_dataset(test_p, test_l, training=False)

    #Trains or loads the model
    do_train = st.radio("Would you like to train a new model or use a pre-trained one?", ["Use Pre-trained model", "Train New"]) == "Train New"
    model = get_trained_model(train_ds, val_ds, do_train=do_train)

    if model is None:
        return

    #Evaluates the model
    loss, acc, auc = model.evaluate(val_ds)
    st.write(f" Model's Test Loss: `{loss:.4f}`")
    st.write(f" Model's Test Accuracy: `{acc:.4f}`")
    st.write(f" Model's Test AUC: `{auc:.4f}`")

    #Confusion Matrix
    y_true = np.concatenate([y for _, y in val_ds], axis=0)
    y_pred = model.predict(val_ds)
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    


#Grad-CAM Section
st.header("📊 Grad-CAM Visualization")
uploaded = st.file_uploader("Upload a skin lesion image for Grad-CAM to classify the lesions category", type=["jpg", "jpeg", "png"])

#Function to get the confidence score
def get_confidence_score(model, img):
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=-1)
    confidence_score = np.max(predictions)
    return predicted_class, confidence_score

#Streamlit UI
st.title("Skin Cancer Classification")
st.write("Upload a skin lesion image to classify as cancerous or non-cancerous.")

#The function to process the new uploaded image
if uploaded:
    #Load the uploaded image and preprocess
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((CONFIG['image_size'], CONFIG['image_size']))
    img_array = np.array(img_resized) / 255.0  # Normalize the image to [0, 1]
    
    #Ensure the image is of shape [1, H, W, C] to match model input
    img_array = np.expand_dims(img_array, axis=0)
    
    #Converts to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    #Displays the uploaded image
    st.image(img_resized, caption="Uploaded image", use_container_width=True)
    
    #Gets the last convolutional layer name
    last_name = get_last_conv_layer_name(model)

    #Generates the Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name=last_name)

    #Resizes and colorize the heatmap for visualization
    heatmap_resized = cv2.resize(heatmap, (CONFIG['image_size'], CONFIG['image_size']))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    #Uses OpenCV to apply a color map (Jet) to the heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    #Ensures both images are numpy arrays
    heatmap_color = np.array(heatmap_color)
    img_resized = np.array(img_resized)

    #Checks and resize heatmap if necessary
    if heatmap_color.shape != img_resized.shape:
        print("Resizing heatmap to match the image shape...")
        heatmap_color = cv2.resize(heatmap_color, (img_resized.shape[1], img_resized.shape[0]))

  
    heatmap_color = np.uint8(255 * heatmap_color) #Scales if the values are between 0 and 1
    img_resized = np.uint8(img_resized)

    #Uses addWeighted for blending
    superimposed_img = cv2.addWeighted(heatmap_color, 0.4, img_resized, 0.6, 0)

    #Displays the combined image with heatmap overlay
    st.image(superimposed_img, caption="Grad-CAM Overlay", use_container_width=True)

    #Shows the Grad-CAM heatmap alone
    st.image(heatmap_color, caption="Grad-CAM Heatmap", use_container_width=True)

    #Predicts the uploaded image class
    predictions = model.predict(img_tensor)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_label = 'Malignant' if predicted_class == 1 else 'Benign'
    st.write(f"**Predicted Class:** {class_label}")

    
    st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)

    #Show the heatmap itself
    st.image(heatmap_color, caption="Heatmap", use_container_width=True)


if __name__ == '__main__':
    app()