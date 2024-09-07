import os
import numpy as np
import tensorflow as tf

# Set a seed value for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Define class names for the dataset
class_names = ['covid', 'normal', 'pneumonia']

def load_filenames(dataset_dir):
    '''Load the filenames and their corresponding label indices from a specified dataset directory.'''

    filenames = []
    label_idxs = []

    i = 0
    for lab in os.listdir(dataset_dir):
        current_dir = os.path.join(dataset_dir, lab)
        for img in os.listdir(current_dir):
            filenames.append(os.path.join(current_dir, img))
            label_idxs.append(i)
        i += 1

    print('Loading filenames completed.')

    return filenames, label_idxs

def preprocess(image):
    '''Preprocess an image applying normalization, cropping and resizing'''
    
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [224, 224])
    image = tf.image.crop_to_bounding_box(image, offset_height=int(224*0.1), offset_width=0, target_height=int(224*0.8), target_width=int(224))
    image = tf.image.resize(image, [224, 224])
    return image


# Set up a data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode='constant'),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant'),
    tf.keras.layers.RandomRotation(factor=0.1, fill_mode='constant'),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomContrast(factor=0.1)
])

def get_label(file_path):
    '''Parse the label from the file path'''

    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)

def decode_img(img):
    '''
    Decode the image and convert it into a Tensor using tf.io.decode_image. 
    It supports multiple image formats including is a BMP, GIF, JPEG, and PNG.
    '''

    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_image(img, channels=3)
    # Ensure the image has shape [height, width, channels]
    img = tf.ensure_shape(img, [None, None, 3])

    return img

def load_image_with_label(file_path):
    '''Load an image and its label given the file path'''

    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    
    return img, label

def create_dataset(file_names, batch_size, shuffle, augment=True, cache_file=None):
    '''
    Constructs a TensorFlow Dataset object configured for image processing.
    
    This function prepares a dataset by loading images, applying preprocessing,
    and optionally augmenting, shuffling, batching, and prefetching the data.
    The dataset can be cached to speed up operations.
    '''

    # Create a Dataset object
    dataset = tf.data.Dataset.from_tensor_slices(file_names)

    # Map the load_image function
    dataset = dataset.map(load_image_with_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Map the preprocess function
    dataset = dataset.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # Map the augment_image function
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(100)

    # Repeat the dataset indefinitely
    dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Dataset correctly created.")

    return dataset