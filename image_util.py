import os
import tensorflow as tf


def read_images_batch(input_queue, batch_size=32, image_size=160, num_threads=1):
    images_and_labels = []
    for _ in range(num_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            image = read_image_from_disk(filename, image_size=image_size)
            images.append(image)
        images_and_labels.append([images, label])

    image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size,
                                                   shapes=[(image_size, image_size, 3), ()], enqueue_many=True,
                                                   capacity=4 * num_threads * batch_size,
                                                   allow_smaller_final_batch=True)
    return image_batch, label_batch


def read_image_from_disk(filename, image_size):
    file_contents = tf.read_file(filename)
    example = tf.image.decode_jpeg(file_contents, channels=3)

    example = tf.image.resize_image_with_crop_or_pad(example, image_size, image_size)

    example.set_shape((image_size, image_size, 3))
    example = tf.image.per_image_standardization(example)
    return example


def get_data(data_dir):
    classes = [f for f in os.listdir(data_dir) if not f.startswith('.')]

    dataset = []
    for cl in classes:
        class_path = os.path.join(data_dir, cl)
        images_paths = [os.path.join(class_path, image_path) for image_path in os.listdir(class_path)]
        dataset.append((cl, images_paths))

    return dataset


def get_image_list_with_labels(dataset):
    image_list, label_list = [], []
    for i in range(len(dataset)):
        image_list += dataset[i][1]
        label_list += [i] * len(dataset[i][1])
    return image_list, label_list