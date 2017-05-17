import tensorflow as tf
import numpy as np
import math
import time
import os

from image_util import read_images_batch, get_data, get_image_list_with_labels
from model.inception_resnet_v1 import inference
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('data_dir', '', 'Directory with data (photos of faces)')
tf.app.flags.DEFINE_string('train_dir', '/tmp/tensorflow/facerecognition', 'Directory with trained model')

tf.app.flags.DEFINE_string('pretrained_model', None, 'Pretrained model path')

tf.app.flags.DEFINE_integer('epochs', 100, 'Num epochs')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch Size')
tf.app.flags.DEFINE_integer('image_size', 160, 'Image size in pixels (height, width).')

tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_epochs', 100, 'Epochs after which learning rate decays')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'Learning rate decay factor')
tf.app.flags.DEFINE_float('weight_decay', 0.0, 'L2 regularization scale for weights decay')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'Moving average decay')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_float('embedding_size', 128, 'Bottleneck layer size of the model (last fully connected layer)')

FLAGS = tf.app.flags.FLAGS


def main(_):
    dataset = get_data(FLAGS.data_dir)
    num_classes = len(dataset)
    image_list, label_list = get_image_list_with_labels(dataset)
    num_samples = len(image_list)

    print('Total number of classes:', num_classes)
    print('Total number of samples:', num_samples)

    batches_per_epoch = int(math.floor(num_samples / FLAGS.batch_size))

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = tf.Variable(0, trainable=False, name='global_step')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels_placeholder')
        input_queue = tf.FIFOQueue(capacity=100000,
                                   dtypes=[tf.string, tf.int64],
                                   shapes=[(1,), (1,)],
                                   shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
        image_batch, label_batch = read_images_batch(input_queue, FLAGS.batch_size, FLAGS.image_size)

        bottleneck, _ = inference(image_batch, FLAGS.dropout_keep_prob, phase_train=True,
                                  bottleneck_layer_size=FLAGS.embedding_size, weight_decay=FLAGS.weight_decay)
        logits = slim.fully_connected(bottleneck, num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                                      scope='Logits', reuse=False)
        # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        probabilities_op = tf.nn.softmax(logits, name='Predictions')
        predictions_op = tf.argmax(probabilities_op, 1)
        accuracy_op, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions_op, label_batch)

        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step=global_step,
                                                   # change to configuration (should be epoch_size)
                                                   decay_steps=FLAGS.learning_rate_decay_epochs * batches_per_epoch,
                                                   decay_rate=FLAGS.learning_rate_decay_factor,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits,
                                                                       name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # creating train_op
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_average_op = loss_averages.apply(losses + [total_loss])
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + '_raw', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        with tf.control_dependencies([loss_average_op]):
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
            grads = optimizer.compute_gradients(total_loss, tf.global_variables())
        apply_gradients_op = optimizer.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        summary_op = tf.summary.merge_all()

        session_conf = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
        sess = tf.Session(config=session_conf)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if FLAGS.pretrained_model:
                print('Loading pretrained model', FLAGS.pretrained_model)
                saver.restore(sess, FLAGS.pretrained_model)

            images = np.expand_dims(np.array(image_list), 1)
            labels = np.expand_dims(np.array(label_list), 1)

            try:
                for epoch in range(FLAGS.epochs):
                    sess.run(enqueue_op, feed_dict={image_paths_placeholder: images, labels_placeholder: labels})

                    for batch_num in range(batches_per_epoch):
                        start_time = time.time()

                        _, step, loss, reg_loss = sess.run([train_op, global_step, total_loss, regularization_losses])

                        duration = time.time() - start_time
                        print('Epoch {}, Batch {}, Step {}, Loss: {}, Duration {}'.format(epoch, batch_num,
                                                                                          step, loss, duration))

                        if step % 10 == 0:
                            summary_str, predictions = sess.run([summary_op, predictions_op])
                            summary_writer.add_summary(summary_str, step)

                            print('Predictions: ', predictions)
                        if step % 1000 == 0:
                            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=step)
            except Exception:
                coord.request_stop()
            finally:
                coord.request_stop()
                coord.join(threads)

        sess.close()


if __name__ == '__main__':
    tf.app.run()
