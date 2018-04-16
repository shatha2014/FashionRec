# Author: Kim Hammar <kimham@kth.se> KTH 2018
#
# Original paper for the CNN model:
# @inproceedings{kimyoon_cnn,
#                author    = {Yoon Kim},
# title     = {Convolutional Neural Networks for Sentence Classification},
# booktitle = {{EMNLP}},
# pages     = {1746--1751},
# publisher = {{ACL}},
# year      = 2014
# }
#
# Tensorflow implementation inspiration from Denny Britz:
# https://github.com/dennybritz/cnn-text-classification-tf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pre_process
import argparse
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from tensorflow.python.tools import freeze_graph

"""
Script for training a text classifier with Kim Yoon's CNN, either with pre-trained or random initialized embeddings. 
Can use either noisy or binary labels and either a single input channel or multi-channel.
"""


# Enable deterministic comparisons between executions
tf.set_random_seed(0)
# Constants
NUM_CLASSES = 13
NUM_CHANNELS = 2

def define_placeholders(sequence_length, multichannel=False):
    """ Define placeholders for input features,labels, and dropout """

    if(multichannel):
        x_placeholder = tf.placeholder(tf.int32, [None, sequence_length, NUM_CHANNELS], name="input_x")
    else:
        x_placeholder = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")

    y_placeholder = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="input_y")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return x_placeholder, y_placeholder, dropout_keep_prob


def build_graph(x_placeholder, vocab_size, embedding_size, dropout_placeholder, sequence_length, filter_sizes,
                num_filters, initW, pretrained=False, multichannel=False):
    """ Build the computational graph for forward and backward propagation """

    # Keeping track of l2 regularization loss
    l2_loss = tf.constant(0.0)

    # Embedding layer
    with tf.name_scope("embedding"):
        if(pretrained):
            W = initW
        else:
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
        if(multichannel):
            #Lookup word-ids in the embedding matrix
            embedded_chars = tf.nn.embedding_lookup(W, x_placeholder)
            #Transpose to get correct format
            embedded_chars_expanded = tf.transpose(embedded_chars, [0,1,3,2])
        else:
            #Lookup word-ids in the embedding matrix
            embedded_chars = tf.nn.embedding_lookup(W, x_placeholder)
            #CNN expects 3D input, expand to be 1 channel so it fits
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            if(multichannel):
                filter_shape = [filter_size, embedding_size, NUM_CHANNELS, num_filters]
            else:
                filter_shape = [filter_size, embedding_size, 1, num_filters]
            #Initialize weights randomly
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            #Initialize bias
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            #Convolution operation, 2D convolution (patch strides over 2d surface for all input channels one at a time) on 4D input
            #VALID padding => No padding, means output width =  (width-filter-width +1)/stride
            #strides = [1,1,1,1], one stride for each dimension
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # Apply RELU nonlinearity to the output of conv operation added with the bias
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs of RELU
            # ksize is the dimensions of patch
            # the patch is slided over the input and outputs the max element of each region
            # (intuitively sub-sample the input by focusing on keywords and dropping noise)
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name="pool")
            # Since we have one pooling for each conv channel we store all outputs (multi dimensional) in an array
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    # append pooled features on last axis
    h_pool = tf.concat(pooled_outputs, 3)
    # flatten output
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_placeholder)

    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
        # Weights between pooled features and output, uses "Xavier" initialization from paper "Understanding the difficulty of training deep feedforward neural networks"
        W = tf.get_variable(
            "W",
            shape=[num_filters_total, NUM_CLASSES],
            initializer=tf.contrib.layers.xavier_initializer())
        # initialize bias
        b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b")
        # l2 loss
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        # h_drop x weights + b
        logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        # cast logits to binary predictions
        predictions = tf.where(logits > 0.5, tf.ones_like(logits), tf.zeros_like(logits), name="predictions")
    return logits, predictions, l2_loss


def define_optimizer(learning_rate, logits, y_placeholder, l2_loss, predictions, l2_reg_lambda):
    """ Define the optimizer, loss, accuracy etc for doing the learning """

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        # Binary logistic loss for each class (works with both probabilistic labels and binary labels)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_placeholder,
                                                         name="losses")
        # Sum the log-loss for each class and add l2 regularization
        loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        tf.summary.scalar("regularized_loss", loss)

    # When using probabilistic labels this casting is necessary to get binary labels for computing statistics
    y_preds = tf.where(y_placeholder > 0.5, tf.ones_like(y_placeholder), tf.zeros_like(y_placeholder))
    # Compare labels with predictions
    correct_predictions = tf.equal(tf.cast(predictions, dtype=tf.int32), tf.cast(y_preds, dtype=tf.int32))

    # Compute stats and update tensorboard
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("streaming_acc"):
        streaming_accuracy, str_acc_update = tf.metrics.accuracy(labels=y_preds, predictions=predictions)
        tf.summary.scalar("streaming_ accuracy", str_acc_update)

    with tf.name_scope('recall'):
        recall, rec_update = tf.metrics.recall(labels=y_preds, predictions=predictions)
        tf.summary.scalar("recall", rec_update)

    with tf.name_scope('precision'):
        precision, pre_update = tf.metrics.precision(labels=y_preds, predictions=predictions)
        tf.summary.scalar("precision", precision)

    with tf.name_scope('F1'):
        F1 = (2 * pre_update * rec_update) / (pre_update + rec_update)
        tf.summary.scalar("F1", F1)

    TP = tf.count_nonzero(tf.cast(predictions, dtype=tf.int32) * tf.cast(y_preds, dtype=tf.int32), dtype=tf.float32)
    TN = tf.count_nonzero((tf.cast(predictions, dtype=tf.int32) - 1) * (tf.cast(y_preds, dtype=tf.int32) - 1),dtype=tf.float32)
    FP = tf.count_nonzero(tf.cast(predictions, dtype=tf.int32) * (tf.cast(y_preds, dtype=tf.int32) - 1),dtype=tf.float32)
    FN = tf.count_nonzero((tf.cast(predictions, dtype=tf.int32) - 1) * tf.cast(y_preds, dtype=tf.int32), dtype=tf.float32)
    batch_precision = TP / (TP + FP)
    batch_recall = TP / (TP + FN)
    batch_f1 = 2 * ((batch_precision * batch_recall) / (batch_precision + batch_recall))
    tf.summary.scalar("batch_precision", batch_precision)
    tf.summary.scalar("batch_recall", batch_recall)
    tf.summary.scalar("batch_f1", batch_f1)

    # Define Training procedure

    # Uncomment this if using exp decay
    # global_step = tf.Variable(0, name="global_step", trainable=False)
    # starter_learning_rate = learning_rate
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.96, staircase=True)

    tf.summary.scalar("learning rate", learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return train_step, accuracy, loss, recall, rec_update, precision, pre_update, F1, streaming_accuracy, str_acc_update, batch_precision, batch_recall, batch_f1, y_preds


def init_graph():
    """ Initialize the graph and variables for Tensorflow engine """

    # initialize and run start operation
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess = tf.Session()
    sess.run(init_g)
    sess.run(init_l)
    return sess


def training_step(i, update_dev_data, update_train_data, update_test_data, x_placeholder, y_placeholder,
                  dropout_placeholder,
                  x_train_batch, y_train_batch, x_dev_batch, y_dev_batch, x_test, y_test,
                  dropout_keep_prob, training_step, accuracy, loss, sess,
                  predictions, train_writer, test_writer,
                  merged, recall, rec_update, precision, pre_update, F1, streaming_accuracy, str_acc_update, batch_precision,
                  batch_recall, batch_f1, logits, y_preds, verbose, multichannel):
    """
    Function representing a single iteration during training.
    Returns a tuple of accuracy and loss statistics.
    """
    if(multichannel):
        x_train_batch = np.transpose(x_train_batch,axes=[0,2,1])
        x_dev_batch = np.transpose(x_dev_batch,axes=[0,2,1])
        x_test = np.transpose(x_test,axes=[0,2,1])

    # the backpropagation training step
    sess.run(training_step,
             feed_dict={x_placeholder: x_train_batch, y_placeholder: y_train_batch,
                        dropout_placeholder: dropout_keep_prob})

    # evaluating model performance for printing purposes
    # evaluation used to later visualize how well the model did at a particular time in the training
    train_a = []  # Array of training-accuracy for a single iteration
    train_str_a = []  # Array of streaming training-accuracy
    train_c = []  # Array of training-cost for a single iteration
    train_r = []  # Array of streaming training-recall
    train_p = []  # Array of streaming training-precision
    train_f = []  # Array of streaming training-F1
    train_hl = []  # Array of train hamming loss
    dev_a = []  # Array of dev-accuracy for a single iteration
    dev_c = []  # Array of dev-cost for a single iteration
    dev_r = []  # Array of dev-recall for a single iteration
    dev_p = []  # Array of dev-precision for a single iteration
    dev_f = []  # Array of dev-F1 for a single iteration
    dev_hl = []  # Array of dev hamming loss
    test_a = []  # Array of test-accuracy for a single iteration
    test_c = []  # Array of test-cost for a single iteration
    test_r = []  # Array of test-recall for a single iteration
    test_p = []  # Array of test-precision for a single iteration
    test_f = []  # Array of test-F1 for a single iteration
    test_hl = []  # Array of test-hamming loss
    test_class_precision = [] #Array of precision for each class
    test_class_recall = [] #Array of precision for each class
    test_class_ap = [] #Array of avg precision for each class
    test_class_f1 = [] #Array of f1 for each class
    test_m_f = [] #Array of macro f1 for test set

    # Compute streaming recall, precision, accuracy on train set
    train_recall, train_precision, train_str_acc = sess.run([rec_update, pre_update, str_acc_update],
                                                            feed_dict={x_placeholder: x_train_batch,
                                                                       y_placeholder: y_train_batch,
                                                                       dropout_placeholder: dropout_keep_prob})
    # If stats for train-data should be updated, compute loss and accuracy for the batch and store it
    if update_train_data:
        train_acc, train_loss, train_preds, train_logits, summary, train_f1, y_tr_pred = sess.run([accuracy, loss, predictions, logits, merged, F1, y_preds],
                                                                         feed_dict={x_placeholder: x_train_batch,
                                                                                    y_placeholder: y_train_batch,
                                                                                    dropout_placeholder: dropout_keep_prob})
        train_hls = hamming_loss(train_preds, y_tr_pred)
        tf.summary.scalar("hamming_loss",train_hls)
        train_writer.add_summary(summary, i)
        train_a.append(train_acc)
        train_c.append(train_loss)
        train_r.append(train_recall)
        train_p.append(train_precision)
        train_f.append(train_f1)
        train_str_a.append(train_str_acc)
        train_hl.append(train_hls)
        if(verbose):
            print("train loss: {}".format(train_loss))
            print("train batch accuracy: {}".format(train_acc))
            print("train recall: {}".format(train_recall))
            print("train precision: {}".format(train_precision))
            print("train micro-averaged f1: {}".format(train_f1))
            print("train streaming accuracy: {}".format(train_str_acc))
            print("train hamming loss: {}".format(train_hls))

    # If stats for dev-data should be updated, compute loss and accuracy for the batch and store it
    if update_dev_data:
        dev_acc, dev_loss, dev_preds, dev_logits, summary, dev_recall, dev_precision, dev_f1, y_d_pred = sess.run(
            [accuracy, loss, predictions, logits, merged, batch_recall, batch_precision, batch_f1, y_preds],
            feed_dict={x_placeholder: x_dev_batch,
                       y_placeholder: y_dev_batch,
                       dropout_placeholder: 1.0})
        dev_hls = hamming_loss(dev_preds, y_d_pred)
        tf.summary.scalar("hamming_loss",dev_hls)
        dev_a.append(dev_acc)
        dev_c.append(dev_loss)
        dev_r.append(dev_recall)
        dev_p.append(dev_precision)
        dev_f.append(dev_f1)
        dev_hl.append(dev_hls)
        test_writer.add_summary(summary, i)
        if(verbose):
            print("dev loss: {}".format(dev_loss))
            print("dev accuracy: {}".format(dev_acc))
            print("dev recall: {}".format(dev_recall))
            print("dev precision: {}".format(dev_precision))
            print("dev micro-averaged f1: {}".format(dev_f1))
            print("dev hamming loss: {}".format(dev_hls))

    # At the end of training, test on the held-out ground truth testset
    if update_test_data:
        test_acc, test_loss, test_preds, test_logits, test_recall, test_precision, test_f1, y_t_pred= sess.run([accuracy, loss, predictions, logits, batch_recall, batch_precision, batch_f1, y_preds],
                                                                                        feed_dict={x_placeholder: x_test,
                                                                                                   y_placeholder: y_test,
                                                                                                   dropout_placeholder: 1.0})
        test_hls = hamming_loss(test_preds, y_t_pred)
        test_macro_f1 = f1_score(y_test, test_preds, average='macro')
        test_a.append(test_acc)
        test_c.append(test_loss)
        test_r.append(test_recall)
        test_p.append(test_precision)
        test_f.append(test_f1)
        test_hl.append(test_hls)
        test_m_f.append(test_macro_f1)
        if(verbose):
            print("test loss: {}".format(test_loss))
            print("test accuracy: {}".format(test_acc))
            print("test recall: {}".format(test_recall))
            print("test precision: {}".format(test_precision))
            print("test micro-averaged f1: {}".format(test_f1))
            print("macro averaged f1: {}".format(test_macro_f1))
            print("test hamming loss: {}".format(test_hls))

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(NUM_CLASSES):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                test_preds[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], test_preds[:, i])

        f1_per_class = f1_score(y_test, test_preds, average=None)

        test_class_precision.append(" ".join(map(lambda x: str(x), precision.values())))
        test_class_recall.append(" ".join(map(lambda x: str(x), recall.values())))
        test_class_ap.append(" ".join(map(lambda x: str(x), average_precision.values())))
        test_class_f1.append(" ".join(map(lambda x: str(x), f1_per_class)))

        if(verbose):
            for i in range(NUM_CLASSES):
                print("precision for class {}: {}".format(i, precision[i]))
                print("recall for class {}: {}".format(i, recall[i]))
                print("average_precision for class {}: {}".format(i, average_precision[i]))
                print("f1 for class {}: {}".format(i, f1_per_class[i]))

    return train_a, train_c, train_r, train_p, train_f, train_str_acc, train_hl, dev_a, dev_c, dev_r, dev_p, dev_f, dev_hl, test_a, test_c, test_r, test_p, test_f, test_hl, test_class_precision, test_class_recall, test_class_ap, test_m_f


def hype_grid(args):
    """ Grid search for hyperparameter tuning """
    learning_rates = [0.0001, 0.001, 0.003, 0.05, 0.03]
    dropout_rates = [0.1, 0.2, 0.3, 0.6, 0.8]
    l2reglambda_rates = [0.1,0.2,0.3,0.6,0.8]
    if(args.pretrained):
        # Preprocess data to the right format
        x, vocab_processor, y, x_test, y_test, W = pre_process.combine_labels_features(
            args.featurestrain,
            args.labelstrain,
            args.featurestest,
            args.labelstest,
            args.vectors,
            args.vectordim,
            args.maxdocumentsize
        )
    else:
        # Preprocess data to the right format
        x, vocab_processor, y, x_test, y_test = pre_process.combine_labels_features(
            args.featurestrain,
            args.labelstrain,
            args.featurestest,
            args.labelstest,
            "",
            args.vectordim,
            args.maxdocumentsize,
        )
    x_train, x_dev, y_train, y_dev = pre_process.split(x, y, args.testsplit, vocab_processor.vocabulary_)
    sequence_length = x_train.shape[1]
    results = []
    results.append("learning_rate,dropout_rate,l2reglamda,test_accuracy")
    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for l2lambda in l2reglambda_rates:
                #To be able to run hyperparam tuning on the same graph
                tf.reset_default_graph()
                print(
                    "Trying following " + str(learning_rate) + " learning rate and " + str(dropout_rate) + ' dropout rate and l2reglambda: ' + str(l2lambda))
                if(args.pretrained):
                    test_accuracy = main(sequence_length, len(vocab_processor.vocabulary_), x_train, y_train, x_dev, y_dev,
                                         x_test, y_test, W=W, pretrained=True, num_epochs=args.epochs, batch_size=args.batchsize, vectorDim=args.vectordim,
                                         learning_rate=args.learningrate, dropout_keep_prob=args.dropoutkeepprob, filter_sizes=args.filtersizes,
                                         num_filters=args.numfilters,
                                         l2_reg_lambda=args.l2reglambda)
                else:
                    test_accuracy = main(sequence_length, len(vocab_processor.vocabulary_), x_train, y_train, x_dev, y_dev,
                                         x_test, y_test, num_epochs=args.epochs, batch_size=args.batchsize, vectorDim=args.vectordim,
                                         learning_rate=args.learningrate, dropout_keep_prob=args.dropoutkeepprob, filter_sizes=args.filtersizes,
                                         num_filters=args.numfilters,
                                         l2_reg_lambda=args.l2reglambda, maxiterations=args.maxiterations)
                print('Test accuracy ' + str(test_accuracy))
                results.append(str(learning_rate) + "," + str(dropout_rate) + "," + str(dropout_rate) + "," + str(l2lambda) + "," + str(test_accuracy))
    np.savetxt('./results/tuning/tuning.txt', np.array(results), delimiter=',', fmt="%s")


def hype_random(args):
    """
    Random search for hyperparameter tuning
    """
    learning_rates = np.random.uniform(0.0001, 0.03, 10).tolist()
    dropout_rates = np.random.uniform(0.1, 0.8, 1).tolist()
    l2reglambda_rates = np.random.uniform(0.1, 0.7, 10).tolist()
    if(args.pretrained):
        # Preprocess data to the right format
        x, vocab_processor, y, x_test, y_test, W = pre_process.combine_labels_features(
            args.featurestrain,
            args.labelstrain,
            args.featurestest,
            args.labelstest,
            args.vectors,
            args.vectordim,
            args.maxdocumentsize
        )
    else:
        # Preprocess data to the right format
        x, vocab_processor, y, x_test, y_test = pre_process.combine_labels_features(
            args.featurestrain,
            args.labelstrain,
            args.featurestest,
            args.labelstest,
            "",
            args.vectordim,
            args.maxdocumentsize,
        )
    x_train, x_dev, y_train, y_dev = pre_process.split(x, y, args.testsplit, vocab_processor.vocabulary_)
    sequence_length = x_train.shape[1]
    results = []
    results.append("learning_rate,dropout_rate,l2reglamda,test_accuracy")
    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for l2lambda in l2reglambda_rates:
                #To be able to run hyperparam tuning on the same graph
                tf.reset_default_graph()
                print(
                        "Trying following " + str(learning_rate) + " learning rate and " + str(dropout_rate) + ' dropout rate and l2reglambda: ' + str(l2lambda))
                if(args.pretrained):
                    test_accuracy = main(sequence_length, len(vocab_processor.vocabulary_), x_train, y_train, x_dev, y_dev,
                                         x_test, y_test, W=W, pretrained=True, num_epochs=args.epochs, batch_size=args.batchsize, vectorDim=args.vectordim,
                                         learning_rate=args.learningrate, dropout_keep_prob=args.dropoutkeepprob, filter_sizes=args.filtersizes,
                                         num_filters=args.numfilters,
                                         l2_reg_lambda=args.l2reglambda)
                else:
                    test_accuracy = main(sequence_length, len(vocab_processor.vocabulary_), x_train, y_train, x_dev, y_dev,
                                         x_test, y_test, num_epochs=args.epochs, batch_size=args.batchsize, vectorDim=args.vectordim,
                                         learning_rate=args.learningrate, dropout_keep_prob=args.dropoutkeepprob, filter_sizes=args.filtersizes,
                                         num_filters=args.numfilters,
                                         l2_reg_lambda=args.l2reglambda, maxiterations=args.maxiterations)
                print('Test accuracy ' + str(test_accuracy))
                results.append(str(learning_rate) + "," + str(dropout_rate) + "," + str(dropout_rate) + "," + str(l2lambda) + "," + str(test_accuracy))
    np.savetxt('./results/tuning/tuning.txt', np.array(results), delimiter=',', fmt="%s")


def main(sequence_length, vocabSize, x_train, y_train, x_dev, y_dev, x_test, y_test, W = [], pretrained = False,
         vectorDim=300, learning_rate=0.01, dropout_keep_prob=0.7,
         batch_size=64, num_epochs=100, filter_sizes=[3, 4, 5], num_filters=128, l2_reg_lambda=0.0,
         output="./results", maxiterations=100000000000000, verbose=False, plot=False, multichannel=False):
    """
    Orchestrates the training, initiates and builds graph, performs training, saves results
    """

    # Containers for results
    train_accuracy = []
    train_streaming_accuracy = []
    train_loss = []
    train_recall = []
    train_precision = []
    train_f1 = []
    train_hl = []
    dev_accuracy = []
    dev_loss = []
    dev_recall = []
    dev_precision = []
    dev_f1 = []
    dev_hl = []
    test_accuracy = []
    test_loss = []
    test_recall = []
    test_precision = []
    test_f1 = []
    test_hl = []
    test_class_precision = []
    test_class_recall = []
    test_class_ap = []
    test_m_f = []
    # Get placeholders
    x_placeholder, y_placeholder, dropout_placeholder = define_placeholders(sequence_length, multichannel=multichannel)

    # Build graph and get necessary variables
    if(pretrained):
        logits, predictions, l2_loss = build_graph(x_placeholder, vocabSize, vectorDim, dropout_placeholder,
                                                   sequence_length, filter_sizes, num_filters, W, pretrained=True, multichannel=multichannel)
    else:
        logits, predictions, l2_loss = build_graph(x_placeholder, vocabSize, vectorDim, dropout_placeholder,
                                                   sequence_length, filter_sizes, num_filters, W, pretrained=False, multichannel=multichannel)

    # Define optimizer and get reference to operations to train with
    training_step_tf, accuracy, cross_entropy_loss, recall, rec_update, precision, \
    pre_update, F1, streaming_accuracy, str_acc_update, batch_precision, batch_recall, batch_f1, y_preds = define_optimizer(
        learning_rate, logits, y_placeholder, l2_loss,
        predictions, l2_reg_lambda)

    # Initialize TF
    sess = init_graph()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Merge all the summaries and write them to tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(output + "/tensorboard" + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(output + "/tensorboard" + '/test')


    # Train NumEpochs over the entire train set
    # Each iteration uses BatchSize number of examples
    epoch = 1
    total_i = 0
    while epoch <= num_epochs:
        i = 0
        # Generate batches
        batches = pre_process.batch_iter(
        list(zip(x_train, y_train)), batch_size, num_epochs)
        for batch in batches:
            if(total_i < maxiterations):
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(list(x_batch))
                y_batch = np.array(list(y_batch))
                train_test = False
                dev_test = False
                test_test = False
                if i % 10 == 0:
                    # save checkpoint
                    saver.save(sess, output + "/models/model.ckpt")
                    train_test = True
                    dev_test = True
                    test_test = False
                    if(len(dev_f1) > 0):
                        print("#########     epoch: {} , iteration: {}, dev f1: {}   ##########".format(epoch, i, dev_f1[-1]))
                    else:
                        print("#########     epoch: {} , iteration: {}, dev f1: -1     ##########".format(epoch, i))
                a, c, r, p, f, stra, hl, da, dc, dr, dp, df1, dhl, ta, tc, tr, tp, tf1, thl, tcp, tcr, tca, tmf = training_step(i * epoch, dev_test,
                                                                                                            train_test, test_test,
                                                                                                            x_placeholder, y_placeholder,
                                                                                                            dropout_placeholder, x_batch,
                                                                                                            y_batch,
                                                                                                            x_dev, y_dev, x_test, y_test,
                                                                                                            dropout_keep_prob,
                                                                                                            training_step_tf, accuracy,
                                                                                                            cross_entropy_loss, sess,
                                                                                                            predictions, train_writer,
                                                                                                            test_writer, merged, recall,
                                                                                                            rec_update, precision,
                                                                                                            pre_update, F1,
                                                                                                            streaming_accuracy,
                                                                                                            str_acc_update, batch_precision,
                                                                                                            batch_recall, batch_f1, logits, y_preds, verbose, multichannel)
                # Update training stats
                train_accuracy += a
                train_streaming_accuracy += stra
                train_loss += c
                train_recall += r
                train_precision += p
                train_f1 += f
                train_hl += hl
                dev_accuracy += da
                dev_loss += dc
                dev_recall += dr
                dev_precision += dp
                dev_f1 += df1
                dev_hl += dhl
                test_accuracy += ta
                test_loss += tc
                test_recall += tr
                test_precision += tp
                test_f1 += tf1
                test_hl += thl
                test_class_precision += tcp
                test_class_recall += tcr
                test_class_ap += tca
                test_m_f += tmf
                i += 1
                total_i += 1
        epoch += 1

    # Compute stats on the test set:
    train_test = False
    dev_test = False
    test_test = True
    a, c, r, p, f, stra, hl, da, dc, dr, dp, df1, dhl, ta, tc, tr, tp, tf1, thl, tcp, tcr, tca, tmf = training_step(i * epoch, dev_test,
                                                                                                                    train_test, test_test,
                                                                                                                    x_placeholder, y_placeholder,
                                                                                                                    dropout_placeholder, x_batch,
                                                                                                                    y_batch,
                                                                                                                    x_dev, y_dev, x_test, y_test,
                                                                                                                    dropout_keep_prob,
                                                                                                                    training_step_tf, accuracy,
                                                                                                                    cross_entropy_loss, sess,
                                                                                                                    predictions, train_writer,
                                                                                                                    test_writer, merged, recall,
                                                                                                                    rec_update, precision,
                                                                                                                    pre_update, F1,
                                                                                                                    streaming_accuracy,
                                                                                                                    str_acc_update, batch_precision,
                                                                                                                    batch_recall, batch_f1, logits, y_preds, verbose, multichannel)
    # Update training stats
    test_accuracy += ta
    test_loss += tc
    test_recall += tr
    test_precision += tp
    test_f1 += tf1
    test_hl += thl
    test_class_precision += tcp
    test_class_recall += tcr
    test_class_ap += tca
    test_m_f += tmf
    # Save the variables to disk.
    save_model(output, sess, saver)

    # Save results
    np.savetxt('./results/train_stats/train_acc.txt', np.array(train_accuracy), delimiter=',')
    np.savetxt('./results/train_stats/train_str_acc.txt', np.array(train_streaming_accuracy), delimiter=',')
    np.savetxt('./results/train_stats/train_loss.txt', np.array(train_loss), delimiter=',')
    np.savetxt('./results/train_stats/train_recall.txt', np.array(train_recall), delimiter=',')
    np.savetxt('./results/train_stats/train_precision.txt', np.array(train_precision), delimiter=',')
    np.savetxt('./results/train_stats/train_f1.txt', np.array(train_f1), delimiter=',')
    np.savetxt('./results/train_stats/train_hamming_loss.txt', np.array(train_hl), delimiter=',')
    np.savetxt('./results/train_stats/dev_acc.txt', np.array(dev_accuracy), delimiter=',')
    np.savetxt('./results/train_stats/dev_loss.txt', np.array(dev_loss), delimiter=',')
    np.savetxt('./results/train_stats/dev_recall.txt', np.array(dev_recall), delimiter=',')
    np.savetxt('./results/train_stats/dev_precision.txt', np.array(dev_precision), delimiter=',')
    np.savetxt('./results/train_stats/dev_f1.txt', np.array(dev_f1), delimiter=',')
    np.savetxt('./results/train_stats/dev_hamming_loss.txt', np.array(dev_hl), delimiter=',')
    np.savetxt('./results/train_stats/test_acc.txt', np.array(test_accuracy), delimiter=',')
    np.savetxt('./results/train_stats/test_loss.txt', np.array(test_loss), delimiter=',')
    np.savetxt('./results/train_stats/test_recall.txt', np.array(test_recall), delimiter=',')
    np.savetxt('./results/train_stats/test_precision.txt', np.array(test_precision), delimiter=',')
    np.savetxt('./results/train_stats/test_f1.txt', np.array(test_f1), delimiter=',')
    np.savetxt('./results/train_stats/test_hamming_loss.txt', np.array(test_hl), delimiter=',')
    np.savetxt('./results/train_stats/test_class_precision.txt', np.array(test_class_precision), delimiter=',',fmt="%s")
    np.savetxt('./results/train_stats/test_class_recall.txt', np.array(test_class_recall), delimiter=',',fmt="%s")
    np.savetxt('./results/train_stats/test_class_ap.txt', np.array(test_class_ap), delimiter=',',fmt="%s")
    np.savetxt('./results/train_stats/test_m_f.txt', np.array(test_m_f), delimiter=',')

    # Shutdown
    sess.close()

    # Plot results
    if(plot):
        plot(train_accuracy, train_loss, dev_accuracy, dev_loss, epoch)

    return test_f1[-1]

def save_model(output, sess, saver):
    # save checkpoint
    saver.save(sess, output + "/models/model.ckpt")

    #Freeze graph and weights for later model serving
    tf.train.write_graph(sess.graph_def, output + "/models/", 'model.pbtxt')
    input_graph_path = output + "/models/model.pbtxt"
    checkpoint_path = output + "/models/model.ckpt"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = output + '/models/frozen_model.pb'

    freeze_graph.freeze_graph(input_graph_path, input_saver="",
                              input_binary=False, input_checkpoint=checkpoint_path,
                              output_node_names="output/predictions", restore_op_name="save/restore_all",
                              filename_tensor_name="save/Const:0",
                              output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")


def plot(train_accuracy, train_cost, dev_accuracy, dev_cost, epochs):
    """ Plot and visualise the accuracy and loss """

    print('Final train accuracy ' + str(dev_accuracy[-1]))
    print('Final train loss ' + str(dev_cost[-1]))

    # accuracy training vs devset
    plt.plot(train_accuracy, label='Train data')
    plt.xlabel('Epoch')
    plt.plot(dev_accuracy, label='Test data')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.title('Accuracy per epoch train vs dev')
    plt.show()

    # loss training vs devset
    plt.plot(train_cost, label='Train data')
    plt.plot(dev_cost, label='Test data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per epoch, train vs dev')
    plt.grid(True)
    plt.show()


def restore_model(restorePath):
    """ Restore model from disk """

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, restorePath)
        print("Model restored.")

def parse_args():
    """Parses the commandline arguments with argparse"""
    parser = argparse.ArgumentParser(description='Parse flags to configure the json parsing')
    parser.add_argument("-ftr", "--featurestrain", help="path to input train features",
                        default="./data/cleaned/train/features.csv")
    parser.add_argument("-ltr", "--labelstrain", help="path to input train labels",
                        default="./data/cleaned/train/labels.csv")
    parser.add_argument("-fte", "--featurestest", help="path to input test features",
                        default="./data/cleaned/test/features.csv")
    parser.add_argument("-lte", "--labelstest", help="path to input test labels",
                        default="./data/cleaned/test/labels.csv")
    parser.add_argument("-md", "--maxdocumentsize", help="max document size", default=2000, type=int)
    parser.add_argument("-tp", "--testsplit", help="test split", default=0.1, type=float)
    parser.add_argument("-ovf", "--outputvocabularyfile", help="path to where to save the vocab",
                        default='./vocab/vocab.csv')
    parser.add_argument("-e", "--epochs", help="number of epochs to train", default=10, type=int)
    parser.add_argument("-mi", "--maxiterations", help="maxiterations, if want to quit before a whole epoch", default=10000000000, type=int)
    parser.add_argument("-bs", "--batchsize", help="batch size for training", default=64, type=int)
    parser.add_argument("-vd", "--vectordim", help="word vector dimension", default=300, type=int)
    parser.add_argument("-lr", "--learningrate", help="learningrate", default=0.01, type=float)
    parser.add_argument("-dkb", "--dropoutkeepprob", help="dropout keep probability", default=0.7, type=float)
    parser.add_argument("-fs", "--filtersizes", help="filtersizes", nargs='+', default=[3, 4, 5], type=int)
    parser.add_argument("-nf", "--numfilters", help="number of filters", default=128, type=int)
    parser.add_argument("-l2r", "--l2reglambda", help="lambda factor term for L2 regularization", default=0.0,
                        type=float)
    parser.add_argument("-o", "--output", help="folder to save results", default="./results")
    parser.add_argument("-hr", "--hyperandom", help="flag whether to run hyperparameter tuning with random search",action="store_true")
    parser.add_argument("-hg", "--hypegrid", help="flag whether to run hyperparameter tuning with grid search",action="store_true")
    parser.add_argument("-pr", "--pretrained", help="flag whether to user pretrained embeddings",action="store_true")
    parser.add_argument("-vec", "--vectors", help="path to pre-trained vectors", default="./data/vectors2.vec")
    parser.add_argument("-mc", "--multichannel", help="flag whether to run with multiple input channels or merge into one channel",action="store_true")
    parser.add_argument("-ver", "--verbose", help="flag whether to run with verbose logging",action="store_true")
    parser.add_argument("-pl", "--plot", help="flag whether to plot at the end",action="store_true")
    parser.add_argument("-gen", "--generative", help="flag whether to use generative model and data programming to combine votes, otherwise uses majority vote",action="store_true")
    parser.add_argument("-pp", "--preprocess", help="flag whether to do the data cleaning steps",action="store_true")
    args = parser.parse_args()
    return args

def setup(args):
    """Setup labels and features for training"""
    pre_process.test_labels_to_csv("./data/annotated.json", "./data/cleaned/test/labels.csv")
    pre_process.pre_process_labels("./data/votes.json", "./data/cleaned/train/labels.csv")
    pre_process.pre_process_features(["./data/text.csv"], "./data/cleaned/train/features.csv", args.multichannel)
    pre_process.pre_process_features(["./data/text.csv"], "./data/cleaned/test/features.csv", args.multichannel)

def apply_labeling():
    """ Apply key word indicators (weak labels), this takes time"""
    pre_process.keyword_labeling_funs("./data/high_level_class/total2.json", "./data/features.csv", "./data/vectors2.vec", "./data/high_level_class/total3.json")


def combine_rdd():
    """ Combine files stored as rdds parallelized"""
    pre_process.combine_labels([
    "./data/high_level_class/rdd_1",
    "./data/high_level_class/rdd_2",
    "./data/high_level_class/rdd_2",
    "./data/high_level_class/rdd_3",
    "./data/high_level_class/rdd_4",
    "./data/high_level_class/rdd_5",
    "./data/high_level_class/rdd_6",
    "./data/high_level_class/rdd_7",
    "./data/high_level_class/rdd_8",
    "./data/high_level_class/rdd_9"
    ],
    "./data/high_level_class/total.json")
    pre_process.combine_labels2([
    "./data/labelled/test_5k00",
    "./data/labelled/test_5k01",
    "./data/labelled/test_5k03",
    "./data/labelled/test_5k0204050607",
    "./data/labelled/test_5k0809101112"
    ],
    "./data/labelled/total.json", "./data/labelled/total_json_clean.json")
    pre_process.merge_totals("./data/high_level_class/total.json", "./data/labelled/total.json", "./data/high_level_class/total2.json")


if __name__ == "__main__":
    # Parse args
    args = parse_args()
    if(args.preprocess):
        setup(args)
    if(args.hyperandom):
        hype_random(args)
    if(args.hypegrid):
        hype_grid(args)
    if(not args.hyperandom and not args.hypegrid):
        if(args.pretrained):
            # Preprocess data to the right format
            x, vocab_processor, y, x_test, y_test, W = pre_process.combine_labels_features(
                args.featurestrain,
                args.labelstrain,
                args.featurestest,
                args.labelstest,
                args.vectors,
                args.vectordim,
                args.maxdocumentsize,
                args.multichannel,
                args.generative
            )
            # Train/Dev splits of the train data
            x_train, x_dev, y_train, y_dev = pre_process.split(x, y, args.testsplit, vocab_processor.vocabulary_)
            sequence_length = x_train.shape[1]
            # Get vocabulary and save to disk
            vocab_dict = vocab_processor.vocabulary_._mapping
            with open(args.outputvocabularyfile, 'w') as fp:
                for k, v in vocab_dict.iteritems():
                    fp.write(str(k) + "," + str(v) + "\n")

            # Begin training
            main(sequence_length, len(vocab_processor.vocabulary_), x_train, y_train, x_dev, y_dev,
                 x_test, y_test, W=W, pretrained=True, num_epochs=args.epochs, batch_size=args.batchsize, vectorDim=args.vectordim,
                 learning_rate=args.learningrate, dropout_keep_prob=args.dropoutkeepprob, filter_sizes=args.filtersizes,
                 num_filters=args.numfilters,
                 l2_reg_lambda=args.l2reglambda,
                 maxiterations=args.maxiterations, verbose=args.verbose, plot=args.plot,
                 multichannel=args.multichannel)
        else:
            # Preprocess data to the right format
            x, vocab_processor, y, x_test, y_test = pre_process.combine_labels_features(
                args.featurestrain,
                args.labelstrain,
                args.featurestest,
                args.labelstest,
                "",
                args.vectordim,
                args.maxdocumentsize,
                args.multichannel,
                args.generative
            )
            # Train/Dev splits of the train data
            x_train, x_dev, y_train, y_dev = pre_process.split(x, y, args.testsplit, vocab_processor.vocabulary_)
            if(args.multichannel):
                sequence_length = x_train.shape[2]
            else:
                sequence_length = x_train.shape[1]
            # Get vocabulary and save to disk
            vocab_dict = vocab_processor.vocabulary_._mapping
            with open(args.outputvocabularyfile, 'w') as fp:
                for k, v in vocab_dict.iteritems():
                    fp.write(str(k) + "," + str(v) + "\n")
            # Begin training
            main(sequence_length, len(vocab_processor.vocabulary_), x_train, y_train, x_dev, y_dev,
                 x_test, y_test, num_epochs=args.epochs, batch_size=args.batchsize, vectorDim=args.vectordim,
                 learning_rate=args.learningrate, dropout_keep_prob=args.dropoutkeepprob, filter_sizes=args.filtersizes,
                 num_filters=args.numfilters,
                 l2_reg_lambda=args.l2reglambda, maxiterations=args.maxiterations, verbose=args.verbose, plot=args.plot,
                 multichannel=args.multichannel)

