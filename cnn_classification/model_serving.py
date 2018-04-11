
import logging
import random
import time
from google.protobuf import text_format
from flask import Flask, jsonify, request
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf

app = Flask(__name__)
app.config.from_object(__name__)

#Frozen model to server
MODEL_PATH = './results/models/frozen_model.pb'

#Conversion between numeric and categorical labels
int_to_label = {}
int_to_label[1] = "tops_and_tshirts"
int_to_label[2] = "bags"
int_to_label[3] = "all_accessories"
int_to_label[4] = "shoes"
int_to_label[5] = "jeans"
int_to_label[6] = "skirts"
int_to_label[7] = "tights_and_socks"
int_to_label[8] = "dresses"
int_to_label[9] = "jackets"
int_to_label[10] = "blouses_and_tunics"
int_to_label[11] = "trouser_and_shorts"
int_to_label[12] = "coats"
int_to_label[0] = "jumpers_and_cardigans"

def load_vocab(vocab_path):
    """ Load the vocabulary used for converting text into numeric format and lookup embeddings"""
    vocab = {}
    with open(vocab_path, 'r') as fp:
        for line in fp:
            parts = line.split(",")
            vocab[parts[0]] = int(parts[1])
    return vocab


def load_graph(frozen_graph_filename):
    """ Loads a frozen graph into memory """

    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph(MODEL_PATH)

def print_ops():
    """ Helper to print what operations exists in the frozen model"""
    for op in graph.get_operations():
        print(op.name)

# We access the input and output nodes, as well as the dropout node
x = graph.get_tensor_by_name('prefix/input_x:0')
y = graph.get_tensor_by_name('prefix/output/predictions:0')
dropout_keep_prob = graph.get_tensor_by_name('prefix/dropout_keep_prob:0')

# Create session
sess = tf.Session(graph=graph)

# Load saved vocabulary
vocab = load_vocab("./vocab/vocab.csv")

@app.route('/', methods=['POST'])
def classify():
    """ REST endpoint for classifying input text"""
    app.logger.info("Serving classification request")
    req_data = request.get_json(silent=True)
    text = req_data["text"]
    x_input = []
    missing_from_vocab = False
    for word in text.split(" "):
        if word in vocab:
            x_input.append(vocab[word])
        else:
            x_input.append(0)
            missing_from_vocab = True
    if len(x_input) < 2000:
        x_input_padded = np.pad(x_input, (0,2000-len(x_input)), 'constant', constant_values=(0, 0))
    else:
        x_input_padded = x_input[0:2000]
    t = time.time()
    y_out = sess.run(y, feed_dict={
        x: [x_input_padded],
        dropout_keep_prob:1.0
    })
    dt = time.time() - t
    app.logger.info("Execution time: %0.2f" % (dt * 1000.))
    pretty_print_output = {}
    y_out = y_out.tolist()
    for i in range(0, len(y_out[0])):
        pretty_print_output[int_to_label[i]] = y_out[0][i]
    return jsonify({"softmax_output":pretty_print_output, "execution_time": dt, "word_missing_from_vocab": missing_from_vocab})

# Startup the server
if __name__ == '__main__':
    app.run(debug=True, port=8009)