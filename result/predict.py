import ast
import pickle
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf
import classifier.tbcnn.network as network
import classifier.tbcnn.sampling as sampling
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def _name(node):
    return type(node).__name__

def _traverse_tree(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = list(ast.iter_child_nodes(current_node))
        queue.extend(children)
        for child in children:
            child_json = {
                "node": _name(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes


def build_tree(infile):
    """Builds an AST from a script."""
    script = ""
    with open(infile, 'r') as file_handler:
        script = file_handler.read()
    
    return ast.parse(script)

def save_tree(infile, outfile):
    """Save the tree of the given script"""
    result = []
    result.append({ 'tree': build_tree(infile), 'metadata': {'label': 'bubblesort'} })
    with open(outfile, 'wb') as file_handler:
        pickle.dump(result, file_handler)

def create_sample(infile, outfile, label_file):
    print("opening pickle file")
    data_source = ""
    with open(infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)
        file_handler.close()

    test_samples = []

    with open(label_file, 'rb') as file_handler:
        labels = pickle.load(file_handler)
        file_handler.close()

    for item in data_source:
        root = item['tree']
        label = item['metadata']['label']
        sample, size = _traverse_tree(root)
        datum = {'tree': sample, 'label': label}
        test_samples.append(datum)
    
    # create sample pickle file
    with open(outfile, 'wb') as file_handler:
        pickle.dump((test_samples, labels), file_handler)
        file_handler.close()
    print('Created sample pickle file '+outfile)

def classify_item(logdir, infile, embedfile, label_file):

    with open(label_file, 'rb') as fh:
        labels = pickle.load(fh)

    with open(infile, 'rb') as fh:
        trees, labels = pickle.load(fh)

    with open(embedfile, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh)
        num_feats = len(embeddings[0])

    # build the inputs and outputs of the network
    nodes_node, children_node, hidden_node = network.init_net(
        num_feats,
        len(labels)
    )
    out_node = network.out_layer(hidden_node)

    ### init the graph
    sess = tf.Session()#config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.global_variables_initializer())

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise 'Checkpoint not found.'
    
    for batch in sampling.batch_samples(
        sampling.gen_samples(trees, labels, embeddings, embed_lookup), 1
    ):
        nodes, children, batch_labels = batch
        # print(nodes)
        # print(children)
        # print(batch_labels)
        # return
        output = sess.run([out_node],
            feed_dict={
                nodes_node: nodes,
                children_node: children,
            }
        )
        print(labels[np.argmax(output)])


save_tree('result/input.py', 'result/example.pkl')
create_sample('result/example.pkl', 'result/nodes.pkl', 'sampler/data/labels.pkl');
classify_item('classifier/logs/1', 'result/nodes.pkl', 'vectorizer/data/vectors.pkl', 'sampler/data/labels.pkl');

