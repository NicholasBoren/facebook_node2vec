# Required Imports
import os
from collections import defaultdict
import math
import networkx as nx
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# Authentication and Google Drive setup
def setup_google_drive():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive


# Load Dataset
def load_dataset(drive):
    id = '1jSuo8v4I0EUHFe_L_Z9WcHoym81h5QYy'
    downloaded = drive.CreateFile({'id': id})
    downloaded.GetContentFile('musae_facebook_edges.csv')

    id = '1fWARzj2We5ihvzPn8oHEJkqhlNN5CKNl'
    downloaded = drive.CreateFile({'id': id})
    downloaded.GetContentFile('musae_facebook_target.csv')

    edges = pd.read_csv("musae_facebook_edges.csv")
    targets_df = pd.read_csv('musae_facebook_target.csv')
    return edges, targets_df


# Convert IDs to Names and vice-versa
def id_to_name(id, targets_df):
    return targets_df[targets_df.id == id].page_name.values[0]


def name_to_id(name, targets_df):
    return targets_df[targets_df.page_name == name].id.values[0]


# Function to get filtered dataframe
def get_filtered_df(graph, targets_df):
    # Obtain a filtered dataframe from smaller graph and sort by node degree
    d = {
        "id": [],
        "name": [],
        "page_type": [],
        "degree": [],
    }

    for n in graph.nodes:
        d["id"].append(n)
        d["name"].append(id_to_name(n))
        d["page_type"].append(targets_df[targets_df.id == n].page_type.values[0])
        d["degree"].append(graph.degree(n))

    filtered_target_df = pd.DataFrame.from_dict(d)
    filtered_target_df.sort_values(by=['degree'], inplace=True, ascending=False)

    # Get TV shows with highest degree
    print(filtered_target_df[filtered_target_df.page_type == 'tvshow'].head(10))
    print()
    # Get politicians with highest degree
    print(filtered_target_df[filtered_target_df.page_type == 'politician'].head(10))
    print()
    return filtered_target_df


# Random Walk Functions
def next_step(graph, previous, current, p, q):
    """
    Get the next step for the random walk based on current location in graph,
    previous node visited, and transition probabalities.

    :param graph: networkx graph of Facebook pages
    :param previous: previous node visited in walk
    :param current: current node in walk
    :param p: return parameter
    :param q: in-out parameter
    :returns: choice of neighbor to visit
    """
    np.random.seed(5)
    neighbors = list(graph.neighbors(current))

    weights = []

    # Adjust the weights of the edges to the neighbors with respect to p and q.
    for neighbor in neighbors:
        if neighbor == previous:
            weights.append(1 / p)
        elif graph.has_edge(previous, neighbor):
            weights.append(1)
        else:
            weights.append(1 / q)

    # Compute the probabilities of visiting each neighbor.
    total_weight = sum(weights)
    probs = [weight / total_weight for weight in weights]

    # Probabilistically select a neighbor to visit.
    next_step = np.random.choice(neighbors, p=probs)

    return next_step


def random_walk(graph, num_walks, num_steps, p, q):
    """
    Generate a sequence of random walks on the input graphwith length num_walks, 
    each taking num_steps. Use next_step to decide which node to visit next 
    during each step of each random walk.

    :param graph: networkx graph of Facebook pages
    :param num_walks: total number of random walks
    :param num_steps: number of steps within a single random walk
    :param p: return parameter
    :param q: in-out parameter
    :returns: sequence of random walks as a list of lists
    """
    # walks is the random walk sequence which contains num_walks lists of
    # individual walks taken for num_steps
    walks = []
    nodes = list(graph.nodes())
    # Perform multiple iterations of the random walk.
    for walk_iteration in range(num_walks):
        random.shuffle(nodes)

        for node in tqdm(
                nodes,
                position=0,
                leave=True,
                desc=f"Random walks iteration {walk_iteration + 1} of {num_walks}",
        ):
            walk = []
            current_node = node
            prev_node = None
            walk.append(vocabulary_lookup[current_node])
            for i in range(1, num_steps):
                next_node = next_step(graph, prev_node, current_node, p, q)
                prev_node = current_node
                current_node = next_node
                walk.append(vocabulary_lookup[current_node])

            walks.append(walk)

    return walks


# Generate Examples
def generate_examples(sequences, window_size, num_negative_samples, vocabulary_size):
    example_weights = defaultdict(int)
    # Iterate over all sequences (walks).
    for sequence in tqdm(
        sequences,
        position=0,
        leave=True,
        desc=f"Generating postive and negative examples",
    ):
        # Generate positive and negative skip-gram pairs for a sequence (walk).
        pairs, labels = keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            negative_samples=num_negative_samples,
        )
        for idx in range(len(pairs)):
            pair = pairs[idx]
            label = labels[idx]
            target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
            if target == context:
                continue
            entry = (target, context, label)
            example_weights[entry] += 1

    targets, contexts, labels, weights = [], [], [], []
    for entry in example_weights:
        weight = example_weights[entry]
        target, context, label = entry
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        weights.append(weight)

    return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)

# Create Dataset
def create_dataset(targets, contexts, labels, weights, batch_size):
    inputs = {
        "target": targets,
        "context": contexts,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    dataset = dataset.shuffle(buffer_size=batch_size * 2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Create Model
def create_model(vocabulary_size, embedding_dim):
    inputs = {
        "target": layers.Input(name="target", shape=(), dtype="int32"),
        "context": layers.Input(name="context", shape=(), dtype="int32"),
    }
    # Initialize item embeddings.
    embed_item = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="item_embeddings",
    )
    # Lookup embeddings for target.
    target_embeddings = embed_item(inputs["target"])
    # Lookup embeddings for context.
    context_embeddings = embed_item(inputs["context"])
    # Compute dot similarity between target and context embeddings.
    logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
        [target_embeddings, context_embeddings]
    )
    # Create the model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# Main Function
def main():
    # Setup
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(5)
    drive = setup_google_drive()

    # Load Data
    edges, targets_df = load_dataset(drive)

    # Construct Graph
    graph = nx.convert_matrix.from_pandas_edgelist(edges, "id_1", "id_2")
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph_small = nx.k_core(graph, k=15)
    graph_tiny = nx.k_core(graph, k=30)

    # Get Filtered DataFrames
    filtered_target_df = get_filtered_df(graph, targets_df)
    filtered_target_df_tiny = get_filtered_df(graph_tiny, targets_df)

    # Perform Random Walks
    selected_graph = graph_small  # Change this to graph_tiny for debugging/sanity checks
    p = 1
    q = 1
    num_walks = 5
    num_steps = 10
    walks = random_walk(selected_graph, num_walks, num_steps, p, q)

    # Generate Examples
    num_negative_samples = 4
    vocabulary_size = len(set(selected_graph.nodes))
    targets, contexts, labels, weights = generate_examples(walks, num_steps, num_negative_samples, vocabulary_size)

    # Create TensorFlow Dataset
    batch_size = 2048
    dataset = create_dataset(targets, contexts, labels, weights, batch_size)

    # Create and Train Model
    embedding_dim = 50
    learning_rate = 0.01
    num_epochs = 10
    model = create_model(vocabulary_size, embedding_dim)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))
    history = model.fit(dataset, epochs=num_epochs)


# Execute main function
if __name__ == "__main__":
    main()
