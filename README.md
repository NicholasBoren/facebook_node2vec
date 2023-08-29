### Project Title: 
Facebook Network Embedding using Random Walks and TensorFlow

### Description:

This repository contains a Python-based implementation of a network embedding model for Facebook pages. The project leverages NetworkX for graph manipulation, TensorFlow for machine learning, and Google Drive API for data storage and retrieval. The model uses random walks to generate sequences of nodes, which are then used to train a skip-gram model for node embeddings.

### Key Features:

- **Google Drive Integration**: Securely loads datasets from Google Drive.
- **Data Preprocessing**: Utilizes Pandas for data manipulation and NetworkX for graph-based operations.
- **Random Walks**: Generates random walks on the Facebook network graph to create contextual relationships between nodes.
- **Skip-gram Model**: Utilizes TensorFlow to implement a skip-gram model for learning node embeddings.
- **Optimized Performance**: Utilizes TensorFlow's data pipeline for efficient batching and training.

### Technologies Used:

- Python
- TensorFlow
- NetworkX
- Pandas
- Google Drive API
- Google Colab

### How to Run:

1. Clone the repository.
2. Make sure you have all the required packages installed.
3. Run the `main()` function.

### Use-Cases:

- Social Network Analysis
- Community Detection
- Recommendation Systems
