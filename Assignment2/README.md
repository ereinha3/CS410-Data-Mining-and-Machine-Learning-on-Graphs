# Assignment 2: Node Classification and Recommender Systems

This assignment explores advanced topics in graph machine learning, focusing on node classification and recommender systems.

## Structure

The assignment is divided into two main questions:

### Question 1: Node Classification
Located in `./q1/`, this question focuses on:
- Implementation of multiple node classification models:
  - Multi-Layer Perceptron (MLP)
  - Label Propagation
  - Graph Neural Networks (GNN)
- Generation of networks with varying homophily using Stochastic Block Model
- Comparative analysis of model performance

### Question 2: Recommender System
Located in `./q2/`, this question focuses on:
- Implementation of a recommender system using the RecBole library
- Hyperparameter tuning for optimal performance
- Model evaluation and comparison

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NetworkX
- RecBole library (for Question 2)
- Other dependencies as specified in each question's requirements

### Running the Code

#### Question 1
1. Navigate to the q1 directory:
   ```bash
   cd q1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the node classification experiments:
   ```bash
   python main.py
   ```

#### Question 2
1. Navigate to the q2 directory:
   ```bash
   cd q2
   ```
2. Install RecBole and dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the recommender system:
   ```bash
   python main.py
   ```

## Output

- Question 1: Results will include model performance metrics and visualizations of the networks
- Question 2: Results will include recommender system performance metrics and hyperparameter tuning results

## Note

Please refer to the assignment specification in `AS2.pdf` for detailed requirements and context for each question.
