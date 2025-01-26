import time
import json
import numpy as np


from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



def generate_data(users, items):
    """Generate a synthetic user-item interaction matrix with given sparsity."""

    return np.random.rand(users, items)


def calculate_similarity(data, method="cosine"):
    """
    Calculate similarity using selected method.

    :param data: user-item interaction matrix
    :param method: options are 'c'osine, 'e'uclidean, 'p'earson, 'j'accard
    :return: similarity matrix
    """
    
    if method == "c":
        return cosine_similarity(data)

    elif method == "e":
        return 1 / (1 + euclidean_distances(data))  # Normalize to 0-1 range
    
    elif method == "p":
        users = data.shape[0]
        similarity_matrix = np.zeros((users, users))

        for i in range(users):
            for j in range(users):
                if i != j:
                    valid_indices = (data[i] > 0) & (data[j] > 0)
                    if np.any(valid_indices):
                        similarity_matrix[i, j] = pearsonr(data[i, valid_indices], data[j, valid_indices])[0]
                    else:
                        similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = 1
        
        return similarity_matrix
    
    elif method == "j":
        users = data.shape[0]
        similarity_matrix = np.zeros((users, users))
        
        for i in range(users):
            for j in range(users):
                if i != j:
                    bin_i = data[i] > 0
                    bin_j = data[j] > 0
                    similarity_matrix[i, j] = jaccard_score(bin_i, bin_j)
                
                else:
                    similarity_matrix[i, j] = 1
        
        return similarity_matrix


def predict_ratings(dataset, similarity_matrix, user_based=True):
    """Predict ratings based on user or item-based collaborative filtering"""

    if user_based:
        return similarity_matrix.dot(dataset) / np.array(
            [np.abs(similarity_matrix).sum(axis=1)]).T

    else:
        return dataset.dot(similarity_matrix.T) / np.array(
            [np.abs(similarity_matrix).sum(axis=0)])


def evaluate_performance(true_ratings, predicted_ratings, threshold=0.5):
    """Evaluate the performance using F1-score."""


    mask = true_ratings > 0  # Only evaluate on non-zero entries
    y_true = (true_ratings[mask] > threshold).astype(int)  # Binarize true ratings
    y_pred = (predicted_ratings[mask] > threshold).astype(int)  # Binarize predicted ratings
    f1 = f1_score(y_true, y_pred, average='macro')  # Macro F1-score

    return f1


def main(users, items):
    # Generate synthetic data
    dataset = generate_data(users, items)

    # Split into train and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    train_dataset = scaler.fit_transform(train_data)
    test_dataset = scaler.transform(test_data)

    results = []
    method_map = {'c': 'cosine', 'e': 'euclidean', 'p': 'pearson', 'j': 'jaccard'}

    for method in ["c", "e", "p", "j"]:
        # Measure calculation time
        start_time = time.time()
        similarity_matrix = calculate_similarity(train_dataset, method=method)
        calculate_time = time.time() - start_time

        # Predict ratings for all data
        predicted_ratings = predict_ratings(train_data, similarity_matrix, user_based=True)

        # Evaluate only the test data points
        f1 = evaluate_performance(test_data, predicted_ratings[-test_data.shape[0]:], threshold=0.5)

        results.append({
            "method": method_map[method],
            "f1_score": round(f1, 4),
            "calculate_time": round(calculate_time, 4)
        })



    formatted_results = {"header": "", "body": []}
    formatted_results["header"] = {
        "users": users,
        "items": items
    }

    for result in results:
        formatted_results["body"].append(result)


    # write results as json formatted
    data = []

    try:
        with open("results.json", 'r') as file:
            data = json.load(file)
    except:
        pass

    with open("results.json", "w") as json_file:
        data.append(formatted_results)
        json.dump(data, json_file, indent=4)