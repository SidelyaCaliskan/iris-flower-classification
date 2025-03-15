import numpy as np
import csv
import random
from collections import Counter
import pickle
import matplotlib.pyplot as plt

species_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

species_reverse_mapping = {v: k for k, v in species_mapping.items()}


def load_data(filename):
    dataset = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if len(row) < 5:
                continue
            row[-1] = int(species_mapping[row[-1].strip()])
            dataset.append([float(x) for x in row[:4]] + [row[-1]])
    return np.array(dataset)


def train_test_split(data, test_size=0.2):
    np.random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def get_neighbors(train_data, test_sample, k=3):
    distances = []
    for train_sample in train_data:
        dist = euclidean_distance(test_sample, train_sample[:-1])
        distances.append((train_sample, dist))
    distances.sort(key=lambda x: x[1])
    return [neighbor[0] for neighbor in distances[:k]]


def predict(train_data, test_sample, k=3):
    test_sample = np.array(test_sample).reshape(1, -1)[0]
    neighbors = get_neighbors(train_data, test_sample, k)
    species = [int(neighbor[-1]) for neighbor in neighbors]
    return Counter(species).most_common(1)[0][0]


def evaluate_model(train_data, test_data, k=3):
    correct = sum(1 for test_sample in test_data if predict(train_data, test_sample[:-1], k) == test_sample[-1])
    return correct / len(test_data)


def plot_all_graphs(train_data, test_data, iris_data, k=3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['red', 'green', 'blue']
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    for i in range(3):
        subset = iris_data[iris_data[:, -1] == i]
        axes[0].scatter(subset[:, 0], subset[:, 1], color=colors[i], label=labels[i], alpha=0.6)
    axes[0].set_xlabel("Sepal Length")
    axes[0].set_ylabel("Sepal Width")
    axes[0].set_title("Iris Veri Setinin Dağılımı")
    axes[0].legend()

    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = np.array([predict(train_data[:, :2], point, k) for point in grid_points])
    predictions = predictions.reshape(xx.shape)
    axes[1].contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.coolwarm)
    for i in range(3):
        subset = train_data[train_data[:, -1] == i]
        axes[1].scatter(subset[:, 0], subset[:, 1], color=colors[i], label=labels[i], edgecolor='black')
    axes[1].set_xlabel("Sepal Length")
    axes[1].set_ylabel("Sepal Width")
    axes[1].set_title(f"KNN Karar Sınırı (k={k})")
    axes[1].legend()

    true_labels = [int(test_sample[-1]) for test_sample in test_data]
    predicted_labels = [int(predict(train_data, test_sample[:-1], k)) for test_sample in test_data]
    axes[2].scatter(range(len(test_data)), true_labels, color="blue", label="Gerçek Etiketler", marker="o")
    axes[2].scatter(range(len(test_data)), predicted_labels, color="red", label="Tahmin Edilen Etiketler", marker="x")
    axes[2].set_xlabel("Test Örnekleri")
    axes[2].set_ylabel("Tür Etiketi (0: Setosa, 1: Versicolor, 2: Virginica)")
    axes[2].set_title("Gerçek vs. Tahmin Edilen Sonuçlar")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    iris_data = load_data("/Users/sidelya/PycharmProjects/PythonProject/Iris.csv")
    train_data, test_data = train_test_split(iris_data, test_size=0.2)
    accuracy = evaluate_model(train_data, test_data, k=3)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    new_sample = np.array([5.1, 3.5, 1.4, 0.2])
    predicted_species = predict(train_data, new_sample, k=3)
    print("Predicted Species:", species_reverse_mapping[predicted_species])
    plot_all_graphs(train_data, test_data, iris_data, k=3)
    with open("knn_iris_model.pkl", "wb") as file:
        pickle.dump(train_data, file)
    with open("knn_iris_model.pkl", "rb") as file:
        loaded_train_data = pickle.load(file)
    accuracy_loaded_model = evaluate_model(loaded_train_data, test_data, k=3)
    print(f"Loaded Model Accuracy: {accuracy_loaded_model * 100:.2f}%")


if __name__ == '__main__':
    main()