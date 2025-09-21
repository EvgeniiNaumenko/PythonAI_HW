import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
#   Перцептрон
# ==========================
class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Xavier initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        self.loss_history = []
        self.accuracy_history = []

    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        espilon = 1e-15
        y_pred = np.clip(y_pred, espilon, 1 - espilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def backward_propagation(self, X, y_true, y_pred):
        m = X.shape[0]
        dZ2 = y_pred - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # обновляем веса
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def caluclate_accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(y_true_labels == y_pred_labels)

    def train(self, X_train, y_train, X_val, y_val, epochs=500, verbose=True):
        for epoch in range(epochs):
            y_pred_train = self.forward_propagation(X_train)
            loss = self.compute_loss(y_train, y_pred_train)
            self.loss_history.append(loss)

            self.backward_propagation(X_train, y_train, y_pred_train)

            if epoch % 50 == 0:
                val_pred = self.forward_propagation(X_val)
                val_accuracy = self.caluclate_accuracy(y_val, val_pred)
                self.accuracy_history.append(val_accuracy)
                if verbose:
                    print(f"Эпоха {epoch+1}/{epochs} - Втрата: {loss:.4f} - Вал. точність: {val_accuracy:.4f}")


# ==========================
#   Утилиты
# ==========================
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def load_and_prepare_data():
    print("Завантаження даних (letters A–Z)...")
    letters = fetch_openml("letter", version=1, as_frame=False)
    X, y = letters.data, letters.target

    # Преобразуем буквы в числа
    le = LabelEncoder()
    y = le.fit_transform(y)  # A=0, ..., Z=25

    # train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    num_classes = 26
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_val_onehot = one_hot_encode(y_val, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)

    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, y_test, scaler, le

def evaluate_model(model, X_test, y_test_onehot, y_test_labels, le):
    y_pred = model.forward_propagation(X_test)
    test_accuracy = model.caluclate_accuracy(y_test_onehot, y_pred)
    print(f"\nТочність на тестовому наборі: {test_accuracy:.4f}")

    y_pred_labels = np.argmax(y_pred, axis=1)

    print("\nЗвіт про класифікацію:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=le.classes_))

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Передбачені мітки')
    plt.ylabel('Справжні мітки')
    plt.title('Матриця плутанини (A-Z)')
    plt.show()



def main():
    print(" -- ПЕРЦЕПТРОН ДЛЯ РОЗПІЗНАВАННЯ БУКВ --")
    X_train, y_train, X_val, y_val, X_test, y_test_onehot, y_test_labels, scaler, le = load_and_prepare_data()

    input_size = X_train.shape[1]   # 16x16=256
    hidden_size = 128
    output_size = 26
    learning_rate = 0.1

    perceptron = Perceptron(input_size, hidden_size, output_size, learning_rate)
    perceptron.train(X_train, y_train, X_val, y_val, epochs=2000, verbose=True)

    evaluate_model(perceptron, X_test, y_test_onehot, y_test_labels, le)

    # Примеры предсказаний
    preds = perceptron.predict(X_test[:20])
    print("\nПриклади передбачень:")
    for i, p in enumerate(preds):
        print(f"Зразок {i+1}: {le.inverse_transform([y_test_labels[i]])[0]} -> {le.inverse_transform([p])[0]}")

    return perceptron

if __name__ == "__main__":
    model = main()

