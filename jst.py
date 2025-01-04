import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class BackpropagationNN:
    def _init_(self, input_size=4, hidden_size=2, output_size=1):
        # Inisialisasi arsitektur jaringan
        self.input_size = input_size   # input_size: jumlah neuron di layer input
        self.hidden_size = hidden_size # hidden_size: jumlah neuron di layer tersembunyi
        self.output_size = output_size # output_size: jumlah neuron di layer output

        # Inisialisasi parameter pembelajaran
        self.learning_rate = 0.1  # learning_rate: menentukan seberapa besar perubahan bobot pada setiap iterasi
        self.momentum = 0.9  # momentum: membantu mempercepat pembaruan bobot agar lebih stabil

        # Inisialisasi bobot dengan nilai random kecil
        np.random.seed(42) # np.random.normal(mean, std_dev, size): menghasilkan nilai acak dengan distribusi normal
        self.hidden_weights = np.random.normal(0, 0.5, (input_size, hidden_size)) # Bobot dari input ke hidden
        self.output_weights = np.random.normal(0, 0.5, (hidden_size, output_size)) # Bobot dari hidden ke output

        # Inisialisasi bias
        self.hidden_bias = np.zeros((1, hidden_size))
        self.output_bias = np.zeros((1, output_size))

        # Inisialisasi momentum
        self.hidden_weights_momentum = np.zeros_like(self.hidden_weights)
        self.output_weights_momentum = np.zeros_like(self.output_weights)

    def sigmoid(self, x):
        # Fungsi aktivasi sigmoid
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Turunan fungsi sigmoid
        return x * (1 - x)

    def forward(self, X):
        # Proses propagasi maju melalui jaringan
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Input ke hidden layer
        self.hidden_sum = np.dot(X, self.hidden_weights) + self.hidden_bias
        self.hidden_output = self.sigmoid(self.hidden_sum)

        # Hidden ke output layer
        self.output_sum = np.dot(self.hidden_output, self.output_weights) + self.output_bias
        self.output = self.sigmoid(self.output_sum)

        return self.output

    def backward(self, X, y, output):
        # Proses propagasi mundur untuk memperbarui bobot
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Hitung error di output layer
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Hitung error di hidden layer
        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Perbarui bobot dengan momentum
        output_weights_update = (self.learning_rate * np.dot(self.hidden_output.T, output_delta) +
                               self.momentum * self.output_weights_momentum)
        hidden_weights_update = (self.learning_rate * np.dot(X.T, hidden_delta) +
                               self.momentum * self.hidden_weights_momentum)

        # Perbarui bobot dan bias
        self.output_weights += output_weights_update
        self.hidden_weights += hidden_weights_update
        self.hidden_bias += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        self.output_bias += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        # Simpan momentum
        self.output_weights_momentum = output_weights_update
        self.hidden_weights_momentum = hidden_weights_update

    def train(self, X, y, epochs=5000, target_mse=0.0001, batch_size=1):
        # Melatih jaringan
        X = np.array(X)
        y = np.array(y)

        mse_history = []

        for epoch in range(epochs):
            total_mse = 0

            # Pelatihan mini-batch
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, output)

                # Hitung MSE
                mse = np.mean(np.square(batch_y - output))
                total_mse += mse

            avg_mse = total_mse / (len(X) / batch_size)
            mse_history.append(avg_mse)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, MSE: {avg_mse:.6f}")

            if avg_mse <= target_mse:
                print(f"Target MSE tercapai pada epoch {epoch}")
                break

        return mse_history

    def predict(self, X):
        # Melakukan prediksi menggunakan jaringan yang sudah dilatih
        return self.forward(X)

if __name__ == "_main_":
    # Dataset dari jurnal
    data = {
        'X1': [0.8, 0.8, 0.8, 0.8, 0.6, 0.8, 0.6, 0.8, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.8, 0.8, 0.8],
        'X2': [0.6, 1.0, 0.6, 0.8, 0.6, 0.8, 0.8, 0.6, 0.8, 0.8, 0.8, 0.4, 0.8, 0.6, 1.0, 0.8, 0.8, 0.6],
        'X3': [0.6, 0.8, 0.4, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0, 0.8, 0.8, 0.8, 0.6, 0.8],
        'X4': [0.8, 0.8, 0.4, 1.0, 0.6, 0.8, 0.8, 0.8, 0.8, 0.6, 0.8, 0.8, 1.0, 0.6, 0.8, 0.8, 0.8, 1.0],
        'Target': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }

    # Konversi ke DataFrame
    df = pd.DataFrame(data)

    # Pisahkan fitur dan target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # Normalisasi data
    scaler_X = MinMaxScaler(feature_range=(0.1, 0.8))
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.8))

    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y)

    # Buat dan latih model
    model = BackpropagationNN(input_size=4, hidden_size=2, output_size=1)
    history = model.train(X_normalized, y_normalized, epochs=5000, target_mse=0.00001)

    # Prediksi dan metrik
    predictions_normalized = model.predict(X_normalized)
    predictions = scaler_y.inverse_transform(predictions_normalized)
    mse = np.mean(np.square(y - predictions))
    rmse = np.sqrt(mse)

    # Perhitungan akurasi
    predicted_classes = [1 if pred >= 0.5 else 0 for pred in predictions.flatten()]
    actual_classes = y.flatten()
    correct_predictions = np.sum(np.array(predicted_classes) == np.array(actual_classes))
    accuracy = (correct_predictions / len(y)) * 100

    # Tampilkan hasil akhir
    print("HASIL PENGUJIAN DAN PREDIKSI")
    print("=" * 80)
    print(f"{'NIS':^6} | {'Prediksi':^10} | {'Target':^6} | {'Kategori':^15} | {'JST (Error)':^15} | {'Hasil':^10}")
    print("=" * 80)

    # Hitung nilai JST dan kategori
    jst_values = np.abs(predictions.flatten() - y.flatten())
    categories = [
        "Sangat Memahami" if jst <= 0.0010 else
        "Memahami" if 0.0011 <= jst <= 0.0100 else
        "Cukup Memahami" if 0.0100 <= jst <= 0.10001 else
        "Tidak Paham"
        for jst in jst_values
    ]
    results = ["Benar" if category != "Tidak Paham" else "Salah" for category in categories]

    # Tampilkan hasil per siswa
    for i, (nis, pred, target, category, jst, result) in enumerate(zip(range(1542, 1542 + len(y)), predictions.flatten(), y.flatten(), categories, jst_values, results)):
        print(f"{nis:^6} | {pred:^10.4f} | {target:^6.1f} | {category:^15} | {jst:^15.5f} | {result:^10}")
    print("=" * 80)

    # Plot grafik MSE selama epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), history, label="MSE", color="blue")
    plt.title("Training Loss (MSE) vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

    # Plot perbandingan prediksi vs target
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y)), y, label="Target", marker='o', color="blue")
    plt.plot(range(len(predictions)), predictions.flatten(), label="Predictions", marker='x', color="red")
    plt.title("Predictions vs Target")
    plt.xlabel("Data Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # Tampilkan ringkasan hasil
    print("\nHASIL AKHIR:")
    print(f"MSE (Mean Squared Error): {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Akurasi: {accuracy:.2f}%")