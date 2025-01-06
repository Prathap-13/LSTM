import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import seaborn as sns

# Feature Extraction
def extract_features(df):
    df['Start time UTC'] = pd.to_datetime(df['Start time UTC'])
    df['End time UTC'] = pd.to_datetime(df['End time UTC'])
    df['Duration'] = (df['End time UTC'] - df['Start time UTC']).dt.total_seconds() / 3600.0  # Duration in hours
    df = df.set_index('Start time UTC')
    return df

# Resample and Prepare Dataset
def prepare_dataset(df, column, window_size, forecast_horizon):
    df_resampled = df[column].resample('h').mean()  # Use 'h' instead of 'H'
    df_resampled = df_resampled.ffill()  # Use .ffill() instead of fillna(method='ffill')
    
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_resampled.values.reshape(-1, 1))
    
    X, y = [], []  # Ensure X and y are initialized correctly
    for i in range(window_size, len(scaled_data) - forecast_horizon):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i:i+forecast_horizon, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Split dataset
def split_dataset(X, y, train_ratio=0.7, val_ratio=0.2):
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Define LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plotting function
def plot_results(original, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(original, color='blue', label='Actual')
    plt.plot(predicted, color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.show()

# Future Forecasting
def forecast_future_steps(model, data, n_steps, scaler, forecast_horizon):
    predictions = []
    current_input = data[-1].reshape(1, n_steps, 1)  # Assuming n_features = 1
    
    for _ in range(forecast_horizon):
        next_prediction = model.predict(current_input)
        predictions.append(next_prediction[0, 0])
        current_input = np.append(current_input[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)  # Reshape next_prediction
        
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# GUI Setup
class EnergyForecastingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analysis of the Energy Consumption Forecasting Using LSTM")

        # Load background image
        self.background_image = Image.open("P:/Final_project/light.jpg")
        self.background_image = self.background_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.canvas = tk.Canvas(root, width=self.background_photo.width(), height=self.background_photo.height())
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")

        # Header Label
        self.header_label = tk.Label(root, text="Analysis of the Energy Consumption Forecasting Using LSTM", font=("Panton Rust Sans", 23, "bold"), bg='black', fg='white', padx=15, pady=15, borderwidth=2, relief="solid")
        self.header_label.place(relx=0.5, rely=0.1, anchor="center")  # Centered horizontally and 10% from the top

        # Label and Buttons
        self.label = tk.Label(root, text="Upload File for Energy Consumption Forecasting", font=("Math Bold", 18, "bold"), bg='Blue', fg='white', padx=10, pady=10, borderwidth=2, relief="solid")
        self.label.place(relx=0.5, rely=0.3, anchor="center")  # Centered horizontally and 30% from the top

        # Function to create rounded buttons
        def create_rounded_button(master, text, command, width=20, padx=10, pady=10, bg='aqua', fg='black'):
            button = tk.Button(master, text=text, font=("Panton Rust Script ExtraBold Base Shadow", 18, "bold"),
                               command=command, width=width, padx=padx, pady=pady, bg=bg, fg=fg,
                               borderwidth=2, relief="raised", highlightthickness=1, highlightbackground="black", highlightcolor="black")
            return button

        self.upload_button = create_rounded_button(root, "Upload file", self.upload_file)
        self.upload_button.place(relx=0.5, rely=0.4, anchor="center")
        self.bind_button_hover_effect(self.upload_button)

        self.evaluate_button = create_rounded_button(root, "Model Evaluation", self.model_evaluation, width=20, padx=10, pady=10, bg='aqua', fg='black')
        self.evaluate_button.place(relx=0.5, rely=0.5, anchor="center")
        self.evaluate_button.config(state=tk.DISABLED)
        self.bind_button_hover_effect(self.evaluate_button)

        self.forecast_button = create_rounded_button(root, "Future Forecasting", self.future_forecasting, width=20, padx=10, pady=10, bg='aqua', fg='black')
        self.forecast_button.place(relx=0.5, rely=0.6, anchor="center")
        self.forecast_button.config(state=tk.DISABLED)
        self.bind_button_hover_effect(self.forecast_button)

    def bind_button_hover_effect(self, button):
        button.bind("<Enter>", lambda e, b=button: self.on_enter(e, b, 'green'))
        button.bind("<Leave>", lambda e, b=button: self.on_leave(e, b, 'aqua'))

    def on_enter(self, event, button, color):
        button.config(bg=color)

    def on_leave(self, event, button, color):
        button.config(bg='aqua')

    def upload_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.label.config(text="File uploaded successfully.")
            self.process_file()

    def process_file(self):
        # Process the file and enable the required buttons
        self.df = pd.read_csv(self.file_path)
        self.df = extract_features(self.df)
        self.column = 'Electricity consumption in Finland'
        self.window_size = 24
        self.forecast_horizon = 24
        self.X, self.y, self.scaler = prepare_dataset(self.df, self.column, self.window_size, self.forecast_horizon)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = split_dataset(self.X, self.y)
        self.model = create_lstm_model((self.X_train.shape[1], 1))
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_data=(self.X_val, self.y_val))
        self.evaluate_button.config(state=tk.NORMAL)
        self.forecast_button.config(state=tk.NORMAL)
        self.plot_consumption_vs_year()

    def model_evaluation(self):
        y_pred_train = self.model.predict(self.X_train)
        y_pred_val = self.model.predict(self.X_val)
        y_pred_test = self.model.predict(self.X_test)
        
        y_train_inv = self.scaler.inverse_transform(self.y_train[:, 0].reshape(-1, 1))
        y_val_inv = self.scaler.inverse_transform(self.y_val[:, 0].reshape(-1, 1))
        y_test_inv = self.scaler.inverse_transform(self.y_test[:, 0].reshape(-1, 1))
        
        y_pred_train_inv = self.scaler.inverse_transform(y_pred_train)
        y_pred_val_inv = self.scaler.inverse_transform(y_pred_val)
        y_pred_test_inv = self.scaler.inverse_transform(y_pred_test)
        
        train_mse = mean_squared_error(y_train_inv, y_pred_train_inv)
        val_mse = mean_squared_error(y_val_inv, y_pred_val_inv)
        test_mse = mean_squared_error(y_test_inv, y_pred_test_inv)
        
        messagebox.showinfo("Model Evaluation Results",
                            f"Train MSE: {train_mse:.4f}\n"
                            f"Validation MSE: {val_mse:.4f}\n"
                            f"Test MSE: {test_mse:.4f}")

        plot_results(y_train_inv, y_pred_train_inv, 'Training Data')
        plot_results(y_val_inv, y_pred_val_inv, 'Validation Data')
        plot_results(y_test_inv, y_pred_test_inv, 'Test Data')

    def future_forecasting(self):
        forecast_horizon = 24  # Predict for the next 24 hours
        predictions = forecast_future_steps(self.model, self.X_test, self.window_size, self.scaler, forecast_horizon)
        
        # Plotting the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(predictions, color='red', label='Future Predictions')
        plt.title('Future Forecasting')
        plt.xlabel('Time (hours)')
        plt.ylabel('Energy Consumption')
        plt.legend()
        plt.show()

    def plot_consumption_vs_year(self):
        # Plotting the boxplot of energy consumption vs year
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(x=self.df.index.year, y=self.df[self.column], ax=ax)
        ax.set_title("Energy Consumption vs Year")
        ax.set_xlabel("Year")
        ax.grid(True, alpha=1)

        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(90)

        plt.show()

# Main code to run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyForecastingApp(root)
    root.mainloop()
