# Number storage and manipulation imports.
import numpy as np
import pandas as pd  

# Machine learning related imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam


ticker1 = "BTC".upper()
ticker2 = "XRP".upper()

csv_path = f"D:\\Coding\\VisualStudioCode\\Projects\\Python\\TA_Bot\\CandleStorage\\{ticker1}_candles.csv"

class SpikeML:
    def __init__(self) -> None:
        self.ticker1 = ticker1
        self.ticker2 = ticker2

        self.csv_path1 = f"D:\\Coding\\VisualStudioCode\\Projects\\Python\\TA_Bot\\CandleStorage\\{self.ticker1}_candles.csv"
        self.csv_path2 = f"D:\\Coding\\VisualStudioCode\\Projects\\Python\\TA_Bot\\CandleStorage\\{self.ticker2}_candles.csv"


    ##################################################################################################### Single Data
    def load_data(self, data=None, feature_cols:list = ["high"], use_csv:bool=False) -> None:

        # Step 1: Load and preprocess data
        if use_csv:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = data
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data.set_index('time', inplace=True)

        # Step 2: Define features and labels
        # 'high' column is the feature, and we will create the 'spike' column as the label.
        # If there is a spike, the spike column will be filled with 1, if there is 0, no spike occured. 
        self.data['spike'] = np.where(self.data['high'].diff() > 0, 1, 0)

        # Step 3: Data Splitting
        features = self.data[feature_cols].values
        labels = self.data['spike'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Step 4: Feature Scaling
        scaler = MinMaxScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    
    def train_model(self):

        # Step 5: Build the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(units=50, input_shape=(self.X_train_scaled.shape[1], 1)))
        self.model.add(Dense(units=1, activation='sigmoid'))

        # Step 6: Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Step 7: Model Training
        self.model.fit(self.X_train_scaled.reshape(self.X_train_scaled.shape[0], self.X_train_scaled.shape[1], 1),
                self.y_train, epochs=20, batch_size=32, validation_split=0.1)

    def evaluate_model(self):
        # Step 8: Model Evaluation
        loss, accuracy = self.model.evaluate(self.X_test_scaled.reshape(self.X_test_scaled.shape[0], self.X_test_scaled.shape[1], 1), self.y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def predict_single(self, new_data: pd.DataFrame, feature_cols: list = ["high"], label_col:str = "spike", steps:int = 0 ):
        # Step 9: Prediction
        # Process new data 
        new_data["time"] = pd.to_datetime(new_data["time"])
        new_data.set_index("time", inplace=True)

        # Extract features from the new data.
        new_features = new_data[feature_cols].values

        # Scale the new features
        scaler = MinMaxScaler()
        new_features_scaled = scaler.fit_transform(new_features)

         # Check if the number of time steps matches the model's input shape (6 time steps in this case)
        num_time_steps = steps
        if new_features_scaled.shape[0] < num_time_steps:
            raise ValueError(f"Input data must have at least {num_time_steps} time steps.")

        # Reshape the new_features_scaled to match the input shape. 
        # Reshape the new_features_scaled to match the input shape of the LSTM model
        new_features_reshaped = np.expand_dims(new_features_scaled[-num_time_steps:], axis=0)

        # Predict spike probabilities.
        predicted_probabilities = self.model.predict(new_features_reshaped)

        return predicted_probabilities
    
    ##################################################################################################### Multi Data
    def load_data_multi(self, dataframes: list, feature_cols:list = ["high"], label_col:str="spike",
                        test_size=0.2, random_state=42) -> None:
         # Step 1: Concatenate features from multiple DataFrames
        features_list = []
        for df in dataframes:
            df['spike'] = np.where(df['high'].diff() > 0, 1, 0)
            features = df[feature_cols].values
            df_features = pd.DataFrame(features, columns=feature_cols)  # Convert to DataFrame
            features_list.append(df_features)
        
        
        features = pd.concat(features_list, axis=1)  # Concatenate features along columns
        
        # Step 2: Get labels from any of the DataFrames (assuming the label is the same for all DataFrames)
        labels = dataframes[0][label_col].values
        
        # Step 3: Data Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

        # Step 4: Feature Scaling
        scaler = MinMaxScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)



    def predict_multi_data(self, new_dataframes: list, feature_cols: list = ["high"], steps:int = 0) -> np.ndarray:
        """
        Predict the likelihood of spikes in new, unseen multi-data.

        Parameters:
            new_dataframes (list): List of DataFrames with the same feature columns as the training data.

        Returns:
            np.ndarray: Predicted spike probabilities for the new data.
        """
        # Process new data
        new_features_list = []

        for new_data in new_dataframes:
            new_data["time"] = pd.to_datetime(new_data["time"])
            new_data.set_index("time", inplace=True)

            # Extract features from the new data.
            new_features = new_data[feature_cols].values

            # Scale the new features
            scaler = MinMaxScaler() 
            new_features_scaled = scaler.fit_transform(new_features)

            # Check if the number of time steps matches the model's input shape (6 time steps in this case)
            num_time_steps = steps or new_features_scaled.shape[0]
            if new_features_scaled.shape[0] < num_time_steps:
                raise ValueError(f"Input data must have at least {num_time_steps} time steps.")

            # Reshape the new_features_scaled to match the input shape of the LSTM model
            new_features_reshaped = np.expand_dims(new_features_scaled[-num_time_steps:], axis=0)

            new_features_list.append(new_features_reshaped)

        # Combine all the new_features_reshaped into a single array
        new_features_combined = np.vstack(new_features_list)

        # Predict spike probabilities.
        predicted_probabilities = self.model.predict(new_features_combined)

        return predicted_probabilities



        # Step 10: Monitoring and Refinement
        # Continuously monitor the model's performance and adjust the architecture and hyperparameters as needed.

'''------------------------------------'''
'''------------------------------------'''
'''------------------------------------'''
'''------------------------------------'''

