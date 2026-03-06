import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=1, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        # Adding some dense layers for better feature extraction before final output
        self.fc1 = nn.Linear(hidden_layer_size, 64)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        x = self.fc1(lstm_out[:, -1, :]) # Take the output from the last time step
        x = self.relu(x)
        x = self.dropout_layer(x)
        predictions = self.fc2(x)
        return self.sigmoid(predictions)


class LSTMTrader:
    def __init__(self, sequence_length=20, features=None):
        self.sequence_length = sequence_length
        self.features = features if features is not None else ['Close']
        
        self.models = [] # List to hold ensemble models
        self.num_models = 10
        self.scaler = StandardScaler()
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def save_model(self, base_path='models/'):
        import os
        import pickle
        os.makedirs(base_path, exist_ok=True)
        # Save scaler
        with open(os.path.join(base_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save each model in the ensemble
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(base_path, f'lstm_weights_ensemble_{i}.pth'))
        print(f"Ensemble of {len(self.models)} models saved to {base_path}")
        
    def load_model(self, base_path='models/'):
        import os
        import pickle
        scaler_path = os.path.join(base_path, 'scaler.pkl')
        
        if not os.path.exists(scaler_path):
            return False
            
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.models = []
        for i in range(self.num_models):
            model_path = os.path.join(base_path, f'lstm_weights_ensemble_{i}.pth')
            if os.path.exists(model_path):
                model = LSTMModel(input_size=len(self.features)).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.models.append(model)
            else:
                print(f"Warning: Missing ensemble model {i} at {model_path}")
                return False
                
        self.is_trained = True
        print(f"Ensemble of {len(self.models)} models loaded from {base_path}")
        return True
        
    def prepare_data(self, df: pd.DataFrame, is_training=True):
        data = df[self.features].copy()
        
        if is_training:
            # Predict a 0.2% gain within the next day
            returns = df['Close'].pct_change().shift(-1)
            df['Target'] = (returns > 0.002).astype(int)
            targets = df['Target'].values[:-1] 
            data = data.iloc[:-1]
        
        if is_training:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
            
        X = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            X.append(scaled_data[i:(i + self.sequence_length)])
            
        X = np.array(X)
        
        if is_training:
            y = targets[self.sequence_length - 1:]
            return torch.FloatTensor(X.copy()).to(self.device), torch.FloatTensor(y.copy()).unsqueeze(1).to(self.device)
        
        # If not enough data for even one sequence, return empty tensor
        if len(X) == 0:
            return torch.FloatTensor([]).to(self.device)
            
        return torch.FloatTensor(X.copy()).to(self.device)

    def train(self, df: pd.DataFrame, epochs=50, batch_size=64, learning_rate=0.0005):
        print("Preparing data for training...")
        X, y = self.prepare_data(df, is_training=True)
        
        num_pos = y.sum().item()
        num_neg = len(y) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(self.device)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCELoss(weight=pos_weight) 
        
        self.models = []
        for i in range(self.num_models):
            print(f"\n--- Training Ensemble Model {i+1}/{self.num_models} ---")
            # Create a new model for each ensemble member without setting a fixed random seed
            model = LSTMModel(input_size=len(self.features)).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    y_pred = model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                scheduler.step(avg_loss)
                
            self.models.append(model)
                
        self.is_trained = True
        print("\nEnsemble Training complete.")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained or not self.models:
            raise ValueError("Ensemble is not trained or loaded yet!")
            
        df_eval = df.copy()
        X = self.prepare_data(df_eval, is_training=False)
        
        if len(X) == 0:
            df_eval['signal'] = 0.0
            df_eval['Position'] = np.nan
            df_eval['confidence'] = 0.0
            return df_eval
            
        all_predictions = []
        
        # 從每個模型獲取預測，然後平均以形成最終的預測概率
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds = model(X).cpu().numpy().flatten()
            
            padding = np.zeros(self.sequence_length - 1)
            padding[:] = np.nan
            full_predictions = np.concatenate([padding, preds])
            all_predictions.append(full_predictions)

        ensemble_preds = np.array(all_predictions)
        
        # 每天的平均預測概率
        df_eval['lstm_prob'] = np.nanmean(ensemble_preds, axis=0)
        df_eval['lstm_prob_smooth'] = df_eval['lstm_prob'].rolling(window=3).mean()
        
        # 根據訓練期間的預測概率分佈動態設定買賣閾值
        prob_75 = np.nanpercentile(df_eval['lstm_prob_smooth'], 75)
        prob_25 = np.nanpercentile(df_eval['lstm_prob_smooth'], 25)
        median_prob = np.nanmedian(df_eval['lstm_prob_smooth'])
        
        prob_75 = median_prob + (prob_75 - median_prob) * 0.5 
        prob_25 = median_prob - (median_prob - prob_25) * 0.5 
        
        if prob_75 - prob_25 < 0.05:
            mean_prob = np.nanmean(df_eval['lstm_prob_smooth'])
            prob_75 = mean_prob + 0.005
            prob_25 = mean_prob - 0.005
            
        df_eval['signal'] = 0.0
        df_eval.loc[df_eval['lstm_prob_smooth'] > prob_75, 'signal'] = 1.0  # Buy
        df_eval.loc[df_eval['lstm_prob_smooth'] < prob_25, 'signal'] = 0.0  # Sell
        
        df_eval['confidence'] = df_eval['lstm_prob_smooth']
        
        # Forward fill signals so we hold until a clear sell signal
        signals = []
        current_signal = 0.0
        for i in range(len(df_eval)):
            if pd.isna(df_eval['lstm_prob'].iloc[i]):
                signals.append(np.nan)
            elif df_eval['lstm_prob_smooth'].iloc[i] > prob_75:
                current_signal = 1.0
                signals.append(current_signal)
            elif df_eval['lstm_prob_smooth'].iloc[i] < prob_25:
                current_signal = 0.0
                signals.append(current_signal)
            else:
                signals.append(current_signal)
                
        df_eval['signal'] = signals
        df_eval['Position'] = df_eval['signal'].diff()
        
        first_valid_idx = df_eval['signal'].first_valid_index()
        if first_valid_idx is not None:
            df_eval.loc[first_valid_idx, 'Position'] = df_eval.loc[first_valid_idx, 'signal']
            
        # Position 代表「根據第 T 日資料，預測 T+1 日的動作」。
        # 在回測時 (test.py / multi.py) 必須自行執行 df_scored['Position'].shift(1) 避免前瞻偏誤。
        # 在實盤時 (main.py) 直接讀取最後一行的 Position 即可作為隔日下單依據。
            
        return df_eval

def lstm_strategy(row, current_position, current_capital, current_price):
    action = None
    quantity = 0
    
    if row.get('Position') == 1.0:
        action = 'BUY'
        # 根據信心度調整買入量，最低 50% 的資金用於買入，最高不超過 95%
        confidence = row.get('confidence', 0.5)
        alloc_pct = min(max(confidence, 0.5), 0.95)
        quantity = int((current_capital * alloc_pct) // current_price)
    elif row.get('Position') == -1.0:
        action = 'SELL'
        quantity = current_position
        
    return action, quantity