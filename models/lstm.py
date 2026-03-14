import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BASE_DIR, NUM_ENSEMBLE_MODELS

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=1, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_layer_size, 64)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
        # 移除 sigmoid，改用 BCEWithLogitsLoss 直接輸出 logits

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        x = self.fc1(lstm_out[:, -1, :])
        x = self.relu(x)
        x = self.dropout_layer(x)
        logits = self.fc2(x)
        return logits


class LSTMTrader:
    def __init__(self, sequence_length=20, features=None):
        self.sequence_length = sequence_length
        self.features = features if features is not None else ['Close']

        self.models = []
        self.num_models = NUM_ENSEMBLE_MODELS
        self.scaler = StandardScaler()
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        print(f"Using device: {self.device}")

    def save_model(self, base_path=None):
        if base_path is None:
            base_path = os.path.join(BASE_DIR, 'models')
        os.makedirs(base_path, exist_ok=True)
        with open(os.path.join(base_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(base_path, f'lstm_weights_ensemble_{i}.pth'))
        print(f"Ensemble of {len(self.models)} models saved to {base_path}")

    def load_model(self, base_path=None):
        if base_path is None:
            base_path = os.path.join(BASE_DIR, 'models')
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

    def prepare_data(self, df: pd.DataFrame, is_training=True, fit_end_idx: int = None):
        """
        準備 LSTM 序列資料。
        不會修改傳入的 df，避免副作用。

        Args:
            fit_end_idx: 若提供，scaler 只用 data[:fit_end_idx] 做 fit，
                         避免訓練時將驗證集的統計資訊洩漏給 scaler。
        """
        df_work = df.copy()
        data = df_work[self.features].copy()

        if is_training:
            # Predict a 0.2% gain within the next day
            returns = df_work['Close'].pct_change().shift(-1)
            targets_series = (returns > 0.002).astype(int)
            # 最後一行的 shift(-1) 為 NaN，排除
            targets = targets_series.values[:-1]
            data = data.iloc[:-1]

        if is_training:
            if fit_end_idx is not None and fit_end_idx > 0:
                # 只用訓練集部分做 fit，避免資料洩漏到驗證集
                self.scaler.fit(data.iloc[:fit_end_idx])
            else:
                self.scaler.fit(data)
            scaled_data = self.scaler.transform(data)
        else:
            scaled_data = self.scaler.transform(data)

        X = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            X.append(scaled_data[i:(i + self.sequence_length)])

        X = np.array(X)

        if is_training:
            y = targets[self.sequence_length - 1:]
            return torch.FloatTensor(X.copy()).to(self.device), torch.FloatTensor(y.copy()).unsqueeze(1).to(self.device)

        if len(X) == 0:
            return torch.FloatTensor([]).to(self.device)

        return torch.FloatTensor(X.copy()).to(self.device)

    def _train_single_model(self, model_idx, X_train, y_train, X_val, y_val, 
                           train_dataloader, val_dataloader, criterion, 
                           epochs, learning_rate):
        """訓練單個模型 (用於平行或順序訓練)"""
        print(f"\n--- Training Ensemble Model {model_idx+1}/{self.num_models} ---")
        model = LSTMModel(input_size=len(self.features)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        model_train_losses = []
        model_val_losses = []
        
        best_val_loss = float('inf')
        best_state_dict = None
        patience_counter = 0
        early_stop_patience = 7

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0

            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    y_pred = model(batch_X)
                    val_loss = criterion(y_pred, batch_y)
                    epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0

            scheduler.step(avg_val_loss)
            model_train_losses.append(avg_train_loss)
            model_val_losses.append(avg_val_loss)

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) == epochs or (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break

        # 載入 best model
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return model, model_train_losses, model_val_losses

    def train(self, df: pd.DataFrame, epochs=50, batch_size=64, learning_rate=0.0005, parallel=False):
        print("Preparing data for training...")

        # 先計算 80% 切分點（以原始資料列數為基準），
        # 讓 scaler 只用訓練集部分做 fit，避免驗證集資訊洩漏。
        n_rows = len(df) - 1  # prepare_data 會去掉最後一行
        raw_split = int(n_rows * 0.8)
        X, y = self.prepare_data(df, is_training=True, fit_end_idx=raw_split)

        # Split into training and validation sets (80/20 split)
        split_idx = int(len(X) * 0.8)
        if split_idx == 0 or split_idx == len(X):
            X_train, y_train = X, y
            X_val, y_val = X, y
            print("Warning: Dataset too small for validation split. Using same data for train and val.")
        else:
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]

        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        num_pos = y_train.sum().item()
        num_neg = len(y_train) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 使用 BCEWithLogitsLoss + pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.models = []
        all_train_losses = []
        all_val_losses = []
        
        # 平行訓練模式
        if parallel:
            print(f"Using PARALLEL training mode for {self.num_models} models...")
            with ThreadPoolExecutor(max_workers=self.num_models) as executor:
                futures = {}
                for i in range(self.num_models):
                    future = executor.submit(self._train_single_model, i, X_train, y_train, 
                                           X_val, y_val, train_dataloader, val_dataloader, 
                                           criterion, epochs, learning_rate)
                    futures[future] = i
                
                # 收集結果並按索引排序
                results = {}
                for future in as_completed(futures):
                    model_idx = futures[future]
                    model, train_losses, val_losses = future.result()
                    results[model_idx] = (model, train_losses, val_losses)
                
                # 按照索引順序重新排列
                for i in range(self.num_models):
                    model, train_losses, val_losses = results[i]
                    self.models.append(model)
                    all_train_losses.append(train_losses)
                    all_val_losses.append(val_losses)
            
        else:
            # 順序訓練模式
            print(f"Using SEQUENTIAL training mode for {self.num_models} models...")
            for i in range(self.num_models):
                model, train_losses, val_losses = self._train_single_model(
                    i, X_train, y_train, X_val, y_val, 
                    train_dataloader, val_dataloader, criterion, epochs, learning_rate)
                self.models.append(model)
                all_train_losses.append(train_losses)
                all_val_losses.append(val_losses)

        self.is_trained = True
        print("\nEnsemble Training complete.")

        # plot training and validation loss
        max_epochs_trained = max(len(l) for l in all_train_losses)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        for i, losses in enumerate(all_train_losses):
            ax1.plot(range(1, len(losses) + 1), losses, label=f'Model {i+1}')
        ax1.set_title('Training Loss')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        for i, losses in enumerate(all_val_losses):
            ax2.plot(range(1, len(losses) + 1), losses, label=f'Model {i+1}', linestyle='--')
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        if self.num_models <= 10:
            ax1.legend(loc='best')
            ax2.legend(loc='best')

        plt.tight_layout()
        loss_chart_path = os.path.join(BASE_DIR, 'models', 'training_loss.png')
        plt.savefig(loss_chart_path)
        print(f"Training & Validation loss chart saved to {loss_chart_path}")
        plt.close()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained or not self.models:
            raise ValueError("Ensemble is not trained or loaded yet!")

        df_eval = df.copy()
        X = self.prepare_data(df_eval, is_training=False)

        if len(X) == 0:
            df_eval['signal'] = 0.0
            df_eval['Position'] = np.nan
            df_eval['confidence'] = 0.0
            df_eval['lstm_prob'] = np.nan
            df_eval['lstm_prob_smooth'] = np.nan
            return df_eval

        all_predictions = []

        # 從每個模型獲取預測，model 輸出 logits，需 sigmoid 轉成機率
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X).cpu().numpy().flatten()
                preds = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

            padding = np.zeros(self.sequence_length - 1)
            padding[:] = np.nan
            full_predictions = np.concatenate([padding, preds])
            all_predictions.append(full_predictions)

        ensemble_preds = np.array(all_predictions)

        # 每天的平均預測概率
        df_eval['lstm_prob'] = np.nanmean(ensemble_preds, axis=0)
        df_eval['lstm_prob_smooth'] = df_eval['lstm_prob'].rolling(window=3).mean()

        # 使用 expanding window 計算動態閾值，避免 look-ahead bias
        # 每天只用「截至當天為止」的歷史概率來計算分位數
        prob_smooth_values = df_eval['lstm_prob_smooth'].values
        min_window = self.sequence_length + 3  # 至少需要足夠樣本才能計算有意義的分位數

        # Forward fill signals: 1.0=BUY, -1.0=SELL, 0.0=HOLD
        signals = []
        current_signal = 0.0
        for i in range(len(df_eval)):
            prob = prob_smooth_values[i]
            if pd.isna(df_eval['lstm_prob'].iloc[i]):
                signals.append(np.nan)
                continue

            if pd.isna(prob):
                signals.append(current_signal)
                continue

            # 只用截至第 i 天（含）的歷史數據計算閾值
            historical_probs = prob_smooth_values[:i+1]
            valid_probs = historical_probs[~np.isnan(historical_probs)]

            if len(valid_probs) < min_window:
                # 樣本不足時不產生信號
                signals.append(current_signal)
                continue

            raw_75 = np.percentile(valid_probs, 75)
            raw_25 = np.percentile(valid_probs, 25)
            median_prob = np.median(valid_probs)

            prob_75 = median_prob + (raw_75 - median_prob) * 0.5
            prob_25 = median_prob - (median_prob - raw_25) * 0.5

            if prob_75 - prob_25 < 0.05:
                mean_prob = np.mean(valid_probs)
                prob_75 = mean_prob + 0.025
                prob_25 = mean_prob - 0.025

            if prob > prob_75:
                current_signal = 1.0
                signals.append(current_signal)
            elif prob < prob_25:
                current_signal = -1.0
                signals.append(current_signal)
            else:
                signals.append(current_signal)

        df_eval['signal'] = signals
        df_eval['confidence'] = df_eval['lstm_prob_smooth']

        # Position: 信號變化點才觸發動作
        # 使用 sign 化的 diff，確保跨越信號（如 -1→1）也能正確觸發
        raw_diff = df_eval['signal'].diff()
        df_eval['Position'] = np.sign(raw_diff)

        first_valid_idx = df_eval['signal'].first_valid_index()
        if first_valid_idx is not None:
            first_signal = df_eval.loc[first_valid_idx, 'signal']
            if first_signal != 0.0:
                df_eval.loc[first_valid_idx, 'Position'] = np.sign(first_signal)

        # Position 代表「根據第 T 日資料，預測 T+1 日的動作」。
        # 在回測時 (test.py / multi.py) 必須自行執行 df_scored['Position'].shift(1) 避免前瞻偏誤。
        # 在實盤時 (main.py) 直接讀取最後一行的 Position 即可作為隔日下單依據。

        return df_eval


def lstm_strategy(row, current_position, current_capital, current_price):
    action = None
    quantity = 0

    position = row.get('Position')
    # Position 可能為 NaN，需要先檢查
    if pd.isna(position):
        return action, quantity

    if position == 1.0:
        if current_position < 0:
            # 持有空單時收到做多信號，先回補空單
            action = 'COVER'
            quantity = abs(current_position)
        elif current_capital > current_price:
            action = 'BUY'
            confidence = row.get('confidence', 0.5)
            alloc_pct = min(max(confidence, 0.5), 0.95)
            quantity = int((current_capital * alloc_pct) // current_price)
    elif position == -1.0:
        if current_position > 0:
            # 持有多單時收到做空信號，先賣出平倉
            action = 'SELL'
            quantity = current_position

    return action, quantity
