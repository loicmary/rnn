import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class LSTMForecastWithExog:
    def __init__(self, lookback_days=7, forecast_horizon=24, lstm_units=64, dropout_rate=0.2):
        """
        Initialise le modèle LSTM avec variables exogènes
        
        Args:
            lookback_days: Nombre de jours d'historique à utiliser (7 jours)
            forecast_horizon: Nombre d'heures à prédire (24 heures)
            lstm_units: Nombre d'unités LSTM
            dropout_rate: Taux de dropout
        """
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.lookback_hours = lookback_days * 24
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # Scalers pour normaliser les données
        self.scaler_y = MinMaxScaler()
        self.scaler_x1 = MinMaxScaler()
        self.scaler_x2 = MinMaxScaler()
        
        self.model = None
        self.history = None
    
    def create_sequences(self, y, x1, x2):
        """
        Crée les séquences d'entraînement
        
        Args:
            y: Série temporelle cible (target)
            x1, x2: Variables exogènes
            
        Returns:
            X_y: Séquences historiques de y (lookback_hours)
            X_x1: Séquences de x1 (lookback_hours + forecast_horizon)
            X_x2: Séquences de x2 (lookback_hours + forecast_horizon)
            y_target: Valeurs cibles (forecast_horizon)
        """
        X_y, X_x1, X_x2, y_target = [], [], [], []
        
        # Pour chaque séquence possible
        for i in range(len(y) - self.lookback_hours - self.forecast_horizon + 1):
            # Séquence historique de y (7 jours = 168 heures)
            seq_y = y[i:i + self.lookback_hours]
            
            # Séquences de variables exogènes (7 jours + 1 jour futur = 192 heures)
            seq_x1 = x1[i:i + self.lookback_hours + self.forecast_horizon]
            seq_x2 = x2[i:i + self.lookback_hours + self.forecast_horizon]
            
            # Valeurs cibles (24 heures suivantes)
            target = y[i + self.lookback_hours:i + self.lookback_hours + self.forecast_horizon]
            
            X_y.append(seq_y)
            X_x1.append(seq_x1)
            X_x2.append(seq_x2)
            y_target.append(target)
        
        return np.array(X_y), np.array(X_x1), np.array(X_x2), np.array(y_target)
    
    def build_model(self):
        """
        Construit le modèle LSTM avec variables exogènes
        """
        # Input pour la série temporelle historique (y)
        input_y = Input(shape=(self.lookback_hours, 1), name='input_y')
        
        # Input pour les variables exogènes (x1, x2) - incluent le futur
        input_x1 = Input(shape=(self.lookback_hours + self.forecast_horizon, 1), name='input_x1')
        input_x2 = Input(shape=(self.lookback_hours + self.forecast_horizon, 1), name='input_x2')
        
        # LSTM pour la série temporelle principale
        lstm_y = LSTM(self.lstm_units, return_sequences=True, name='lstm_y_1')(input_y)
        lstm_y = Dropout(self.dropout_rate)(lstm_y)
        lstm_y = LSTM(self.lstm_units // 2, return_sequences=False, name='lstm_y_2')(lstm_y)
        lstm_y = Dropout(self.dropout_rate)(lstm_y)
        
        # LSTM pour les variables exogènes
        # Concaténer x1 et x2
        exog_combined = Concatenate(axis=2)([input_x1, input_x2])
        
        lstm_exog = LSTM(self.lstm_units // 2, return_sequences=True, name='lstm_exog_1')(exog_combined)
        lstm_exog = Dropout(self.dropout_rate)(lstm_exog)
        lstm_exog = LSTM(self.lstm_units // 4, return_sequences=False, name='lstm_exog_2')(lstm_exog)
        lstm_exog = Dropout(self.dropout_rate)(lstm_exog)
        
        # Combiner les features de y et des variables exogènes
        combined = Concatenate()([lstm_y, lstm_exog])
        
        # Couches denses pour la prédiction
        dense = Dense(self.lstm_units, activation='relu', name='dense_1')(combined)
        dense = Dropout(self.dropout_rate)(dense)
        dense = Dense(self.lstm_units // 2, activation='relu', name='dense_2')(dense)
        dense = Dropout(self.dropout_rate)(dense)
        
        # Sortie finale (24 valeurs)
        output = Dense(self.forecast_horizon, activation='linear', name='output')(dense)
        
        # Créer le modèle
        self.model = Model(inputs=[input_y, input_x1, input_x2], outputs=output)
        
        # Compiler le modèle
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def prepare_data(self, df):
        """
        Prépare les données pour l'entraînement
        
        Args:
            df: DataFrame avec colonnes 'datetime', 'y', 'x1', 'x2'
        """
        # Assurer que les données sont triées par date
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Extraire les séries
        y = df['y'].values
        x1 = df['x1'].values
        x2 = df['x2'].values
        
        # Normaliser les données
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        x1_scaled = self.scaler_x1.fit_transform(x1.reshape(-1, 1)).flatten()
        x2_scaled = self.scaler_x2.fit_transform(x2.reshape(-1, 1)).flatten()
        
        # Créer les séquences
        X_y, X_x1, X_x2, y_target = self.create_sequences(y_scaled, x1_scaled, x2_scaled)
        
        return X_y, X_x1, X_x2, y_target
    
    def train(self, df, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """
        Entraîne le modèle
        
        Args:
            df: DataFrame avec les données
            validation_split: Pourcentage de données pour la validation
            epochs: Nombre d'époques
            batch_size: Taille des batches
            verbose: Verbosité
        """
        # Préparer les données
        X_y, X_x1, X_x2, y_target = self.prepare_data(df)
        
        print(f"Shape des données:")
        print(f"X_y: {X_y.shape}")
        print(f"X_x1: {X_x1.shape}")
        print(f"X_x2: {X_x2.shape}")
        print(f"y_target: {y_target.shape}")
        
        # Reshape pour LSTM
        X_y = X_y.reshape(X_y.shape[0], X_y.shape[1], 1)
        X_x1 = X_x1.reshape(X_x1.shape[0], X_x1.shape[1], 1)
        X_x2 = X_x2.reshape(X_x2.shape[0], X_x2.shape[1], 1)
        
        # Construire le modèle
        self.build_model()
        
        print("\nArchitecture du modèle:")
        self.model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Entraînement
        self.history = self.model.fit(
            [X_y, X_x1, X_x2], y_target,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, y_history, x1_future, x2_future):
        """
        Fait une prédiction
        
        Args:
            y_history: Historique de 7 jours de y (168 valeurs)
            x1_future: Valeurs de x1 pour 7 jours passés + 1 jour futur (192 valeurs)
            x2_future: Valeurs de x2 pour 7 jours passés + 1 jour futur (192 valeurs)
            
        Returns:
            Prédiction dénormalisée
        """
        # Normaliser les données d'entrée
        y_scaled = self.scaler_y.transform(y_history.reshape(-1, 1)).flatten()
        x1_scaled = self.scaler_x1.transform(x1_future.reshape(-1, 1)).flatten()
        x2_scaled = self.scaler_x2.transform(x2_future.reshape(-1, 1)).flatten()
        
        # Reshape pour le modèle
        X_y = y_scaled.reshape(1, -1, 1)
        X_x1 = x1_scaled.reshape(1, -1, 1)
        X_x2 = x2_scaled.reshape(1, -1, 1)
        
        # Prédiction
        prediction_scaled = self.model.predict([X_y, X_x1, X_x2], verbose=0)
        
        # Dénormaliser
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction.flatten()
    
    def plot_training_history(self):
        """
        Affiche l'historique d'entraînement
        """
        if self.history is None:
            print("Aucun historique d'entraînement trouvé.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Train MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Fonction pour générer des données d'exemple
def generate_sample_data(start_date='2023-01-01', end_date='2024-12-31'):
    """
    Génère des données d'exemple pour tester le modèle
    """
    # Créer une plage de dates horaires
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Générer des données synthétiques
    np.random.seed(42)
    n_points = len(date_range)
    
    # Série temporelle principale avec saisonnalité
    t = np.arange(n_points)
    y = (
        100 + 
        20 * np.sin(2 * np.pi * t / (24 * 7)) +  # Saisonnalité hebdomadaire
        10 * np.sin(2 * np.pi * t / 24) +         # Saisonnalité journalière
        5 * np.random.randn(n_points) +           # Bruit
        0.001 * t                                 # Tendance légère
    )
    
    # Variables exogènes corrélées
    x1 = (
        50 + 
        15 * np.sin(2 * np.pi * t / 24 + np.pi/4) +  # Déphasage par rapport à y
        3 * np.random.randn(n_points)
    )
    
    x2 = (
        30 + 
        10 * np.cos(2 * np.pi * t / (24 * 7)) +      # Saisonnalité hebdomadaire différente
        2 * np.random.randn(n_points)
    )
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'datetime': date_range,
        'y': y,
        'x1': x1,
        'x2': x2
    })
    
    return df

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données d'exemple
    print("Génération des données d'exemple...")
    df = generate_sample_data()
    print(f"Données générées: {len(df)} points de {df['datetime'].min()} à {df['datetime'].max()}")
    
    # Créer et entraîner le modèle
    print("\nCréation du modèle LSTM...")
    model = LSTMForecastWithExog(
        lookback_days=7,
        forecast_horizon=24,
        lstm_units=64,
        dropout_rate=0.2
    )
    
    print("Entraînement du modèle...")
    history = model.train(
        df=df,
        validation_split=0.2,
        epochs=50,  # Réduire pour l'exemple
        batch_size=32,
        verbose=1
    )
    
    # Afficher l'historique d'entraînement
    model.plot_training_history()
    
    # Exemple de prédiction
    print("\nExemple de prédiction...")
    
    # Prendre les dernières données pour faire une prédiction
    last_idx = len(df) - 24 - 1  # Garder 24h pour comparaison
    
    # Historique de y (7 jours)
    y_hist = df['y'].iloc[last_idx-168:last_idx].values
    
    # Variables exogènes (7 jours passés + 1 jour futur)
    x1_fut = df['x1'].iloc[last_idx-168:last_idx+24].values
    x2_fut = df['x2'].iloc[last_idx-168:last_idx+24].values
    
    # Prédiction
    prediction = model.predict(y_hist, x1_fut, x2_fut)
    
    # Valeurs réelles pour comparaison
    actual = df['y'].iloc[last_idx:last_idx+24].values
    
    # Calculer les métriques
    mse = mean_squared_error(actual, prediction)
    mae = mean_absolute_error(actual, prediction)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Visualiser la prédiction
    plt.figure(figsize=(12, 6))
    hours = range(24)
    plt.plot(hours, actual, label='Valeurs réelles', marker='o')
    plt.plot(hours, prediction, label='Prédictions', marker='s')
    plt.xlabel('Heure')
    plt.ylabel('Valeur')
    plt.title('Prédiction vs Réalité (24h)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nModèle entraîné avec succès!")
    print("Vous pouvez maintenant utiliser model.predict() pour faire des prédictions.")
