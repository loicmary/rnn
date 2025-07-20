import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_X_y


class MedianMADScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Scaler robuste utilisant la médiane et la MAD (Median Absolute Deviation).
    
    La transformation appliquée est : (x - median(X)) / mad(X)
    La transformation inverse est : x * mad(X) + median(X)
    
    Cette classe est compatible avec les pipelines sklearn et TransformedTargetRegressor.
    
    Parameters
    ----------
    copy : bool, default=True
        Si True, une copie de X est créée. Si False, la normalisation in-place 
        est effectuée si possible.
    
    Attributes
    ----------
    median_ : ndarray of shape (n_features,)
        Médiane par feature calculée sur les données d'entraînement.
        
    mad_ : ndarray of shape (n_features,)
        MAD (Median Absolute Deviation) par feature calculée sur les données d'entraînement.
        
    n_features_in_ : int
        Nombre de features vues pendant fit.
        
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Noms des features vus pendant fit (seulement si X a des noms de features).
    """
    
    def __init__(self, copy=True):
        self.copy = copy
    
    def _calculate_mad(self, X, median):
        """Calcule la MAD (Median Absolute Deviation)"""
        return np.median(np.abs(X - median), axis=0)
    
    def fit(self, X, y=None):
        """
        Calcule la médiane et la MAD pour chaque feature.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Les données d'entraînement.
        y : array-like of shape (n_samples,), default=None
            Ignoré. Ce paramètre existe pour la compatibilité API.
            
        Returns
        -------
        self : object
            Retourne l'instance elle-même.
        """
        # Stocker les informations sur les feature names AVANT check_array
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        
        # Validation des entrées
        X = check_array(X, accept_sparse=False, dtype=np.float64, 
                       ensure_2d=True, allow_nd=False)
        
        # Stockage des informations sur les features
        self.n_features_in_ = X.shape[1]
        
        # Calcul de la médiane et MAD
        self.median_ = np.median(X, axis=0)
        self.mad_ = self._calculate_mad(X, self.median_)
        
        # Gérer le cas où MAD = 0 (variance nulle)
        # On remplace par 1 pour éviter la division par zéro
        self.mad_ = np.where(self.mad_ == 0, 1, self.mad_)
        
        return self
    
    def transform(self, X):
        """
        Applique la transformation (x - median) / mad.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Les données à transformer.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Les données transformées.
        """
        check_is_fitted(self)
        
        # Stocker le type d'entrée pour potentiel retour en DataFrame
        input_is_dataframe = hasattr(X, 'columns')
        original_columns = None
        original_index = None
        
        if input_is_dataframe:
            original_columns = X.columns
            original_index = X.index
        
        X = check_array(X, accept_sparse=False, dtype=np.float64,
                       ensure_2d=True, allow_nd=False, copy=self.copy)
        
        # Vérification de la cohérence du nombre de features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X a {X.shape[1]} features, mais {self.__class__.__name__} "
                           f"attend {self.n_features_in_} features comme vu dans fit.")
        
        # Application de la transformation
        X_transformed = (X - self.median_) / self.mad_
        
        # Si l'entrée était un DataFrame et qu'on a les noms de colonnes, 
        # on peut retourner un DataFrame (optionnel)
        if input_is_dataframe and hasattr(self, 'feature_names_in_'):
            import pandas as pd
            X_transformed = pd.DataFrame(
                X_transformed, 
                columns=self.feature_names_in_,
                index=original_index
            )
        
        return X_transformed
    
    def inverse_transform(self, X):
        """
        Applique la transformation inverse x * mad + median.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Les données transformées à inverser.
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Les données dans l'espace original.
        """
        check_is_fitted(self)
        
        # Stocker le type d'entrée pour potentiel retour en DataFrame
        input_is_dataframe = hasattr(X, 'columns')
        original_columns = None
        original_index = None
        
        if input_is_dataframe:
            original_columns = X.columns
            original_index = X.index
        
        X = check_array(X, accept_sparse=False, dtype=np.float64,
                       ensure_2d=True, allow_nd=False, copy=self.copy)
        
        # Vérification de la cohérence du nombre de features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X a {X.shape[1]} features, mais {self.__class__.__name__} "
                           f"attend {self.n_features_in_} features comme vu dans fit.")
        
        # Application de la transformation inverse
        X_original = X * self.mad_ + self.median_
        
        # Si l'entrée était un DataFrame et qu'on a les noms de colonnes, 
        # on peut retourner un DataFrame (optionnel)
        if input_is_dataframe and hasattr(self, 'feature_names_in_'):
            import pandas as pd
            X_original = pd.DataFrame(
                X_original, 
                columns=self.feature_names_in_,
                index=original_index
            )
        
        return X_original
    
    def _more_tags(self):
        """Tags additionnels pour sklearn"""
        return {
            'requires_fit': True,
            'requires_positive_X': False,
            'requires_y': False,
            'allow_nan': False,
            'preserves_dtype': [np.float64, np.float32],
        }


# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import make_regression
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    # Génération de données d'exemple avec DataFrame
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
    
    # Conversion en DataFrame avec noms de colonnes
    X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    y_series = pd.Series(y, name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    print("=== Test avec DataFrames ===")
    scaler = MedianMADScaler()
    
    # Test avec DataFrame
    print(f"Type d'entrée: {type(X_train)}")
    print(f"Colonnes d'entrée: {list(X_train.columns)}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Type de sortie: {type(X_train_scaled)}")
    if hasattr(X_train_scaled, 'columns'):
        print(f"Colonnes de sortie: {list(X_train_scaled.columns)}")
    print(f"Feature names stockés: {scaler.feature_names_in_}")
    
    print(f"\nMédiane originale: {X_train.median().values}")
    if hasattr(X_train_scaled, 'median'):
        print(f"Médiane transformée: {X_train_scaled.median().values}")
    else:
        print(f"Médiane transformée: {np.median(X_train_scaled, axis=0)}")
    
    # Vérification de l'inverse
    X_train_inverse = scaler.inverse_transform(X_train_scaled)
    print(f"Type après inverse: {type(X_train_inverse)}")
    
    if hasattr(X_train_inverse, 'values'):
        diff = np.max(np.abs(X_train.values - X_train_inverse.values))
    else:
        diff = np.max(np.abs(X_train.values - X_train_inverse))
    print(f"Erreur reconstruction: {diff}")
    
    print("\n=== Test 1: Utilisation directe (numpy arrays) ===")
    scaler_np = MedianMADScaler()
    X_train_scaled_np = scaler_np.fit_transform(X_train.values)
    X_test_scaled_np = scaler_np.transform(X_test.values)
    
    print(f"Médiane originale: {np.median(X_train.values, axis=0)}")
    print(f"Médiane transformée: {np.median(X_train_scaled_np, axis=0)}")
    print(f"MAD stockée: {scaler_np.mad_}")
    
    # Vérification de l'inverse
    X_train_inverse_np = scaler_np.inverse_transform(X_train_scaled_np)
    print(f"Erreur reconstruction: {np.max(np.abs(X_train.values - X_train_inverse_np))}")
    
    print("\n=== Test 2: Dans un Pipeline avec DataFrame ===")
    pipeline = Pipeline([
        ('scaler', MedianMADScaler()),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred_pipeline = pipeline.predict(X_test)
    mse_pipeline = mean_squared_error(y_test, y_pred_pipeline)
    print(f"MSE avec pipeline (DataFrame): {mse_pipeline:.2f}")
    
    print("\n=== Test 3: Avec TransformedTargetRegressor ===")
    regressor_with_target_transform = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=MedianMADScaler()
    )
    
    regressor_with_target_transform.fit(X_train, y_train)
    y_pred_target = regressor_with_target_transform.predict(X_test)
    mse_target = mean_squared_error(y_test, y_pred_target)
    print(f"MSE avec TransformedTargetRegressor: {mse_target:.2f}")
    
    print("\n=== Test 4: Pipeline complet avec transformation des features ET du target ===")
    full_pipeline = Pipeline([
        ('feature_scaler', MedianMADScaler()),
        ('regressor', TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=MedianMADScaler()
        ))
    ])
    
    full_pipeline.fit(X_train, y_train)
    y_pred_full = full_pipeline.predict(X_test)
    mse_full = mean_squared_error(y_test, y_pred_full)
    print(f"MSE avec pipeline complet: {mse_full:.2f}")
