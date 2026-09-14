import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

# Fijar límite de hilos para Polars (Estabilidad en ThinkPad)
os.environ["POLARS_MAX_THREADS"] = "8"

class DebtPortfolioPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

    def load_and_clean_data(self) -> pl.DataFrame:
        """Carga el dataset optimizado con Polars y elimina duplicados y columnas irrelevantes."""
        print(f"Cargando dataset desde {self.data_path}...")
        
        # Lectura perezosa/eficiente con Polars
        if self.data_path.endswith(".parquet"):
            self.df = pl.read_parquet(self.data_path)
        else:
            self.df = pl.read_csv(self.data_path)

        # Limpieza y deduplicación
        print(f"Registros iniciales: {self.df.height:,}")
        self.df = self.df.unique()
        
        # Eliminación de columnas no relevantes o redundantes
        cols_to_drop = [c for c in ["situacionespecial", "cteagregado", "fecha_", "mes"] if c in self.df.columns]
        self.df = self.df.drop(cols_to_drop)
        
        print(f"Registros tras limpieza de duplicados: {self.df.height:,}")
        return self.df

    def get_correlation_matrix(self) -> pl.DataFrame:
        """Calcula la matriz de correlación con respecto a la variable objetivo (eficiencia)."""
        numeric_df = self.df.select(pl.col(pl.NUMERIC_DTYPES))
        corr_matrix = numeric_df.corr()
        
        num_cols = numeric_df.columns
        corr_eficiencia = (
            corr_matrix
            .with_columns(pl.Series("variable", num_cols))
            .select(["variable", "eficiencia"])
            .sort("eficiencia", descending=True)
        )
        return corr_eficiencia

    def train_decision_tree(self, max_depth: int = 4):
        """Entrena un Regresor basado en Árboles de Decisión para extraer reglas operativas."""
        # Convertir a matriz NumPy para Scikit-Learn
        features = [col for col in self.df.columns if col != "eficiencia"]
        
        X = self.df.select(features).to_numpy()
        y = self.df.select("eficiencia").to_numpy().ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree.fit(X_train, y_train)

        # Reglas extraídas del árbol
        rules = export_text(tree, feature_names=features)
        
        importances = dict(zip(features, tree.feature_importances_))
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return tree, rules, sorted_importances

if __name__ == "__main__":
    # Prueba del pipeline usando datos sintéticos locales
    data_file = "data/synthetic_credit_data.csv"
    if os.path.exists(data_file):
        pipeline = DebtPortfolioPipeline(data_file)
        df_clean = pipeline.load_and_clean_data()
        corr = pipeline.get_correlation_matrix()
        print("\n--- Correlación con Eficiencia ---")
        print(corr)
        
        _, rules, importances = pipeline.train_decision_tree()
        print("\n--- Top Variables Importantes ---")
        for var, imp in importances[:5]:
            print(f"{var}: {imp:.4f}")
            
        print("\n--- Reglas del Árbol de Decisión ---")
        print(rules)
