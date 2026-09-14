import os
import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

# Asignación de 8 hilos para Polars en ThinkPad
os.environ["POLARS_MAX_THREADS"] = "8"

class DebtPortfolioPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

    def load_and_clean_data(self) -> pl.DataFrame:
        """
        Carga la cartera de crédito desde CSV/Parquet y ejecuta la limpieza en Polars:
        - Deduplicación
        - Remoción de columnas irrelevantes
        - Filtrado del rango válido de eficiencia (0 a 100)
        """
        print(f"Cargando dataset con Polars desde {self.data_path}...")
        if self.data_path.endswith(".parquet"):
            self.df = pl.read_parquet(self.data_path)
        else:
            self.df = pl.read_csv(self.data_path)

        print(f"Registros crudos: {self.df.height:,}")

        # 1. Deduplicación eficiente
        self.df = self.df.unique()

        # 2. Eliminación de variables no predictivas o redundantes
        cols_drop = [c for c in ["situacionespecial", "cteagregado", "fecha_", "mes"] if c in self.df.columns]
        self.df = self.df.drop(cols_drop)

        # 3. Casteo de tipos y filtrado del rango objetivo válido [0, 100]
        if "edad" in self.df.columns:
            self.df = self.df.with_columns(pl.col("edad").cast(pl.Int64))

        self.df = self.df.filter(pl.col("eficiencia").is_between(0, 100))

        # 4. Creación de variable binaria objetivo (> 70% eficiencia)
        self.df = self.df.with_columns(
            (pl.col("eficiencia") > 70).cast(pl.Int32).alias("target_bueno")
        )

        print(f"Registros limpios en rango [0, 100]: {self.df.height:,}")
        return self.df

    def get_correlation_matrix() -> pl.DataFrame:
        """Calcula la matriz de correlación de Pearson sobre variables numéricas."""
        numeric_df = self.df.select(cs.numeric())
        num_cols = numeric_df.columns
        corr_matrix = numeric_df.corr()

        corr_eficiencia = (
            corr_matrix
            .with_columns(pl.Series("variable", num_cols))
            .select(["variable", "eficiencia"])
            .sort("eficiencia", descending=True)
        )
        return corr_eficiencia

    def train_decision_tree(self, max_depth: int = 4):
        """Entrena el Árbol de Decisión convirtiendo las estructuras de Polars a NumPy."""
        exclude_cols = ["eficiencia", "target_bueno", "idcte"]
        features = [col for col in self.df.columns if col not in exclude_cols]

        X = self.df.select(features).to_numpy()
        y = self.df.select("eficiencia").to_numpy().ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree.fit(X_train, y_train)

        rules = export_text(tree, feature_names=features)
        importances = dict(zip(features, tree.feature_importances_))
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)

        return tree, rules, sorted_importances


if __name__ == "__main__":
    data_file = "data/synthetic_credit_data.csv"
    if os.path.exists(data_file):
        pipeline = DebtPortfolioPipeline(data_file)
        df_clean = pipeline.load_and_clean_data()
        
        print("\n--- Correlación Ordenada con Eficiencia ---")
        print(pipeline.get_correlation_matrix())

        _, rules, importances = pipeline.train_decision_tree()
        print("\n--- Top Variables Importantes ---")
        for var, imp in importances[:5]:
            print(f"{var}: {imp:.4f}")

        print("\n--- Reglas del Árbol de Decisión ---")
        print(rules)
