import os
import numpy as np
import polars as pl

# Configurar hilos en Polars para estabilidad en ThinkPad (8 hilos)
os.environ["POLARS_MAX_THREADS"] = "8"

def generate_credit_dataset(n_samples: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Genera un dataset sintético para análisis de riesgo crediticio y cobranza."""
    np.random.seed(seed)
    
    customer_ids = np.arange(10000, 10000 + n_samples)
    days_past_due = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=n_samples, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
    monthly_income = np.round(np.random.gamma(shape=3.0, scale=3000.0, size=n_samples) + 2000, 2)
    outstanding_balance = np.round(monthly_income * np.random.uniform(0.05, 0.85, size=n_samples), 2)
    customer_age = np.random.randint(18, 75, size=n_samples)
    contact_score = np.random.randint(0, 101, size=n_samples)
    payment_score = np.random.randint(0, 101, size=n_samples)
    
    # Simulación de probabilidad de pago basada en variables clave
    logit = (
        -0.03 * days_past_due 
        + 0.02 * contact_score 
        + 0.03 * payment_score 
        - 0.0001 * (outstanding_balance / monthly_income)
        - 1.5
    )
    prob_recovery = 1 / (1 + np.exp(-logit))
    recovered_flag = np.random.binomial(1, prob_recovery)
    
    df = pl.DataFrame({
        "customer_id": customer_ids,
        "days_past_due": days_past_due,
        "monthly_income": monthly_income,
        "outstanding_balance": outstanding_balance,
        "customer_age": customer_age,
        "contact_score": contact_score,
        "payment_score": payment_score,
        "recovered_flag": recovered_flag
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df_synthetic = generate_credit_dataset()
    df_synthetic.write_csv("data/synthetic_credit_data.csv")
    print("Dataset sintético generado con éxito en 'data/synthetic_credit_data.csv'.")
