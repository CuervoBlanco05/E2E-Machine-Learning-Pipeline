# End-to-End Machine Learning & Credit Portfolio Optimization Pipeline

## Executive Summary
This repository implements an end-to-end Machine Learning pipeline designed to evaluate credit portfolio recovery efficiency. Utilizing **Polars** for ultra-fast, multi-threaded memory management (optimized for 8 CPU threads) and **Scikit-Learn** for predictive decision tree modeling, the system processes multi-million row datasets to discover actionable operational collection rules.

## Key Features & Architecture

```text
E2E-Machine-Learning-Pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── synthetic_credit_data.csv  <-- Reproducible synthetic dataset
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
└── src/
    ├── generate_synthetic_data.py <-- Synthetic dataset generator
    └── data_pipeline.py           <-- Polars data cleaning & ML pipeline

## Quick Start Guide

1. **Install Dependencies:**
   ```bash
   py -m pip install -r requirements.txt

