# 🚀 Modular Machine Learning Development Pipeline
 
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)

![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange?logo=xgboost)

![Neural Networks](https://img.shields.io/badge/Neural%20Networks-NN%2C%20RNN%2C%20LSTM-green?logo=tensorflow)

![License](https://img.shields.io/badge/License-MIT-yellow)

![Status](https://img.shields.io/badge/Status-Development-lightgrey)
 
> **A config-driven, modular machine learning pipeline** for development and experimentation.  
> Supports **XGBoost**, **Neural Networks**, **RNN**, and **LSTM** with the flexibility to switch models and configurations without touching the code.
 
---
 
## 📖 Overview

This pipeline is designed to streamline **end-to-end machine learning workflows** in development environments.  

It is **highly modular**, allowing developers and data scientists to:

- Quickly integrate new models.

- Load data from various sources (CSV, SQL databases).

- Test multiple configurations automatically.

- Maintain the same structure for both development and production pipelines.
 
---
 
## ✨ Key Features

- **🛠 Modular Design** – Each stage (data loading → training).

- **⚙ Config-Driven** – Switch models & parameters through a single YAML file.

- **📂 Multi-Source Data Loading** – Supports both CSV files and SQL databases.

- **🤖 Multi-Model Support** – XGBoost, NN, RNN, LSTM.

- **🔍 Auto Best Combination Finder** – Run multiple configs and pick the best-performing model automatically.

- **📊 Production-Like Structure** – Development pipeline mirrors production setup for smooth deployment.
 
---
 
## 📌 Pipeline Flow
 
```mermaid

flowchart LR

    A[Data Loader] --> B[Preprocessing]

    B --> C[Feature Engineering]

    C --> D[Model Training]

    D --> E[Evaluation & Metrics]

 
