# Olist Multi-Task Operations Platform

> An AI-powered, full-stack e-commerce intelligence platform for predictive logistics, customer sentiment analysis, and business intelligence — built on the Brazilian Olist dataset.

**Authors:** Nurul Labib Sayeedi & Mehraj Mahmood — United International University, Dhaka, Bangladesh

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

---

## Overview

The Brazilian e-commerce market poses a unique challenge: vast geographic distribution, varied freight infrastructure, and complex customer sentiment. This platform bridges the gap between raw data science research and accessible business intelligence by synthesizing multi-task machine learning and NLP into a single, interactive operational tool.

It empowers e-commerce sellers and logistics managers to:

- Accurately estimate delivery timelines before an order ships
- Forecast customer review scores from logistical metadata alone
- Instantly classify customer sentiment from raw text reviews
- Explore revenue, geographic bottlenecks, and customer behavior through interactive dashboards

---

## Features

The application is organized into four core tabs, each serving a distinct operational purpose.

### Tab 1 — Interactive Analysis Dashboard

The central business intelligence hub. Renders live KPIs (Total Orders, Total Revenue, Average Review Score) alongside 12 interactive Plotly charts:

- **RFM Segmentation** — Recency/Frequency/Monetary customer groupings
- **Geographic Logistics Bottlenecks** — State-level average delay and freight cost maps
- **Shipping Penalty Analysis** — Freight-to-price ratio vs. final review score
- **Brand Health Timeline** — Monthly satisfaction score and review verbosity trends
- **Listing Quality** — Product photo count vs. sales volume and review score
- **Revenue by Category** — Top 15 categories ranked by gross revenue
- **Sales by Hour** — Purchase volume patterns across the day
- **Feature Correlations** — Pearson correlation heatmap across all numeric features

### Tab 2 — NLP Review Analyzer

Accepts a raw (English-translated) customer review and outputs a predicted star rating (1–5) alongside a full confidence probability breakdown. Supports three deep-learning architectures selectable from a dropdown:

| Model | Architecture |
|---|---|
| GPT-2 (124M / 355M) | Autoregressive with Huber Loss |
| RoBERTa Base / Large | Bidirectional encoder |
| Text-JEPA | Joint Embedding Predictive Architecture |

### Tab 3 — Delivery Prediction Estimator

Accepts seven order-level numeric features (price, freight value, weight, dimensions, delay days) and runs them through **six parallel models simultaneously**:

- **NN Regression** — Predicts exact delivery days (custom PyTorch MLP with Huber Loss)
- **Neural Network Classifier** — Predicts categorical delivery status
- **Ordinal Logistic Regression**
- **Random Forest**
- **Support Vector Machine**
- **XGBoost**

### Tab 4 — Review Prediction via Order Metadata

Predicts the customer's expected star rating purely from logistical metadata — no review text needed. Accepts eight numeric features (including actual delivery days and delay days) and outputs predictions from four ML classifiers (OLR, Random Forest, SVM, XGBoost). Enables proactive flagging of high-risk delayed orders for customer service intervention.

---

## Model Performance

### Delivery Status Prediction (RQ1)

| Model | Accuracy (%) | F1-Score (%) |
|---|---|---|
| Random Forest | **50.72** | **50.69** |
| XGBoost | 48.96 | 48.99 |
| Neural Network | 38.00 | 42.35 |
| Support Vector Machine | 36.19 | 33.73 |
| Ordinal Logistic Regression | 33.01 | 30.40 |

> The custom PyTorch regression network achieved a **MAE of 3.09 days** for continuous delivery day prediction.

### Customer Satisfaction via Metadata (RQ2)

| Model | Accuracy (%) | F1-Score (%) |
|---|---|---|
| Random Forest | **62.89** | **57.57** |
| XGBoost | 58.97 | 55.59 |
| Ordinal Logistic Regression | 41.42 | 38.98 |
| Support Vector Machine | 36.03 | 38.10 |

### NLP Sentiment Classification (RQ3)

| Model | Accuracy (%) | MAE |
|---|---|---|
| **Text-JEPA** | **59.10** | **0.479** |
| GPT-2 355M (Huber) | 58.88 | 0.501 |
| GPT-2 124M (Huber) | 58.72 | 0.523 |
| RoBERTa Base | 40.10 | 0.730 |
| RoBERTa Large | 39.54 | 0.750 |
| GPT-2 124M (Cross-Entropy) | 33.34 | 0.800 |
| GPT-2 355M (Cross-Entropy) | 22.19 | 0.956 |

> Key insight: Switching GPT-2 from Cross-Entropy to Huber Loss nearly **doubled accuracy** and halved MAE. Text-JEPA's joint-embedding approach achieved the highest overall accuracy.

---

## Project Structure

```
Olist_analysis/
│
├── data/
│   └── interim/
│       ├── olist_PBI_processed_dataset.csv
│       ├── olist_ml_processed_delivery_dataset.csv
│       └── olist_ml_processed_review_dataset.csv
│
├── notebooks/              # Exploratory analysis and model training notebooks
│
├── src/                    # Source modules and serialized models
│   ├── models/             # Saved .joblib and .pt model files
│   └── ...
│
├── app.py                  # Main Gradio application
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | [Gradio](https://www.gradio.app/) |
| Data Manipulation | Pandas, NumPy |
| Visualization | Plotly (interactive), Plotly Subplots |
| ML Models | Scikit-Learn, Statsmodels (OLR), XGBoost |
| Model Serialization | Joblib |
| Deep Learning | PyTorch |
| NLP / Transformers | Hugging Face Transformers, `tiktoken` |
| Language | Python 3.10+ |

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Labib1610/Olist_analysis.git
cd Olist_analysis
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure the data path**

Open `app.py` and update `PBI_DATA_PATH` to point to your local copy of the processed PBI dataset:

```python
PBI_DATA_PATH = "data/interim/olist_PBI_processed_dataset.csv"
```

**4. Ensure model files are in place**

Trained `.joblib` and `.pt` model files should be present in the `src/models/` directory. These are loaded automatically at app startup.

---

## Usage

Launch the Gradio app:

```bash
python app.py
```

The interface will be available at `http://localhost:7860` by default.

**Tab-by-tab guide:**

1. **Dashboard (Tab 1):** Review KPIs at the top, then expand accordion sections to explore charts. Hover, zoom, and pan using Plotly's built-in cursor tools.

2. **Review Analyzer (Tab 2):** Select a model from the dropdown, paste an English-translated customer review into the text box, and click **Analyze Review**.

3. **Delivery Prediction (Tab 3):** Enter scaled numeric product features (price, freight, dimensions, delay days) and click **Predict Delivery** to see predictions from all six models simultaneously.

4. **Review Prediction (Tab 4):** Enter order metadata including actual delivery days and delay days, then click **Predict Review Score** to forecast the star rating a customer is likely to leave.

---

## Dataset

This project uses the **[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**, sourced from Kaggle. It contains relational data across orders, customers, products, sellers, and reviews.

**Preprocessing steps applied:**
- Joined multiple relational tables into a unified dataset
- Standard-scaled numeric features (weight, dimensions, price, freight value)
- Derived target variables: actual delivery days, 6-class delivery status, and 5-class review score
- Translated customer reviews from Portuguese to English via GPT-assisted translation for NLP evaluation

---

## Methodology

### Tabular Modeling (RQ1 & RQ2)

Models were trained on standard-scaled product and shipping features to predict both continuous delivery days and ordinal review scores:

- **Ensemble Trees (Random Forest, XGBoost):** Natively handle non-linear feature interactions and threshold behaviors common in logistics.
- **Linear/Distance Baselines (OLR, SVM):** Provide interpretable structural benchmarks.
- **Custom PyTorch MLP:** Optimized with Huber Loss for regression of exact delivery days, and Cross-Entropy for multi-class status classification.

### NLP Modeling (RQ3)

5-star review rating prediction framed as a textual classification problem across three architectural paradigms:

- **GPT-2 (Autoregressive):** Evaluated with standard Cross-Entropy and ordinal-aware Huber Loss. Huber Loss dramatically improved performance by penalizing distant misclassifications.
- **RoBERTa (Bidirectional):** Base and Large variants evaluated for complex contextual sentiment mapping.
- **Text-JEPA (Novel):** A Joint Embedding Predictive Architecture adapted from vision-language tasks. A frozen BERT encoder embeds the review context; a trainable transformer predictor maps it to a target rating embedding space via discriminative matching.

---

## Results

The platform demonstrates that:

- **Tree-based ensembles dominate tabular tasks** — Random Forest consistently outperformed neural networks and linear models for both logistics and satisfaction prediction.
- **Loss function choice is critical for NLP** — Huber Loss on GPT-2 nearly doubled accuracy compared to Cross-Entropy, revealing that ordinal distance awareness is essential for rating prediction.
- **Text-JEPA sets the benchmark** — The joint-embedding architecture achieved the highest NLP accuracy (59.10%) and lowest MAE (0.479), outperforming larger bidirectional and autoregressive models.
- **Non-linear satisfaction thresholds exist** — The dominance of tree-based models for RQ2 suggests customers have hard threshold expectations around delivery delays, not a linear relationship.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Built as part of a research project at United International University, Dhaka, Bangladesh — June 2026.*
