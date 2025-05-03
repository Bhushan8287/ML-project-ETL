## ☕ Coffee Shop Revenue Prediction
**End-to-end regression pipeline** for predicting coffee shop revenue using supervised machine learning techniques. This project showcases an industry-style workflow including modular code, hyperparameter tuning, local experiment tracking, and model deployment using Flask.

---

### 📖 Overview
This project demonstrates a foundational machine learning workflow applied to a regression problem. It includes:

- Exploratory Data Analysis (EDA)
- Feature engineering and data transformation
- Model training and evaluation across 9 regression algorithms
- Hyperparameter tuning using custom functions
- Local experiment tracking using MLflow
- Metric logging to JSON
- Basic Flask-based web application for deployment

All components are modularized using Python scripts and a structured folder layout to reflect real-world ML practices.

---

### ❓ Problem Statement
Coffee shop owners face challenges in predicting daily revenue and optimizing operational factors like marketing spend, staffing, and pricing strategies. This dataset provides a foundation for building predictive models to understand and improve revenue generation, enabling more efficient planning, resource allocation, and customer satisfaction.  

---

### 🧰 Tech Stack
- Python
- scikit-learn
- XGBoost
- NumPy, pandas
- matplotlib, seaborn
- MLflow (local)
- Flask (for deployment)
- JSON (for metrics storage)
- Jupyter Notebooks (EDA and model dev)

---

### 📂 Project Structure

```
.
ML-projectETL/
├── artifacts/
├── notebook/
│   ├── EDA.ipynb
│   ├── Model_Building.ipynb
│   ├── transformed.csv
│   ├── splits/
│   ├── dataset/
│   └── metrics.json
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── app.py
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

### ⚙️ How It Works
1. **EDA Notebook**:
   - Visualize feature distributions and correlations
   - Identify potential data quality issues
2. **Data Transformation**:
   - Scaling, encoding, missing value handling
3. **Model Training**:
   - Train 9 ML models: LinearReg, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, SVR, XGBoostRegressor, KNN
   - Custom functions for training, tuning, and evaluation
4. **Metrics Logging**:
   - Metrics stored in `metrics.json`
   - Tracked locally using MLflow
5. **Deployment**:
   - Flask app integrated with final model for simple inference

---

### 📊 Results
- Evaluation metrics R²: 0.953, RMSE: 209.78 , and MAE:168.45.
- Best-performing model: **[XGBoost Regressor]**
- R² Score: **[e.g., 0.953]**
- Model tuning decisions and rationale documented in notebooks.

---

### 🚀 Installation & Usage
```
git clone the repo
pip install -r requirements.txt
```

To run the Flask app:

```
python app.py
```

To launch the MLflow UI:

```
mlflow ui
```

---

### 📉 Limitations
- Dataset is relatively simple.
- MLflow runs are local only—no remote experiment tracking.
- Flask app dosent has frontend styling.

---

### 🔮 Future Improvements
- Integrate cloud MLflow (e.g., S3 backend)
- Add Docker containerization
- Extend with real-world datasets
- Integrate streamlit or Gradio frontend
- Implement CI/CD for model deployment

---
