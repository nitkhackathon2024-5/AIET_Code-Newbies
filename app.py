

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                               precision_score, recall_score, f1_score,
                               matthews_corrcoef, confusion_matrix)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return redirect(url_for('analyze', filename=file.filename))
    return render_template('upload.html')

@app.route('/analyze/<filename>')
def analyze(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    data = pd.read_csv(file_path)

    if 'Class' not in data.columns:
        return "Column 'Class' not found in the uploaded data."

    data_info = {
        "shape": data.shape,
        "description": data.describe().to_html()
    }

    # Check for class imbalance
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    outlierFraction = len(fraud) / float(len(valid)) if len(valid) > 0 else 0

    fraud_info = fraud.Amount.describe().to_dict() if not fraud.empty else {}
    valid_info = valid.Amount.describe().to_dict() if not valid.empty else {}

    # Preprocessing: Handle categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)  # One-hot encoding

    # Splitting the data
    X = data.drop(['Class'], axis=1)
    Y = data['Class']
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Training the Random Forest model
    rfc = RandomForestClassifier()
    rfc.fit(xTrain, yTrain)
    yPred = rfc.predict(xTest)

    # Evaluation metrics
    acc = accuracy_score(yTest, yPred)
    prec = precision_score(yTest, yPred)
    rec = recall_score(yTest, yPred)
    f1 = f1_score(yTest, yPred)
    MCC = matthews_corrcoef(yTest, yPred)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "mcc": MCC
    }

    # Generate and save confusion matrix plot
    LABELS = ['Normal', 'Fraud']
    conf_matrix = confusion_matrix(yTest, yPred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plot_path = os.path.join(UPLOAD_FOLDER, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('results.html', data_info=data_info, outlier_fraction=outlierFraction,
                           fraud_info=fraud_info, valid_info=valid_info, metrics=metrics,
                           plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
