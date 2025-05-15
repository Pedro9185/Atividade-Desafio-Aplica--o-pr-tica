import matplotlib
matplotlib.use('Agg')  # Use backend não interativo para evitar avisos de GUI

from flask import Flask, render_template, request, jsonify
import io
import matplotlib.pyplot as plt
import numpy as np
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

app = Flask(__name__)

# Dicionário para armazenar os modelos
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

X_train, X_test, y_train, y_test = None, None, None, None

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/train', methods=['POST'])
def train():
    global X_train, X_test, y_train, y_test
    # Carregar o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Dividir em treino e teste (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar todos os modelos
    for name, model in models.items():
        model.fit(X_train, y_train)

    return jsonify({"message": "Treinamento concluído para todos os modelos"})

@app.route('/test', methods=['GET'])
def test():
    global X_test, y_test, X_train, y_train
    if not models:
        return jsonify({"error": "Modelos não treinados"}), 400

    results = {}
    plots = {}
    
    # Avaliar cada modelo
    for model_name, model in models.items():
        # Métricas para teste
        y_pred_test = model.predict(X_test)
        test_metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred_test), 3),
            "precision": round(precision_score(y_test, y_pred_test, average='weighted'), 3),
            "recall": round(recall_score(y_test, y_pred_test, average='weighted'), 3)
        }
        
        # Métricas para treino
        y_pred_train = model.predict(X_train)
        train_metrics = {
            "accuracy": round(accuracy_score(y_train, y_pred_train), 3),
            "precision": round(precision_score(y_train, y_pred_train, average='weighted'), 3),
            "recall": round(recall_score(y_train, y_pred_train, average='weighted'), 3)
        }
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_test)))
        classes = load_iris().target_names
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        cm_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Superfície de decisão (apenas para os 2 melhores atributos)
        X_train_2 = X_train[:, 2:4]
        model2 = type(model)(**model.get_params())
        model2.fit(X_train_2, y_train)
        
        x_min, x_max = X_train_2[:, 0].min() - 1, X_train_2[:, 0].max() + 1
        y_min, y_max = X_train_2[:, 1].min() - 1, X_train_2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        for i, target_name in enumerate(classes):
            idx = np.where(y_test == i)
            plt.scatter(X_test[idx, 2], X_test[idx, 3], edgecolors='k', label=target_name)
        plt.xlabel('Comprimento da Pétala (cm)')
        plt.ylabel('Largura da Pétala (cm)')
        plt.title(f'Superfície de Decisão - {model_name}')
        plt.legend()
        
        buf_ds = io.BytesIO()
        plt.savefig(buf_ds, format='png')
        buf_ds.seek(0)
        plt.close()
        ds_img = base64.b64encode(buf_ds.getvalue()).decode('utf-8')
        
        # Relatório de classificação
        report = classification_report(y_test, y_pred_test, target_names=classes, output_dict=True)
        
        results[model_name] = {
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "classification_report": report
        }
        
        plots[model_name] = {
            "confusion_matrix": cm_img,
            "decision_surface": ds_img
        }
    
    # Gráfico comparativo de acurácia
    model_names = list(models.keys())
    test_acc = [results[m]["test_metrics"]["accuracy"] for m in model_names]
    train_acc = [results[m]["train_metrics"]["accuracy"] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_acc, width, label='Treino')
    plt.bar(x + width/2, test_acc, width, label='Teste')
    plt.xlabel('Modelos')
    plt.ylabel('Acurácia')
    plt.title('Comparação de Acurácia entre Modelos')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(train_acc):
        plt.text(i - width/2, v + 0.05, f"{v:.2f}", ha='center')
    for i, v in enumerate(test_acc):
        plt.text(i + width/2, v + 0.05, f"{v:.2f}", ha='center')
    
    buf_comp = io.BytesIO()
    plt.savefig(buf_comp, format='png')
    buf_comp.seek(0)
    plt.close()
    comp_img = base64.b64encode(buf_comp.getvalue()).decode('utf-8')
    
    plots["comparison"] = comp_img
    
    return jsonify({"results": results, "plots": plots})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        values = [float(data[key]) for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    except Exception as e:
        return jsonify({"error": "Dados inválidos"}), 400

    predictions = {}
    iris = load_iris()
    
    for model_name, model in models.items():
        pred = model.predict([values])[0]
        proba = model.predict_proba([values])[0]
        result = iris.target_names[pred]
        confidence = round(proba[pred] * 100, 2)
        predictions[model_name] = {
            "prediction": result,
            "confidence": f"{confidence}%",
            "probabilities": {iris.target_names[i]: round(p*100, 2) for i, p in enumerate(proba)}
        }
    
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)