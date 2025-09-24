"""
Módulo para cálculo de métricas e avaliação de modelos
"""
import numpy as np
from sklearn import metrics, model_selection


def evaluate_metrics(model, X_test, y_test):
    """
    Calcula métricas principais do modelo
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Labels de teste
    
    Returns:
        dict: Dicionário com métricas calculadas
    """
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1w = metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = metrics.confusion_matrix(y_test, y_pred, normalize="true").tolist()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "balanced_f1": float(f1w),
        "confusion_matrix": cm
    }


def collect_performance_over_time(automl):
    """
    Coleta dados de performance ao longo do tempo
    
    Args:
        automl: Modelo treinado do AutoSklearn
    
    Returns:
        dict: Dados de performance temporal
    """
    df = automl.performance_over_time_.copy()
    cols = [c for c in df.columns if c not in ("num_models_trained",)]
    result = {}
    for c in cols:
        if c == "Timestamp":
            # Converte timestamps string iso
            result[c] = [ts.isoformat() for ts in df[c]]
        else:
            result[c] = df[c].tolist()
    return result


def compute_learning_curve(estimator, X_tr, y_tr, scoring, random_state, n_jobs=1):
    """
    Computa curva de aprendizado - VERSÃO OTIMIZADA
    
    Args:
        estimator: Estimador treinado
        X_tr: Dados de treino
        y_tr: Labels de treino
        scoring: Métrica de scoring
        random_state: Seed para reprodutibilidade
        n_jobs: Número de jobs paralelos (padrão 1 para evitar conflitos)
    
    Returns:
        dict: Dados da curva de aprendizado
    """
    sizes, train_scores, val_scores = model_selection.learning_curve(
        estimator, X_tr, y_tr, 
        cv=3,  
        scoring=scoring, 
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.3, 1.0, 4),  # 4 pontos: 30%, 57%, 83%, 100%
        shuffle=True, 
        random_state=random_state
    )
    return {
        "train_sizes": sizes.tolist(),
        "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
        "val_scores_mean": np.mean(val_scores, axis=1).tolist(),
        "train_scores_std": np.std(train_scores, axis=1).tolist(),
        "val_scores_std": np.std(val_scores, axis=1).tolist()
    }