"""
MLflow - Sauvegarder et Charger un Modèle
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Créer des données synthétiques
print("📊 Création des données...")
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ========================================
# PARTIE 1 : ENTRAÎNER ET SAUVEGARDER
# ========================================

mlflow.set_experiment("sauvegarde_modeles")

with mlflow.start_run(run_name="random_forest_v1"):
    
    # Paramètres du modèle
    n_estimators = 200
    max_depth = 20
    
    # Logger les paramètres
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")
    
    # Entraîner le modèle
    print("\n🎯 Entraînement du modèle...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Évaluer
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Logger les métriques
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("n_samples_train", len(X_train))
    mlflow.log_metric("n_samples_test", len(X_test))
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # ⭐ SAUVEGARDER LE MODÈLE ⭐
    mlflow.sklearn.log_model(
        model, 
        "model",  # Nom du dossier dans artifacts
        signature=mlflow.models.infer_signature(X_train, y_train)
    )
    
    # Récupérer l'ID du run pour charger plus tard
    run_id = mlflow.active_run().info.run_id
    print(f"\n📦 Modèle sauvegardé !")
    print(f"🔑 Run ID: {run_id}")
    print(f"📁 Chemin: mlruns/[experiment_id]/{run_id}/artifacts/model")

print("\n" + "="*60)
print("✅ PARTIE 1 TERMINÉE : Modèle entraîné et sauvegardé")
print("="*60)