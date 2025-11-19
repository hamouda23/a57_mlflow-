"""
MLflow - Charger un Modèle Sauvegardé
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
import numpy as np

# ========================================
# MÉTHODE 1 : Charger avec Run ID
# ========================================

print("🔍 Recherche du dernier modèle...")

# Récupérer le dernier run de l'expérience
experiment = mlflow.get_experiment_by_name("sauvegarde_modeles")
if experiment is None:
    print("❌ Lance d'abord 02_mlflow_sauvegarder_modele.py !")
    exit()

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if len(runs) == 0:
    print("❌ Aucun run trouvé ! Lance d'abord le script de sauvegarde.")
    exit()

run_id = runs.iloc[0]['run_id']
print(f"✅ Run ID trouvé: {run_id}")

# Charger le modèle
model_uri = f"runs:/{run_id}/model"
print(f"📂 Chargement depuis: {model_uri}")

loaded_model = mlflow.sklearn.load_model(model_uri)
print("✅ Modèle chargé avec succès !")

# ========================================
# TESTER LE MODÈLE CHARGÉ
# ========================================

print("\n🧪 Test du modèle chargé...")

# Créer de nouvelles données de test
X_new, y_new = make_classification(
    n_samples=5,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=999
)

# Faire des prédictions
predictions = loaded_model.predict(X_new)
probabilities = loaded_model.predict_proba(X_new)

print("\n📊 Résultats des prédictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Sample {i+1}: Classe={pred}, Proba=[{prob[0]:.3f}, {prob[1]:.3f}]")

# ========================================
# MÉTHODE 2 : Charger avec Chemin Direct
# ========================================

print("\n" + "="*60)
print("📌 MÉTHODE ALTERNATIVE : Charger avec chemin")
print("="*60)

# Tu peux aussi charger directement avec le chemin
# Remplace [experiment_id] et [run_id] par les vrais valeurs
experiment_id = experiment.experiment_id
model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"

try:
    loaded_model_2 = mlflow.sklearn.load_model(model_path)
    print(f"✅ Modèle chargé depuis: {model_path}")
except Exception as e:
    print(f"ℹ️ Chemin local: {e}")

print("\n" + "="*60)
print("✅ TOUT FONCTIONNE ! Tu sais maintenant :")
print("   1️⃣ Sauvegarder un modèle avec log_model()")
print("   2️⃣ Charger un modèle avec load_model()")
print("   3️⃣ Utiliser le modèle pour faire des prédictions")
print("="*60)