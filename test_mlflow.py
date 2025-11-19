import mlflow

# Lier MLflow au serveur local
mlflow.set_tracking_uri("http://localhost:5000")

# Créer ou sélectionner une expérience
mlflow.set_experiment("exp_test")

# Démarrer un run
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    print("Run enregistré !")
