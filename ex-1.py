"""
MLflow - Premier Exemple Simple
"""
import mlflow
import random

# Démarrer une expérience
mlflow.set_experiment("mon_premier_experiment")

# Démarrer un "run" (une exécution)
with mlflow.start_run():
    
    # 1. Logger des paramètres
    learning_rate = 0.05
    epochs = 20
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    
    # 2. Simuler un entraînement et logger des métriques
    for epoch in range(epochs):
        # Simuler une métrique qui s'améliore
        accuracy = 0.5 + (epoch / epochs) * 0.4 + random.uniform(-0.05, 0.05)
        loss = 1.0 - (epoch / epochs) * 0.7 + random.uniform(-0.05, 0.05)
        
        # Logger les métriques
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("loss", loss, step=epoch)
        
        print(f"Epoch {epoch}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
    
    # 3. Logger un résultat final
    final_accuracy = accuracy
    mlflow.log_metric("final_accuracy", final_accuracy)
    
    print(f"\n✅ Run terminé ! Accuracy finale: {final_accuracy:.4f}")
    print("🔍 Lance 'mlflow ui' dans le terminal pour voir les résultats")