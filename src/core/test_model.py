#!/usr/bin/env python3
"""Vérifier si le modèle est bien sur Azure"""
from src.core.storage_factory import get_storage

storage = get_storage()

# Vérifier si le modèle existe
model_exists = storage.blob_exists("models/stars_delta_7d_model.joblib")
meta_exists = storage.blob_exists("models/stars_delta_7d_metadata.json")

print(f"Modèle (.joblib): {'✅ EXISTE' if model_exists else '❌ MANQUANT'}")
print(f"Metadata (.json): {'✅ EXISTE' if meta_exists else '❌ MANQUANT'}")

if model_exists:
    # Tester le téléchargement
    model_bytes = storage.download_bytes("models/stars_delta_7d_model.joblib")
    print(f"Taille du modèle: {len(model_bytes) / 1024:.1f} KB")
    
    metadata = storage.download_json("models/stars_delta_7d_metadata.json")
    print(f"Métriques du modèle:")
    print(f"  R²: {metadata['metrics']['r2']:.3f}")
    print(f"  MAE: {metadata['metrics']['mae']:.2f}")
else:
    print("\n❌ PROBLÈME: Le modèle n'a pas été uploadé sur Azure!")
    print("Relance: docker compose run --rm train")