#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from src.core.azure_storage import AzureStorage

storage = AzureStorage()
model_ok = storage.blob_exists('models/stars_delta_7d_model.joblib')
meta_ok = storage.blob_exists('models/stars_delta_7d_metadata.json')

print(f'Modele: {"OK" if model_ok else "MANQUANT"}')
print(f'Metadata: {"OK" if meta_ok else "MANQUANT"}')

if model_ok:
    model = storage.download_bytes('models/stars_delta_7d_model.joblib')
    print(f'Taille: {len(model)/1024/1024:.2f} MB')
    
if meta_ok:
    meta = storage.download_json('models/stars_delta_7d_metadata.json')
    print(f'R2: {meta["metrics"]["r2"]:.3f}')
    print(f'MAE: {meta["metrics"]["mae"]:.2f}')

if not model_ok or not meta_ok:
    print('\nPROBLEME: Le modele n est pas sur Azure!')
    print('Relance: docker compose run --rm train')
    sys.exit(1)
else:
    print('\nTOUT EST BON!')
    sys.exit(0)
