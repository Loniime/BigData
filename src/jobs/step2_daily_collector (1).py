#!/usr/bin/env python3
"""
ÉTAPE 2 : Collecte quotidienne des métriques avec Azure Blob Storage
Ce script s'exécute CHAQUE JOUR pour suivre l'évolution des 10k repos
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
from azure.storage.blob import BlobServiceClient, ContentSettings

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # GitHub API
    GITHUB_TOKENS = os.getenv('GITHUB_TOKENS', '').split(',')
    BASE_URL = 'https://api.github.com'
    
    # Azure Blob Storage
    AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING', '')
    CONTAINER_NAME = os.getenv('AZURE_CONTAINER', 'github-data')
    
    # Fichiers
    REPOS_FILE = 'repos_to_track.json'  # Liste des repos (output du step 1)
    
    # Rate limiting
    REQUESTS_PER_HOUR = 5000  # GitHub API limit

# =============================================================================
# AZURE STORAGE
# =============================================================================

class AzureStorage:
    def __init__(self):
        self.connection_string = Config.AZURE_CONNECTION_STRING
        self.container_name = Config.CONTAINER_NAME
        
        if not self.connection_string:
            raise ValueError("❌ AZURE_CONNECTION_STRING manquante!")
        
        try:
            # Initialiser client Azure
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            
            # Créer container si n'existe pas
            try:
                self.container_client.create_container()
                print(f"✅ Container '{self.container_name}' créé")
            except:
                print(f"✅ Container '{self.container_name}' existe déjà")
                
        except Exception as e:
            print(f"❌ Erreur connexion Azure: {e}")
            raise
    
    def upload_blob_json(self, blob_name: str, data: Dict) -> bool:
        """Upload un blob JSON vers Azure"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            json_string = json.dumps(data, indent=2, ensure_ascii=False)
            
            content_settings = ContentSettings(content_type='application/json')
            blob_client.upload_blob(
                json_string,
                overwrite=True,
                content_settings=content_settings
            )
            return True
        except Exception as e:
            print(f"❌ Erreur upload {blob_name}: {e}")
            return False
    
    def download_blob_json(self, blob_name: str) -> Optional[Dict]:
        """Télécharger un blob JSON depuis Azure"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            json_data = blob_client.download_blob().readall()
            return json.loads(json_data)
        except:
            return None
    
    def blob_exists(self, blob_name: str) -> bool:
        """Vérifier si un blob existe"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except:
            return False

# =============================================================================
# DAILY COLLECTOR
# =============================================================================

class DailyCollector:
    def __init__(self, storage: AzureStorage):
        self.storage = storage
        self.tokens = [t.strip() for t in Config.GITHUB_TOKENS if t.strip()]
        self.current_token_index = 0
        
        if not self.tokens:
            raise ValueError("❌ Aucun token GitHub fourni!")
        
        # Charger la liste des repos à suivre
        if not os.path.exists(Config.REPOS_FILE):
            raise FileNotFoundError(f"❌ Fichier {Config.REPOS_FILE} introuvable!")
        
        with open(Config.REPOS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.repos_to_track = data['repos']
        
        # Stats
        self.stats = {
            'collected': 0,
            'errors': 0,
            'rate_limit_hits': 0,
            'start_time': datetime.now().isoformat()
        }
        
        print(f"✅ {len(self.tokens)} token(s) GitHub chargé(s)")
        print(f"📊 {len(self.repos_to_track)} repos à collecter")
        print(f"☁️  Sauvegarde: Azure Blob Storage ({Config.CONTAINER_NAME})")
    
    def get_headers(self) -> Dict:
        """Rotation des tokens"""
        token = self.tokens[self.current_token_index]
        self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
        return {'Authorization': f'token {token}'}
    
    def check_rate_limit(self) -> tuple:
        """Vérifier rate limit"""
        try:
            response = requests.get(
                f'{Config.BASE_URL}/rate_limit',
                headers=self.get_headers(),
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                remaining = data['resources']['core']['remaining']
                reset_time = data['resources']['core']['reset']
                return remaining, reset_time
        except Exception as e:
            print(f"⚠️ Erreur check rate limit: {e}")
        return None, None
    
    def fetch_repo_info(self, owner: str, name: str) -> Optional[Dict]:
        """Récupérer les infos actuelles d'un repo"""
        for attempt in range(3):
            try:
                url = f'{Config.BASE_URL}/repos/{owner}/{name}'
                response = requests.get(
                    url,
                    headers=self.get_headers(),
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 403:
                    self.stats['rate_limit_hits'] += 1
                    remaining, reset_time = self.check_rate_limit()
                    if remaining == 0 and reset_time:
                        wait_time = reset_time - time.time() + 10
                        if wait_time > 0:
                            print(f"⏳ Rate limit, attente {wait_time/60:.1f} min...")
                            time.sleep(wait_time)
                        continue
                    time.sleep(5)
                
                elif response.status_code == 404:
                    print(f"⚠️ Repo non trouvé: {owner}/{name}")
                    return None
                
                else:
                    print(f"⚠️ Erreur {response.status_code} pour {owner}/{name}")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Exception pour {owner}/{name}: {e}")
                time.sleep(2)
        
        return None
    
    def collect_daily_snapshot(self):
        """Collecter un snapshot quotidien de tous les repos"""
        today = datetime.now().strftime('%Y-%m-%d')
        blob_name = f'daily_snapshots/snapshot_{today}.json'
        
        # Vérifier si le snapshot existe déjà
        if self.storage.blob_exists(blob_name):
            print(f"⚠️ Le snapshot pour {today} existe déjà sur Azure")
            response = input("Voulez-vous le remplacer? (y/n): ")
            if response.lower() != 'y':
                print("❌ Collecte annulée")
                return
        
        snapshot_data = {
            'date': today,
            'collected_at': datetime.now().isoformat(),
            'repos': []
        }
        
        print(f"\n{'='*70}")
        print(f"📅 COLLECTE DU {today}")
        print(f"🎯 {len(self.repos_to_track)} repos à collecter")
        print(f"☁️  Destination: Azure ({blob_name})")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for i, repo_info in enumerate(self.repos_to_track, 1):
            owner = repo_info['owner']
            name = repo_info['name']
            
            # Progression
            if i % 100 == 0:
                elapsed = time.time() - start_time
                repos_per_sec = i / elapsed
                remaining = len(self.repos_to_track) - i
                eta_seconds = remaining / repos_per_sec if repos_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"📊 Progression: {i}/{len(self.repos_to_track)} "
                      f"({i/len(self.repos_to_track)*100:.1f}%) - "
                      f"ETA: {eta_minutes:.1f} min")
            
            # Récupérer les données actuelles
            current_data = self.fetch_repo_info(owner, name)
            
            if current_data:
                repo_snapshot = {
                    'id': repo_info['id'],
                    'full_name': repo_info['full_name'],
                    'stars': current_data['stargazers_count'],
                    'forks': current_data['forks_count'],
                    'watchers': current_data['watchers_count'],
                    'open_issues': current_data['open_issues_count'],
                    'pushed_at': current_data.get('pushed_at'),
                    'size': current_data.get('size'),
                    'language': current_data.get('language')
                }
                
                snapshot_data['repos'].append(repo_snapshot)
                self.stats['collected'] += 1
            else:
                self.stats['errors'] += 1
            
            # Petit délai pour éviter le rate limiting
            time.sleep(0.1)
        
        # Sauvegarder le snapshot sur Azure
        print(f"\n☁️  Upload vers Azure...")
        if self.storage.upload_blob_json(blob_name, snapshot_data):
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"✅ COLLECTE TERMINÉE!")
            print(f"   Fichier Azure: {blob_name}")
            print(f"   Repos collectés: {self.stats['collected']}")
            print(f"   Erreurs: {self.stats['errors']}")
            print(f"   Rate limit hits: {self.stats['rate_limit_hits']}")
            print(f"   Durée: {elapsed_time/60:.1f} minutes")
            print(f"{'='*70}\n")
            
            # Sauvegarder les stats
            self.save_stats()
        else:
            print(f"❌ Erreur lors de l'upload vers Azure")
    
    def save_stats(self):
        """Sauvegarder les stats de la collecte sur Azure"""
        today = datetime.now().strftime('%Y-%m-%d')
        stats_blob = f'daily_snapshots/stats_{today}.json'
        
        self.stats['end_time'] = datetime.now().isoformat()
        
        if self.storage.upload_blob_json(stats_blob, self.stats):
            print(f"📊 Stats sauvegardées: {stats_blob}")
        else:
            print(f"⚠️ Erreur sauvegarde stats")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("📅 COLLECTE QUOTIDIENNE DES MÉTRIQUES - Azure Blob Storage")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Vérifier variables
    if not Config.GITHUB_TOKENS or Config.GITHUB_TOKENS == ['']:
        print("❌ ERREUR: Variable GITHUB_TOKENS manquante")
        print("   Export: export GITHUB_TOKENS='token1,token2,token3'")
        return 1
    
    if not Config.AZURE_CONNECTION_STRING:
        print("❌ ERREUR: Variable AZURE_CONNECTION_STRING manquante")
        print("   Export: export AZURE_CONNECTION_STRING='...'")
        return 1
    
    try:
        # Initialiser Azure Storage
        print("☁️ Connexion à Azure Blob Storage...")
        storage = AzureStorage()
        
        # Initialiser collector
        collector = DailyCollector(storage)
        collector.collect_daily_snapshot()
        
        print("💡 Conseil: Programme ce script pour tourner chaque jour")
        print("   Exemple cron: 0 2 * * * /usr/bin/python3 /path/to/step2_daily_collector.py")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interruption manuelle")
        return 130
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
