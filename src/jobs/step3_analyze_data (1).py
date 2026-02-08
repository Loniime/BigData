#!/usr/bin/env python3
"""
ÉTAPE 3 : Analyse de la qualité des données collectées depuis Azure
Lance ce script après 7 jours minimum pour vérifier la qualité
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
from azure.storage.blob import BlobServiceClient

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Azure Blob Storage
    AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING', '')
    CONTAINER_NAME = os.getenv('AZURE_CONTAINER', 'github-data')

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
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            print(f"✅ Connecté au container: {self.container_name}")
        except Exception as e:
            print(f"❌ Erreur connexion Azure: {e}")
            raise
    
    def list_snapshots(self) -> List[str]:
        """Lister tous les snapshots disponibles"""
        try:
            blob_list = self.container_client.list_blobs(name_starts_with='daily_snapshots/snapshot_')
            return sorted([blob.name for blob in blob_list])
        except Exception as e:
            print(f"❌ Erreur listing blobs: {e}")
            return []
    
    def download_blob_json(self, blob_name: str) -> Dict:
        """Télécharger un blob JSON"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            json_data = blob_client.download_blob().readall()
            return json.loads(json_data)
        except Exception as e:
            print(f"❌ Erreur téléchargement {blob_name}: {e}")
            return None
    
    def upload_blob_json(self, blob_name: str, data: Dict) -> bool:
        """Upload un blob JSON"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            json_string = json.dumps(data, indent=2, ensure_ascii=False)
            blob_client.upload_blob(json_string, overwrite=True)
            return True
        except Exception as e:
            print(f"❌ Erreur upload {blob_name}: {e}")
            return False

# =============================================================================
# ANALYZER
# =============================================================================

class DataAnalyzer:
    def __init__(self, storage: AzureStorage):
        self.storage = storage
        self.snapshots = []
        self.load_snapshots()
    
    def load_snapshots(self):
        """Charger tous les snapshots disponibles depuis Azure"""
        print(f"📂 Chargement des snapshots depuis Azure...")
        snapshot_blobs = self.storage.list_snapshots()
        
        if not snapshot_blobs:
            raise ValueError("❌ Aucun snapshot trouvé sur Azure!")
        
        for blob_name in snapshot_blobs:
            data = self.storage.download_blob_json(blob_name)
            if data:
                self.snapshots.append(data)
        
        print(f"✅ {len(self.snapshots)} snapshots chargés")
    
    def analyze_growth(self):
        """Analyser la croissance des stars"""
        if len(self.snapshots) < 2:
            print("⚠️ Au moins 2 snapshots nécessaires pour analyser la croissance")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 ANALYSE DE LA CROISSANCE")
        print(f"{'='*70}\n")
        
        # Comparer premier et dernier snapshot
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        first_date = first_snapshot['date']
        last_date = last_snapshot['date']
        
        print(f"Période analysée: {first_date} → {last_date}")
        print(f"Nombre de jours: {len(self.snapshots)}\n")
        
        # Créer un dictionnaire des repos
        first_repos = {r['id']: r for r in first_snapshot['repos']}
        last_repos = {r['id']: r for r in last_snapshot['repos']}
        
        # Analyser la croissance
        growth_stats = {
            'no_growth': 0,
            'low_growth_1_5': 0,
            'medium_growth_5_20': 0,
            'high_growth_20_plus': 0,
            'total': 0
        }
        
        growth_by_strata = defaultdict(lambda: {'total': 0, 'with_growth': 0, 'total_growth': 0})
        
        all_growths = []
        
        for repo_id in first_repos.keys():
            if repo_id not in last_repos:
                continue
            
            first_stars = first_repos[repo_id]['stars']
            last_stars = last_repos[repo_id]['stars']
            growth = last_stars - first_stars
            
            all_growths.append(growth)
            growth_stats['total'] += 1
            
            # Catégoriser
            if growth == 0:
                growth_stats['no_growth'] += 1
            elif growth < 5:
                growth_stats['low_growth_1_5'] += 1
            elif growth < 20:
                growth_stats['medium_growth_5_20'] += 1
            else:
                growth_stats['high_growth_20_plus'] += 1
            
            # Par strate
            strata = self.get_strata_for_stars(first_stars)
            growth_by_strata[strata]['total'] += 1
            if growth > 0:
                growth_by_strata[strata]['with_growth'] += 1
                growth_by_strata[strata]['total_growth'] += growth
        
        # Afficher résultats
        total = growth_stats['total']
        
        print(f"📈 Distribution de la croissance:")
        print(f"   Aucune croissance (0):    {growth_stats['no_growth']:5d} ({growth_stats['no_growth']/total*100:5.1f}%)")
        print(f"   Faible (1-4 stars):       {growth_stats['low_growth_1_5']:5d} ({growth_stats['low_growth_1_5']/total*100:5.1f}%)")
        print(f"   Moyenne (5-19 stars):     {growth_stats['medium_growth_5_20']:5d} ({growth_stats['medium_growth_5_20']/total*100:5.1f}%)")
        print(f"   Forte (20+ stars):        {growth_stats['high_growth_20_plus']:5d} ({growth_stats['high_growth_20_plus']/total*100:5.1f}%)")
        
        # Stats par strate
        print(f"\n⭐ Croissance par strate:")
        for strata in ['0-10', '10-100', '100-1000', '1000+']:
            if strata not in growth_by_strata:
                continue
            
            data = growth_by_strata[strata]
            pct_with_growth = (data['with_growth'] / data['total'] * 100) if data['total'] > 0 else 0
            avg_growth = data['total_growth'] / data['total'] if data['total'] > 0 else 0
            
            print(f"   {strata:10s}: {data['with_growth']:4d}/{data['total']:4d} ont gagné ≥1 star "
                  f"({pct_with_growth:5.1f}%) - Moy: {avg_growth:.1f} stars")
        
        # Statistiques globales
        repos_with_growth = total - growth_stats['no_growth']
        repos_with_5plus = growth_stats['medium_growth_5_20'] + growth_stats['high_growth_20_plus']
        
        pct_1plus = (repos_with_growth / total * 100) if total > 0 else 0
        pct_5plus = (repos_with_5plus / total * 100) if total > 0 else 0
        
        print(f"\n🎯 Métriques clés:")
        print(f"   Repos avec ≥1 star:  {repos_with_growth:5d} ({pct_1plus:5.1f}%)")
        print(f"   Repos avec ≥5 stars: {repos_with_5plus:5d} ({pct_5plus:5.1f}%)")
        
        # Moyenne et médiane
        if all_growths:
            avg_growth = sum(all_growths) / len(all_growths)
            sorted_growths = sorted(all_growths)
            median_growth = sorted_growths[len(sorted_growths)//2]
            
            print(f"\n📊 Statistiques de croissance:")
            print(f"   Moyenne: {avg_growth:.2f} stars")
            print(f"   Médiane: {median_growth} stars")
            print(f"   Min: {min(all_growths)} stars")
            print(f"   Max: {max(all_growths)} stars")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if pct_1plus < 15:
            print("   ❌ Dataset trop plat! < 15% ont gagné ≥1 star")
            print("      → Rééquilibrer: augmenter 100-1000 et 1000+, réduire 0-10")
        elif pct_1plus < 25:
            print("   ⚠️  Dataset moyen: 15-25% ont gagné ≥1 star")
            print("      → Acceptable mais peut être amélioré")
        else:
            print("   ✅ Excellent dataset! > 25% ont gagné ≥1 star")
            print("      → Dataset informatif, bon pour le ML")
        
        print(f"{'='*70}\n")
    
    def get_strata_for_stars(self, stars: int) -> str:
        """Déterminer la strate"""
        if stars < 10:
            return "0-10"
        elif stars < 100:
            return "10-100"
        elif stars < 1000:
            return "100-1000"
        else:
            return "1000+"
    
    def analyze_daily_trends(self):
        """Analyser les tendances jour par jour"""
        if len(self.snapshots) < 2:
            return
        
        print(f"\n{'='*70}")
        print(f"📈 TENDANCES QUOTIDIENNES")
        print(f"{'='*70}\n")
        
        daily_stats = []
        
        for i in range(1, len(self.snapshots)):
            prev_snapshot = self.snapshots[i-1]
            curr_snapshot = self.snapshots[i]
            
            prev_repos = {r['id']: r for r in prev_snapshot['repos']}
            curr_repos = {r['id']: r for r in curr_snapshot['repos']}
            
            daily_growth = 0
            repos_changed = 0
            
            for repo_id in prev_repos.keys():
                if repo_id not in curr_repos:
                    continue
                
                growth = curr_repos[repo_id]['stars'] - prev_repos[repo_id]['stars']
                daily_growth += growth
                if growth > 0:
                    repos_changed += 1
            
            daily_stats.append({
                'date': curr_snapshot['date'],
                'total_growth': daily_growth,
                'repos_with_growth': repos_changed
            })
        
        # Afficher
        print(f"Date          | Total growth | Repos with growth")
        print(f"--------------|--------------|------------------")
        for stat in daily_stats:
            print(f"{stat['date']} |     {stat['total_growth']:6d} |         {stat['repos_with_growth']:5d}")
        
        print()
    
    def export_for_ml(self, output_blob='ml_exports/ml_dataset.json'):
        """Exporter les données dans un format prêt pour le ML sur Azure"""
        if len(self.snapshots) < 7:
            print("⚠️ Au moins 7 jours de données recommandés pour le ML")
        
        print(f"\n{'='*70}")
        print(f"🤖 EXPORT POUR ML")
        print(f"{'='*70}\n")
        
        # Construire le dataset
        ml_data = []
        
        # Pour chaque repo, créer une série temporelle
        first_snapshot = self.snapshots[0]
        repo_ids = [r['id'] for r in first_snapshot['repos']]
        
        for repo_id in repo_ids:
            time_series = []
            
            for snapshot in self.snapshots:
                repo_data = next((r for r in snapshot['repos'] if r['id'] == repo_id), None)
                if repo_data:
                    time_series.append({
                        'date': snapshot['date'],
                        'stars': repo_data['stars'],
                        'forks': repo_data['forks'],
                        'watchers': repo_data['watchers'],
                        'open_issues': repo_data['open_issues']
                    })
            
            if len(time_series) >= 2:
                ml_data.append({
                    'repo_id': repo_id,
                    'full_name': next((r['full_name'] for r in first_snapshot['repos'] if r['id'] == repo_id), None),
                    'time_series': time_series
                })
        
        # Sauvegarder sur Azure
        if self.storage.upload_blob_json(output_blob, ml_data):
            print(f"✅ Dataset ML exporté sur Azure: {output_blob}")
            print(f"   {len(ml_data)} repos")
            print(f"   {len(self.snapshots)} points temporels par repo")
            print(f"   Prêt pour l'entraînement!\n")
        else:
            print(f"❌ Erreur lors de l'export vers Azure")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("📊 ANALYSE DES DONNÉES COLLECTÉES - Azure Blob Storage")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Vérifier variables
    if not Config.AZURE_CONNECTION_STRING:
        print("❌ ERREUR: Variable AZURE_CONNECTION_STRING manquante")
        print("   Export: export AZURE_CONNECTION_STRING='...'")
        return 1
    
    try:
        # Initialiser Azure Storage
        print("☁️ Connexion à Azure Blob Storage...")
        storage = AzureStorage()
        
        # Initialiser analyzer
        analyzer = DataAnalyzer(storage)
        
        # Analyses
        analyzer.analyze_growth()
        analyzer.analyze_daily_trends()
        analyzer.export_for_ml()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
