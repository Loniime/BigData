#!/usr/bin/env python3
"""
ÉTAPE 1 : Sélection des 10k repos à suivre
Ce script s'exécute UNE SEULE FOIS pour constituer la liste des repos
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # GitHub API
    GITHUB_TOKENS = os.getenv('GITHUB_TOKENS', '').split(',')
    BASE_URL = 'https://api.github.com'
    
    # Fichier de sortie
    OUTPUT_FILE = 'repos_to_track.json'
    
    # Objectif de collecte
    TARGET_REPOS = 10000
    
    # Stratification par stars (pourcentages de TARGET_REPOS)
    STAR_STRATA = {
        '0-10': {'min': 0, 'max': 10, 'percentage': 0.20},
        '10-100': {'min': 10, 'max': 100, 'percentage': 0.30},
        '100-1000': {'min': 100, 'max': 1000, 'percentage': 0.30},
        '1000+': {'min': 1000, 'max': None, 'percentage': 0.20}
    }
    
    # Filtre d'activité (jours)
    RECENT_ACTIVITY_DAYS = 90
    
    LANGUAGES = [
        'Python', 'JavaScript', 'TypeScript', 'Java', 'Go', 'Rust',
        'C++', 'C', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'C#', 'Scala',
        'Dart', 'R', 'Perl', 'Haskell', 'Elixir', 'Clojure'
    ]

# =============================================================================
# REPO SELECTOR
# =============================================================================

class RepoSelector:
    def __init__(self):
        self.tokens = [t.strip() for t in Config.GITHUB_TOKENS if t.strip()]
        self.current_token_index = 0
        self.selected_repos = []
        self.seen_ids = set()
        
        # Calculer les objectifs par strate
        self.targets = {}
        for strata_name, strata_config in Config.STAR_STRATA.items():
            target = int(Config.TARGET_REPOS * strata_config['percentage'])
            self.targets[strata_name] = {
                'target': target,
                'collected': 0,
                'min_stars': strata_config['min'],
                'max_stars': strata_config['max']
            }
        
        if not self.tokens:
            raise ValueError("❌ Aucun token GitHub fourni!")
        
        print(f"✅ {len(self.tokens)} token(s) GitHub chargé(s)")
    
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
                remaining = data['resources']['search']['remaining']
                reset_time = data['resources']['search']['reset']
                return remaining, reset_time
        except Exception as e:
            print(f"⚠️ Erreur check rate limit: {e}")
        return None, None
    
    def is_recently_active(self, repo: Dict) -> bool:
        """Vérifier si le repo a une activité récente"""
        pushed_at = repo.get('pushed_at')
        if not pushed_at:
            return False
        
        try:
            pushed_date = datetime.fromisoformat(pushed_at.replace('Z', '+00:00'))
            cutoff_date = datetime.now(pushed_date.tzinfo) - timedelta(days=Config.RECENT_ACTIVITY_DAYS)
            return pushed_date > cutoff_date
        except:
            return False
    
    def get_strata_for_stars(self, stars: int) -> Optional[str]:
        """Déterminer la strate pour un nombre de stars"""
        for strata_name, target_info in self.targets.items():
            min_stars = target_info['min_stars']
            max_stars = target_info['max_stars']
            
            if max_stars is None:
                if stars >= min_stars:
                    return strata_name
            else:
                if min_stars <= stars < max_stars:
                    return strata_name
        return None
    
    def is_strata_complete(self, strata_name: str) -> bool:
        """Vérifier si une strate a atteint son objectif"""
        return self.targets[strata_name]['collected'] >= self.targets[strata_name]['target']
    
    def should_select_repo(self, repo: Dict) -> tuple:
        """Décider si on doit sélectionner ce repo"""
        repo_id = repo['id']
        
        # Éviter les doublons
        if repo_id in self.seen_ids:
            return False, "Déjà vu"
        
        stars = repo['stargazers_count']
        strata = self.get_strata_for_stars(stars)
        
        if not strata:
            return False, "Strate non trouvée"
        
        if self.is_strata_complete(strata):
            return False, f"Strate {strata} complète"
        
        # Privilégier les repos avec activité récente
        if not self.is_recently_active(repo):
            # On peut quand même accepter si la strate n'est pas pleine
            pass
        
        return True, strata
    
    def search_repos(self, query: str, page: int = 1) -> Optional[List[Dict]]:
        """Recherche GitHub"""
        for attempt in range(3):
            try:
                params = {
                    'q': query,
                    'page': page,
                    'per_page': 100,
                    'sort': 'stars',
                    'order': 'desc'
                }
                
                response = requests.get(
                    f'{Config.BASE_URL}/search/repositories',
                    headers=self.get_headers(),
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json().get('items', [])
                
                elif response.status_code == 403:
                    remaining, reset_time = self.check_rate_limit()
                    if remaining == 0 and reset_time:
                        wait_time = reset_time - time.time() + 10
                        if wait_time > 0:
                            print(f"⏳ Rate limit, attente {wait_time/60:.1f} min...")
                            time.sleep(wait_time)
                        continue
                
                elif response.status_code == 422:
                    return None
                
                else:
                    print(f"⚠️ Erreur API {response.status_code}")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"❌ Exception: {e}")
                time.sleep(5)
        
        return None
    
    def build_query_for_strata(self, strata_name: str, language: Optional[str] = None) -> str:
        """Construire une query pour une strate spécifique"""
        target_info = self.targets[strata_name]
        min_stars = target_info['min_stars']
        max_stars = target_info['max_stars']
        
        # Base de la query avec activité récente
        cutoff_date = (datetime.now() - timedelta(days=Config.RECENT_ACTIVITY_DAYS)).strftime('%Y-%m-%d')
        query_parts = [f'pushed:>{cutoff_date}']
        
        # Ajouter filtre de stars
        if max_stars is None:
            query_parts.append(f'stars:>={min_stars}')
        else:
            query_parts.append(f'stars:{min_stars}..{max_stars}')
        
        # Ajouter langage si spécifié
        if language:
            query_parts.append(f'language:{language}')
        
        return ' '.join(query_parts)
    
    def select_for_strata(self, strata_name: str):
        """Sélectionner des repos pour une strate"""
        target_info = self.targets[strata_name]
        
        print(f"\n{'='*70}")
        print(f"🎯 STRATE: {strata_name}")
        print(f"   Objectif: {target_info['target']}")
        print(f"{'='*70}")
        
        # Essayer avec différents langages
        languages_to_try = Config.LANGUAGES + [None]
        
        for lang in languages_to_try:
            if self.is_strata_complete(strata_name):
                break
            
            if len(self.selected_repos) >= Config.TARGET_REPOS:
                break
            
            query = self.build_query_for_strata(strata_name, lang)
            lang_name = lang if lang else "ALL"
            
            page = 1
            repos_from_lang = 0
            
            while page <= 10 and repos_from_lang < 200:
                if self.is_strata_complete(strata_name):
                    break
                
                repos = self.search_repos(query, page)
                
                if not repos:
                    break
                
                for repo in repos:
                    should_select, reason = self.should_select_repo(repo)
                    
                    if not should_select:
                        continue
                    
                    # Extraire les infos essentielles
                    repo_info = {
                        'id': repo['id'],
                        'full_name': repo['full_name'],
                        'owner': repo['owner']['login'],
                        'name': repo['name'],
                        'language': repo.get('language'),
                        'created_at': repo['created_at'],
                        'pushed_at': repo.get('pushed_at'),
                        'initial_stars': repo['stargazers_count'],
                        'initial_forks': repo['forks_count'],
                        'initial_watchers': repo['watchers_count'],
                        'initial_open_issues': repo['open_issues_count'],
                        'strata': reason,
                        'selected_at': datetime.now().isoformat()
                    }
                    
                    self.selected_repos.append(repo_info)
                    self.seen_ids.add(repo['id'])
                    self.targets[reason]['collected'] += 1
                    repos_from_lang += 1
                    
                    # Afficher progression
                    if len(self.selected_repos) % 100 == 0:
                        progress = (len(self.selected_repos) / Config.TARGET_REPOS) * 100
                        print(f"  📊 {len(self.selected_repos)}/{Config.TARGET_REPOS} ({progress:.1f}%)")
                
                page += 1
                time.sleep(1)
            
            if repos_from_lang > 0:
                print(f"  ✅ {lang_name}: +{repos_from_lang} repos")
            time.sleep(2)
    
    def run_selection(self):
        """Sélectionner les 10k repos"""
        print("\n" + "="*70)
        print("🚀 SÉLECTION DES 10K REPOS À SUIVRE")
        print(f"🎯 Objectif: {Config.TARGET_REPOS} repos")
        print(f"📅 Activité récente: < {Config.RECENT_ACTIVITY_DAYS} jours")
        print("\n📊 Répartition cible:")
        for strata_name, target_info in self.targets.items():
            strata_config = Config.STAR_STRATA[strata_name]
            print(f"   {strata_name}: {target_info['target']} repos ({strata_config['percentage']*100:.0f}%)")
        print("="*70 + "\n")
        
        # Sélectionner par strate
        strata_order = ['100-1000', '10-100', '1000+', '0-10']
        
        for strata_name in strata_order:
            if len(self.selected_repos) >= Config.TARGET_REPOS:
                break
            
            self.select_for_strata(strata_name)
        
        # Sauvegarder
        self.save_selection()
        
        # Stats finales
        self.print_stats()
    
    def save_selection(self):
        """Sauvegarder la liste des repos sélectionnés"""
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_repos': len(self.selected_repos),
                'target': Config.TARGET_REPOS,
                'activity_filter_days': Config.RECENT_ACTIVITY_DAYS
            },
            'strata_distribution': {
                strata: {
                    'target': info['target'],
                    'collected': info['collected']
                }
                for strata, info in self.targets.items()
            },
            'repos': self.selected_repos
        }
        
        with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Liste sauvegardée dans: {Config.OUTPUT_FILE}")
        print(f"   {len(self.selected_repos)} repos prêts à être suivis")
    
    def print_stats(self):
        """Afficher les statistiques"""
        print("\n" + "="*70)
        print(f"🎉 SÉLECTION TERMINÉE!")
        print(f"   Repos sélectionnés: {len(self.selected_repos)} / {Config.TARGET_REPOS}")
        
        # Stats par strate
        print(f"\n⭐ Répartition par strate:")
        for strata_name in ['0-10', '10-100', '100-1000', '1000+']:
            info = self.targets[strata_name]
            collected = info['collected']
            target = info['target']
            percentage = (collected / target * 100) if target > 0 else 0
            print(f"   {strata_name:10s}: {collected:5d} / {target:5d} ({percentage:5.1f}%)")
        
        # Stats activité
        with_activity = sum(1 for r in self.selected_repos if self.is_recently_active({'pushed_at': r.get('pushed_at')}))
        if len(self.selected_repos) > 0:
            activity_pct = (with_activity / len(self.selected_repos)) * 100
            print(f"\n📅 Activité récente (< {Config.RECENT_ACTIVITY_DAYS}j):")
            print(f"   Avec activité: {with_activity} ({activity_pct:.1f}%)")
        
        # Stats par langage
        lang_counts = {}
        for repo in self.selected_repos:
            lang = repo.get('language', 'Unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        print(f"\n📊 Top 10 langages:")
        sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs[:10]:
            percentage = (count / len(self.selected_repos) * 100)
            print(f"   {lang:15s}: {count:5d} ({percentage:5.1f}%)")
        
        print("="*70)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("🎯 ÉTAPE 1 : Sélection des repos à suivre")
    print(f"⏰ Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Vérifier variables
    if not Config.GITHUB_TOKENS or Config.GITHUB_TOKENS == ['']:
        print("❌ ERREUR: Variable GITHUB_TOKENS manquante")
        print("   Export: export GITHUB_TOKENS='token1,token2,token3'")
        return 1
    
    try:
        selector = RepoSelector()
        selector.run_selection()
        
        print("\n✅ Prochaine étape:")
        print("   Lance le script quotidien pour collecter les données")
        print("   → python3 daily_collector.py")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interruption manuelle")
        if selector.selected_repos:
            selector.save_selection()
        return 130
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
