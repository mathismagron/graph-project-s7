import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# Définition des portées de communication (en mètres)
PORTEES = {
    'courte': 20000,    # 20 km
    'moyenne': 40000,   # 40 km
    'longue': 60000     # 60 km
}

# Fichiers de topologie pour les trois densités
FICHIERS_TOPOLOGIE = {
    'low': 'Topology/topology_low.csv',
    'avg': 'Topology/topology_avg.csv',
    'high': 'Topology/topology_high.csv'
}

def charger_satellites(fichier):
    """Charge les positions des satellites depuis un fichier CSV."""
    df = pd.read_csv(fichier)
    return df

def calculer_distance(pos1, pos2):
    """Calcule la distance euclidienne entre deux positions 3D."""
    return np.sqrt((pos1[0] - pos2[0])**2 + 
                   (pos1[1] - pos2[1])**2 + 
                   (pos1[2] - pos2[2])**2)

def construire_graphe(df, portee):
    """
    Construit un graphe où les satellites sont des noeuds et 
    les arêtes existent si la distance entre deux satellites est <= portée.
    """
    G = nx.Graph()
    
    # Ajouter les noeuds avec leurs positions
    for _, row in df.iterrows():
        G.add_node(row['sat_id'], pos=(row['x'], row['y'], row['z']))
    
    # Ajouter les arêtes si la distance est inférieure à la portée
    positions = df[['sat_id', 'x', 'y', 'z']].values
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos1 = (positions[i][1], positions[i][2], positions[i][3])
            pos2 = (positions[j][1], positions[j][2], positions[j][3])
            distance = calculer_distance(pos1, pos2)
            
            if distance <= portee:
                G.add_edge(int(positions[i][0]), int(positions[j][0]))
    
    return G

# =============================================================================
# ANALYSE DES DEGRÉS
# =============================================================================

def analyser_degres(G):
    """Analyse les degrés du graphe."""
    degres = dict(G.degree())
    liste_degres = list(degres.values())
    
    if len(liste_degres) == 0:
        return {
            'degre_moyen': 0,
            'degre_min': 0,
            'degre_max': 0,
            'ecart_type': 0,
            'distribution': {}
        }
    
    return {
        'degre_moyen': np.mean(liste_degres),
        'degre_min': min(liste_degres),
        'degre_max': max(liste_degres),
        'ecart_type': np.std(liste_degres),
        'distribution': Counter(liste_degres)
    }

# =============================================================================
# ANALYSE DU CLUSTERING
# =============================================================================

def analyser_clustering(G):
    """Analyse le coefficient de clustering du graphe."""
    clustering = nx.clustering(G)
    liste_clustering = list(clustering.values())
    
    if len(liste_clustering) == 0:
        return {
            'clustering_moyen': 0,
            'clustering_min': 0,
            'clustering_max': 0,
            'ecart_type': 0,
            'distribution': {}
        }
    
    # Distribution du clustering (arrondi à 2 décimales pour regroupement)
    clustering_arrondi = [round(c, 2) for c in liste_clustering]
    
    return {
        'clustering_moyen': np.mean(liste_clustering),
        'clustering_min': min(liste_clustering),
        'clustering_max': max(liste_clustering),
        'ecart_type': np.std(liste_clustering),
        'distribution': Counter(clustering_arrondi),
        'valeurs': liste_clustering
    }

# =============================================================================
# ANALYSE DES CLIQUES
# =============================================================================

def analyser_cliques(G):
    """Analyse les cliques du graphe."""
    # Trouver toutes les cliques maximales
    cliques = list(nx.find_cliques(G))
    
    if len(cliques) == 0:
        return {
            'nombre_cliques': 0,
            'taille_max_clique': 0,
            'distribution_tailles': {},
            'cliques_par_taille': {}
        }
    
    # Tailles des cliques
    tailles = [len(c) for c in cliques]
    distribution = Counter(tailles)
    
    # Cliques par taille
    cliques_par_taille = defaultdict(list)
    for c in cliques:
        cliques_par_taille[len(c)].append(c)
    
    return {
        'nombre_cliques': len(cliques),
        'taille_max_clique': max(tailles),
        'taille_min_clique': min(tailles),
        'taille_moyenne': np.mean(tailles),
        'distribution_tailles': dict(distribution),
        'cliques_par_taille': dict(cliques_par_taille)
    }

# =============================================================================
# ANALYSE DES COMPOSANTES CONNEXES
# =============================================================================

def analyser_composantes_connexes(G):
    """Analyse les composantes connexes du graphe."""
    composantes = list(nx.connected_components(G))
    
    if len(composantes) == 0:
        return {
            'nombre_composantes': 0,
            'tailles': [],
            'distribution_tailles': {},
            'plus_grande': 0
        }
    
    tailles = sorted([len(c) for c in composantes], reverse=True)
    
    return {
        'nombre_composantes': len(composantes),
        'tailles': tailles,
        'distribution_tailles': Counter(tailles),
        'plus_grande': max(tailles),
        'plus_petite': min(tailles),
        'taille_moyenne': np.mean(tailles)
    }

# =============================================================================
# ANALYSE DES PLUS COURTS CHEMINS
# =============================================================================

def analyser_plus_courts_chemins(G):
    """Analyse les plus courts chemins du graphe."""
    resultats = {
        'longueurs': [],
        'distribution': {},
        'nombre_chemins_par_longueur': {},
        'diametre': None,
        'longueur_moyenne': None,
        'rayon': None,
        'details_par_composante': []
    }
    
    if G.number_of_edges() == 0:
        return resultats
    
    toutes_longueurs = []
    
    # Analyser chaque composante connexe
    for idx, composante in enumerate(nx.connected_components(G)):
        sous_graphe = G.subgraph(composante).copy()
        
        if sous_graphe.number_of_nodes() < 2:
            continue
        
        # Calculer tous les plus courts chemins dans cette composante
        longueurs_composante = dict(nx.all_pairs_shortest_path_length(sous_graphe))
        
        # Extraire toutes les longueurs (sauf 0 pour les chemins vers soi-même)
        for source, destinations in longueurs_composante.items():
            for dest, longueur in destinations.items():
                if source < dest:  # Éviter les doublons (graphe non orienté)
                    toutes_longueurs.append(longueur)
        
        # Statistiques de la composante
        if len(composante) > 1:
            try:
                diametre_comp = nx.diameter(sous_graphe)
                rayon_comp = nx.radius(sous_graphe)
                centre_comp = nx.center(sous_graphe)
                resultats['details_par_composante'].append({
                    'taille': len(composante),
                    'diametre': diametre_comp,
                    'rayon': rayon_comp,
                    'centre': list(centre_comp)
                })
            except:
                pass
    
    if toutes_longueurs:
        resultats['longueurs'] = toutes_longueurs
        resultats['distribution'] = Counter(toutes_longueurs)
        resultats['longueur_moyenne'] = np.mean(toutes_longueurs)
        resultats['longueur_max'] = max(toutes_longueurs)
        resultats['longueur_min'] = min(toutes_longueurs)
        resultats['ecart_type'] = np.std(toutes_longueurs)
        
        # Nombre de chemins par longueur
        resultats['nombre_chemins_par_longueur'] = dict(Counter(toutes_longueurs))
        
        # Diamètre global (de la plus grande composante connexe)
        if resultats['details_par_composante']:
            plus_grande = max(resultats['details_par_composante'], key=lambda x: x['taille'])
            resultats['diametre'] = plus_grande['diametre']
            resultats['rayon'] = plus_grande['rayon']
    
    return resultats

# =============================================================================
# FONCTION PRINCIPALE D'ANALYSE
# =============================================================================

def analyser_graphe_complet(G, nom_config):
    """Effectue une analyse complète du graphe."""
    print(f"\n{'='*80}")
    print(f"ANALYSE COMPLÈTE: {nom_config}")
    print(f"{'='*80}")
    
    # Informations de base
    print(f"\n--- INFORMATIONS GÉNÉRALES ---")
    print(f"Nombre de noeuds: {G.number_of_nodes()}")
    print(f"Nombre d'arêtes: {G.number_of_edges()}")
    print(f"Densité du graphe: {nx.density(G):.4f}")
    
    # Analyse des degrés
    print(f"\n--- ANALYSE DES DEGRÉS ---")
    degres = analyser_degres(G)
    print(f"Degré moyen: {degres['degre_moyen']:.2f}")
    print(f"Degré min: {degres['degre_min']}")
    print(f"Degré max: {degres['degre_max']}")
    print(f"Écart-type: {degres['ecart_type']:.2f}")
    print(f"Distribution des degrés: {dict(sorted(degres['distribution'].items()))}")
    
    # Analyse du clustering
    print(f"\n--- ANALYSE DU CLUSTERING ---")
    clustering = analyser_clustering(G)
    print(f"Coefficient de clustering moyen: {clustering['clustering_moyen']:.4f}")
    print(f"Clustering min: {clustering['clustering_min']:.4f}")
    print(f"Clustering max: {clustering['clustering_max']:.4f}")
    print(f"Écart-type: {clustering['ecart_type']:.4f}")
    
    # Analyse des cliques
    print(f"\n--- ANALYSE DES CLIQUES ---")
    cliques = analyser_cliques(G)
    print(f"Nombre de cliques maximales: {cliques['nombre_cliques']}")
    print(f"Taille de la plus grande clique: {cliques['taille_max_clique']}")
    if cliques['nombre_cliques'] > 0:
        print(f"Taille moyenne des cliques: {cliques['taille_moyenne']:.2f}")
        print(f"Distribution des tailles de cliques: {dict(sorted(cliques['distribution_tailles'].items()))}")
    
    # Analyse des composantes connexes
    print(f"\n--- ANALYSE DES COMPOSANTES CONNEXES ---")
    composantes = analyser_composantes_connexes(G)
    print(f"Nombre de composantes connexes: {composantes['nombre_composantes']}")
    print(f"Taille de la plus grande composante: {composantes['plus_grande']}")
    print(f"Taille de la plus petite composante: {composantes['plus_petite']}")
    print(f"Taille moyenne des composantes: {composantes['taille_moyenne']:.2f}")
    print(f"Distribution des tailles: {dict(sorted(composantes['distribution_tailles'].items(), reverse=True))}")
    
    # Analyse des plus courts chemins
    print(f"\n--- ANALYSE DES PLUS COURTS CHEMINS ---")
    chemins = analyser_plus_courts_chemins(G)
    if chemins['longueurs']:
        print(f"Nombre total de paires connectées: {len(chemins['longueurs'])}")
        print(f"Longueur moyenne des plus courts chemins: {chemins['longueur_moyenne']:.2f} sauts")
        print(f"Longueur min: {chemins['longueur_min']} saut(s)")
        print(f"Longueur max: {chemins['longueur_max']} sauts")
        print(f"Écart-type: {chemins['ecart_type']:.2f}")
        if chemins['diametre']:
            print(f"Diamètre (plus grande composante): {chemins['diametre']}")
            print(f"Rayon (plus grande composante): {chemins['rayon']}")
        print(f"Distribution des longueurs: {dict(sorted(chemins['nombre_chemins_par_longueur'].items()))}")
    else:
        print("Aucun chemin (graphe sans arêtes ou noeuds isolés uniquement)")
    
    return {
        'degres': degres,
        'clustering': clustering,
        'cliques': cliques,
        'composantes': composantes,
        'chemins': chemins
    }

# =============================================================================
# VISUALISATIONS
# =============================================================================

def visualiser_distributions(resultats_tous, nom_config):
    """Crée les visualisations des distributions pour une configuration."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Distributions - {nom_config}', fontsize=14)
    
    for idx, (nom_portee, resultats) in enumerate(resultats_tous.items()):
        col = idx
        
        # Distribution des degrés
        ax1 = axes[0, col]
        degres = resultats['degres']['distribution']
        if degres:
            ax1.bar(degres.keys(), degres.values(), color='steelblue', alpha=0.7)
            ax1.axvline(resultats['degres']['degre_moyen'], color='red', linestyle='--', 
                       label=f"Moyenne: {resultats['degres']['degre_moyen']:.1f}")
            ax1.set_xlabel('Degré')
            ax1.set_ylabel('Fréquence')
            ax1.set_title(f'Distribution des degrés\nPortée {nom_portee}')
            ax1.legend()
        
        # Distribution des plus courts chemins
        ax2 = axes[1, col]
        chemins = resultats['chemins']['nombre_chemins_par_longueur']
        if chemins:
            ax2.bar(chemins.keys(), chemins.values(), color='forestgreen', alpha=0.7)
            if resultats['chemins']['longueur_moyenne']:
                ax2.axvline(resultats['chemins']['longueur_moyenne'], color='red', linestyle='--',
                           label=f"Moyenne: {resultats['chemins']['longueur_moyenne']:.1f}")
            ax2.set_xlabel('Longueur (sauts)')
            ax2.set_ylabel('Nombre de chemins')
            ax2.set_title(f'Distribution des plus courts chemins\nPortée {nom_portee}')
            ax2.legend()
    
    plt.tight_layout()
    return fig

def visualiser_clustering(resultats_tous, nom_config):
    """Visualise la distribution du clustering."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Distribution du coefficient de clustering - {nom_config}', fontsize=14)
    
    for idx, (nom_portee, resultats) in enumerate(resultats_tous.items()):
        ax = axes[idx]
        
        if 'valeurs' in resultats['clustering'] and resultats['clustering']['valeurs']:
            valeurs = resultats['clustering']['valeurs']
            ax.hist(valeurs, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(resultats['clustering']['clustering_moyen'], color='red', linestyle='--',
                      label=f"Moyenne: {resultats['clustering']['clustering_moyen']:.3f}")
            ax.set_xlabel('Coefficient de clustering')
            ax.set_ylabel('Fréquence')
            ax.set_title(f'Portée {nom_portee}')
            ax.legend()
    
    plt.tight_layout()
    return fig

def visualiser_cliques(resultats_tous, nom_config):
    """Visualise la distribution des cliques."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Distribution des tailles de cliques maximales - {nom_config}', fontsize=14)
    
    for idx, (nom_portee, resultats) in enumerate(resultats_tous.items()):
        ax = axes[idx]
        
        distribution = resultats['cliques']['distribution_tailles']
        if distribution:
            ax.bar(distribution.keys(), distribution.values(), color='orange', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Taille de clique')
            ax.set_ylabel('Nombre de cliques')
            ax.set_title(f'Portée {nom_portee}\n({resultats["cliques"]["nombre_cliques"]} cliques)')
    
    plt.tight_layout()
    return fig

# =============================================================================
# GÉNÉRATION DU RAPPORT
# =============================================================================

def generer_rapport_complet():
    """Génère un rapport complet pour toutes les configurations."""
    
    os.makedirs('resultats', exist_ok=True)
    
    tous_resultats = {}
    tableau_recap = []
    
    for nom_densite, fichier in FICHIERS_TOPOLOGIE.items():
        if not os.path.exists(fichier):
            print(f"Fichier {fichier} non trouvé, ignoré.")
            continue
        
        print(f"\n{'#'*80}")
        print(f"# TRAITEMENT DE LA DENSITÉ: {nom_densite.upper()}")
        print(f"{'#'*80}")
        
        df = charger_satellites(fichier)
        resultats_densite = {}
        
        for nom_portee, portee in PORTEES.items():
            G = construire_graphe(df, portee)
            nom_config = f"Densité {nom_densite} - Portée {nom_portee} ({portee/1000:.0f} km)"
            
            resultats = analyser_graphe_complet(G, nom_config)
            resultats_densite[nom_portee] = resultats
            
            # Ajouter au tableau récapitulatif
            tableau_recap.append({
                'Densité': nom_densite,
                'Portée': f"{nom_portee} ({portee/1000:.0f}km)",
                'Noeuds': G.number_of_nodes(),
                'Arêtes': G.number_of_edges(),
                'Degré moyen': f"{resultats['degres']['degre_moyen']:.2f}",
                'Degré max': resultats['degres']['degre_max'],
                'Clustering moyen': f"{resultats['clustering']['clustering_moyen']:.4f}",
                'Nb cliques': resultats['cliques']['nombre_cliques'],
                'Taille max clique': resultats['cliques']['taille_max_clique'],
                'Nb composantes': resultats['composantes']['nombre_composantes'],
                'Plus grande comp.': resultats['composantes']['plus_grande'],
                'Chemin moyen': f"{resultats['chemins']['longueur_moyenne']:.2f}" if resultats['chemins']['longueur_moyenne'] else 'N/A',
                'Diamètre': resultats['chemins']['diametre'] if resultats['chemins']['diametre'] else 'N/A'
            })
        
        tous_resultats[nom_densite] = resultats_densite
        
        # Générer les visualisations pour cette densité
        fig_dist = visualiser_distributions(resultats_densite, f"Densité {nom_densite}")
        fig_dist.savefig(f'resultats/distributions_{nom_densite}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_dist)
        
        fig_clust = visualiser_clustering(resultats_densite, f"Densité {nom_densite}")
        fig_clust.savefig(f'resultats/clustering_{nom_densite}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_clust)
        
        fig_cliques = visualiser_cliques(resultats_densite, f"Densité {nom_densite}")
        fig_cliques.savefig(f'resultats/cliques_{nom_densite}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_cliques)
    
    # Sauvegarder le tableau récapitulatif
    df_recap = pd.DataFrame(tableau_recap)
    print("\n" + "="*120)
    print("TABLEAU RÉCAPITULATIF COMPLET")
    print("="*120)
    print(df_recap.to_string(index=False))
    df_recap.to_csv('resultats/analyse_complete.csv', index=False)
    
    return tous_resultats, df_recap

def generer_figure_comparative_complete():
    """Génère une figure comparative avec toutes les métriques."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Comparaison des métriques selon densité et portée', fontsize=14)
    
    metriques = {
        'Degré moyen': [],
        'Clustering moyen': [],
        'Nb cliques': [],
        'Taille max clique': [],
        'Nb composantes': [],
        'Chemin moyen': []
    }
    
    labels = []
    
    for nom_densite, fichier in FICHIERS_TOPOLOGIE.items():
        if not os.path.exists(fichier):
            continue
        
        df = charger_satellites(fichier)
        
        for nom_portee, portee in PORTEES.items():
            G = construire_graphe(df, portee)
            
            degres = analyser_degres(G)
            clustering = analyser_clustering(G)
            cliques = analyser_cliques(G)
            composantes = analyser_composantes_connexes(G)
            chemins = analyser_plus_courts_chemins(G)
            
            labels.append(f"{nom_densite}\n{nom_portee}")
            metriques['Degré moyen'].append(degres['degre_moyen'])
            metriques['Clustering moyen'].append(clustering['clustering_moyen'])
            metriques['Nb cliques'].append(cliques['nombre_cliques'])
            metriques['Taille max clique'].append(cliques['taille_max_clique'])
            metriques['Nb composantes'].append(composantes['nombre_composantes'])
            metriques['Chemin moyen'].append(chemins['longueur_moyenne'] if chemins['longueur_moyenne'] else 0)
    
    # Créer les graphiques
    couleurs = ['#1f77b4', '#2ca02c', '#d62728'] * 3  # 3 couleurs pour 3 portées, répétées
    
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    titres = ['Degré moyen', 'Clustering moyen', 'Nb cliques', 
              'Taille max clique', 'Nb composantes', 'Chemin moyen (sauts)']
    cles = ['Degré moyen', 'Clustering moyen', 'Nb cliques',
            'Taille max clique', 'Nb composantes', 'Chemin moyen']
    
    for (i, j), titre, cle in zip(positions, titres, cles):
        ax = axes[i, j]
        bars = ax.bar(range(len(labels)), metriques[cle], color=couleurs[:len(labels)], alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(titre)
        ax.grid(axis='y', alpha=0.3)
    
    # Supprimer les axes non utilisés
    for i in range(2, 3):
        for j in range(3):
            if (i, j) not in positions:
                axes[i, j].axis('off')
    
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('resultats/comparaison_metriques.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure comparative sauvegardée: resultats/comparaison_metriques.png")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ANALYSE DES GRAPHES NON VALUÉS - PARTIE 2")
    print("="*80)
    
    # Générer le rapport complet
    resultats, tableau = generer_rapport_complet()
    
    # Générer la figure comparative
    generer_figure_comparative_complete()
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE!")
    print("Fichiers générés dans le dossier 'resultats/':")
    print("  - analyse_complete.csv (tableau récapitulatif)")
    print("  - distributions_*.png (distributions degrés et chemins)")
    print("  - clustering_*.png (distribution clustering)")
    print("  - cliques_*.png (distribution tailles cliques)")
    print("  - comparaison_metriques.png (comparaison globale)")
    print("="*80)