import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from itertools import combinations
import os

# Définition des portées de communication (en mètres)
PORTEES = {
    'courte': 20000,    # 20 km
    'moyenne': 40000,  # 40 km
    'longue': 60000    # 60 km
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
                G.add_edge(int(positions[i][0]), int(positions[j][0]), weight=1)
    
    return G

def afficher_graphe_3d(G, df, titre, ax):
    """Affiche le graphe en 3D."""
    # Extraire les positions
    positions = {row['sat_id']: (row['x'], row['y'], row['z']) for _, row in df.iterrows()}
    
    # Dessiner les noeuds
    xs = [positions[node][0] for node in G.nodes()]
    ys = [positions[node][1] for node in G.nodes()]
    zs = [positions[node][2] for node in G.nodes()]
    
    ax.scatter(xs, ys, zs, c='blue', s=50, alpha=0.8, label='Satellites')
    
    # Dessiner les arêtes
    for edge in G.edges():
        x = [positions[edge[0]][0], positions[edge[1]][0]]
        y = [positions[edge[0]][1], positions[edge[1]][1]]
        z = [positions[edge[0]][2], positions[edge[1]][2]]
        ax.plot(x, y, z, 'g-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(titre)

def afficher_graphe_2d(G, df, titre, ax):
    """Affiche le graphe en 2D (projection XY)."""
    positions = {row['sat_id']: (row['x'], row['y']) for _, row in df.iterrows()}
    
    # Dessiner avec NetworkX
    nx.draw(G, pos=positions, ax=ax, 
            node_size=30, 
            node_color='blue',
            edge_color='green',
            alpha=0.7,
            width=0.5,
            with_labels=False)
    
    ax.set_title(titre)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

def statistiques_graphe(G, nom_config):
    """Calcule et affiche les statistiques du graphe."""
    print(f"\n{'='*60}")
    print(f"Statistiques pour {nom_config}")
    print(f"{'='*60}")
    print(f"Nombre de noeuds: {G.number_of_nodes()}")
    print(f"Nombre d'arêtes: {G.number_of_edges()}")
    print(f"Densité du graphe: {nx.density(G):.4f}")
    
    if G.number_of_edges() > 0:
        print(f"Degré moyen: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"Nombre de composantes connexes: {nx.number_connected_components(G)}")
        
        if nx.is_connected(G):
            print(f"Le graphe est connexe")
            print(f"Diamètre: {nx.diameter(G)}")
        else:
            print(f"Le graphe n'est pas connexe")
            # Taille de la plus grande composante connexe
            largest_cc = max(nx.connected_components(G), key=len)
            print(f"Taille de la plus grande composante: {len(largest_cc)}")
    else:
        print("Aucune arête dans le graphe")
    
    return {
        'noeuds': G.number_of_nodes(),
        'aretes': G.number_of_edges(),
        'densite': nx.density(G),
        'connexe': nx.is_connected(G) if G.number_of_edges() > 0 else False
    }

def visualiser_toutes_configurations():
    """Génère les visualisations pour toutes les configurations."""
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs('resultats', exist_ok=True)
    
    # Pour chaque densité
    for nom_densite, fichier in FICHIERS_TOPOLOGIE.items():
        if not os.path.exists(fichier):
            print(f"Fichier {fichier} non trouvé, ignoré.")
            continue
            
        print(f"\n{'#'*60}")
        print(f"Traitement de la configuration: {nom_densite}")
        print(f"{'#'*60}")
        
        df = charger_satellites(fichier)
        
        # Figure pour les visualisations 3D
        fig_3d = plt.figure(figsize=(18, 6))
        fig_3d.suptitle(f'Essaim de satellites - Densité {nom_densite.upper()}', fontsize=14)
        
        # Figure pour les visualisations 2D
        fig_2d = plt.figure(figsize=(18, 6))
        fig_2d.suptitle(f'Graphe de communication (projection 2D) - Densité {nom_densite.upper()}', fontsize=14)
        
        for idx, (nom_portee, portee) in enumerate(PORTEES.items()):
            # Construire le graphe
            G = construire_graphe(df, portee)
            
            # Statistiques
            statistiques_graphe(G, f"Densité {nom_densite} - Portée {nom_portee} ({portee/1000:.0f} km)")
            
            # Visualisation 3D
            ax_3d = fig_3d.add_subplot(1, 3, idx + 1, projection='3d')
            afficher_graphe_3d(G, df, f'Portée {nom_portee} ({portee/1000:.0f} km)\n{G.number_of_edges()} liens', ax_3d)
            
            # Visualisation 2D
            ax_2d = fig_2d.add_subplot(1, 3, idx + 1)
            afficher_graphe_2d(G, df, f'Portée {nom_portee} ({portee/1000:.0f} km)\n{G.number_of_edges()} liens', ax_2d)
        
        # Sauvegarder les figures
        fig_3d.tight_layout()
        fig_3d.savefig(f'resultats/graphe_3d_{nom_densite}.png', dpi=150, bbox_inches='tight')
        
        fig_2d.tight_layout()
        fig_2d.savefig(f'resultats/graphe_2d_{nom_densite}.png', dpi=150, bbox_inches='tight')
        
        plt.close(fig_3d)
        plt.close(fig_2d)
    
    print(f"\n{'='*60}")
    print("Visualisations sauvegardées dans le dossier 'resultats/'")
    print(f"{'='*60}")

def generer_tableau_recapitulatif():
    """Génère un tableau récapitulatif de toutes les configurations."""
    
    resultats = []
    
    for nom_densite, fichier in FICHIERS_TOPOLOGIE.items():
        if not os.path.exists(fichier):
            continue
            
        df = charger_satellites(fichier)
        
        for nom_portee, portee in PORTEES.items():
            G = construire_graphe(df, portee)
            
            resultats.append({
                'Densité': nom_densite,
                'Portée': f"{nom_portee} ({portee/1000:.0f} km)",
                'Noeuds': G.number_of_nodes(),
                'Arêtes': G.number_of_edges(),
                'Densité graphe': f"{nx.density(G):.4f}",
                'Degré moyen': f"{sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}" if G.number_of_edges() > 0 else "0",
                'Connexe': 'Oui' if (G.number_of_edges() > 0 and nx.is_connected(G)) else 'Non',
                'Composantes': nx.number_connected_components(G)
            })
    
    df_resultats = pd.DataFrame(resultats)
    print("\n" + "="*100)
    print("TABLEAU RÉCAPITULATIF")
    print("="*100)
    print(df_resultats.to_string(index=False))
    
    # Sauvegarder en CSV
    os.makedirs('resultats', exist_ok=True)
    df_resultats.to_csv('resultats/recapitulatif.csv', index=False)
    
    return df_resultats

def visualisation_comparative():
    """Crée une figure comparative de toutes les configurations."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Comparaison des graphes de communication\n(Projection 2D)', fontsize=16)
    
    densites = list(FICHIERS_TOPOLOGIE.keys())
    portees_list = list(PORTEES.keys())
    
    for i, (nom_densite, fichier) in enumerate(FICHIERS_TOPOLOGIE.items()):
        if not os.path.exists(fichier):
            continue
            
        df = charger_satellites(fichier)
        
        for j, (nom_portee, portee) in enumerate(PORTEES.items()):
            G = construire_graphe(df, portee)
            ax = axes[i, j]
            
            positions = {row['sat_id']: (row['x'], row['y']) for _, row in df.iterrows()}
            
            # Couleur selon la connectivité
            node_color = 'green' if (G.number_of_edges() > 0 and nx.is_connected(G)) else 'red'
            
            nx.draw(G, pos=positions, ax=ax,
                   node_size=20,
                   node_color=node_color,
                   edge_color='gray',
                   alpha=0.7,
                   width=0.3,
                   with_labels=False)
            
            connexe_str = "✓ Connexe" if (G.number_of_edges() > 0 and nx.is_connected(G)) else "✗ Non connexe"
            ax.set_title(f'Densité: {nom_densite}\nPortée: {nom_portee} ({portee/1000:.0f} km)\n{G.number_of_edges()} liens - {connexe_str}', fontsize=10)
    
    plt.tight_layout()
    os.makedirs('resultats', exist_ok=True)
    plt.savefig('resultats/comparaison_complete.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure comparative sauvegardée: resultats/comparaison_complete.png")

if __name__ == "__main__":
    print("="*60)
    print("MODÉLISATION DE L'ESSAIM DE SATELLITES")
    print("="*60)
    
    # Générer toutes les visualisations
    visualiser_toutes_configurations()
    
    # Générer le tableau récapitulatif
    generer_tableau_recapitulatif()
    
    # Générer la visualisation comparative
    visualisation_comparative()
    
    print("\n" + "="*60)
    print("Traitement terminé!")
    print("="*60)