import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def generate_AR2_series(n_samples=150,
                        alpha=0.5, beta=0.4,
                        sigma=0.05, seed=42):
    """
    Génère une série temporelle y_t de longueur n_samples suivant le modèle :
        y_t = alpha*y_{t-1} + beta*y_{t-2} + epsilon_t
    avec epsilon_t ~ N(0, sigma^2).
    """
    np.random.seed(seed)
    
    y = np.zeros(n_samples)
    # Initialisation (valeurs t=0 et t=1)
    y[0] = np.random.randn() * sigma
    y[1] = np.random.randn() * sigma
    
    # Génération récursive
    for t in range(2, n_samples):
        noise = np.random.randn() * sigma
        y[t] = alpha * y[t-1] + beta * y[t-2] + noise
        
    return y


def simulate_future_trajectories(y, T, H, N,
                                 alpha=0.5, beta=0.4,
                                 sigma=0.05, seed=123):
    """
    À partir du point T-1, T-2 (contexte), simule N trajectoires
    futures sur H pas de temps (T, T+1, ..., T+H-1).
    
    Retourne un tableau de dimension (N, H).
    """
    np.random.seed(seed)
    
    # Contexte
    y_tm1 = y[T-1]  # y_{T-1}
    y_tm2 = y[T-2]  # y_{T-2}
    
    trajectories = np.zeros((N, H))
    
    for i in range(N):
        # On part des deux dernières valeurs connues
        prev_1 = y_tm1
        prev_2 = y_tm2
        
        for h in range(H):
            noise = np.random.randn() * sigma
            # AR(2) : y_t = alpha*y_{t-1} + beta*y_{t-2} + noise
            current = alpha * prev_1 + beta * prev_2 + noise
            trajectories[i, h] = current
            # Décalage pour l'étape suivante
            prev_2 = prev_1
            prev_1 = current
    
    return trajectories


def quantize_trajectories_kmeans(trajectories, K, seed=123):
    """
    Applique un k-means (K clusters) sur 'trajectories' (shape: (N, H))
    et renvoie les centroïdes de dimension H, ainsi que les labels.
    """
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=seed)
    kmeans.fit(trajectories)
    centers = kmeans.cluster_centers_  # shape : (K, H)
    labels = kmeans.labels_
    return centers, labels


def lloyds_algorithm(data, K, max_iter=50, tol=1e-5, seed=42):
    """
    Implémentation from-scratch de Lloyd pour la quantification vectorielle.
    data : (N, d)
    Retourne :
      - centers : (K, d)
      - labels  : (N,)
    """
    np.random.seed(seed)
    N, d = data.shape
    
    # Initialisation : K échantillons aléatoires comme centroïdes initiaux
    indices = np.random.choice(N, K, replace=False)
    centers = data[indices, :].copy()
    
    for it in range(max_iter):
        # 1) Assignation
        dist_sq = np.sum((data[:, None] - centers[None, :])**2, axis=2)
        labels = np.argmin(dist_sq, axis=1)
        
        # 2) Mise à jour des centroïdes
        new_centers = np.zeros_like(centers)
        for k in range(K):
            points_in_cluster = data[labels == k]
            if len(points_in_cluster) > 0:
                new_centers[k] = np.mean(points_in_cluster, axis=0)
            else:
                # si cluster vide, on garde l'ancien
                new_centers[k] = centers[k]
        
        # 3) Vérification convergence
        shift = np.sqrt(np.sum((new_centers - centers)**2))
        centers = new_centers
        if shift < tol:
            print(f"[Lloyd] Convergence à l'itération {it+1}, shift={shift:.6f}")
            break
        
    return centers, labels

########################################################################
# FONCTION DE PLOT
########################################################################

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

def plot_trajectories(t, vecteur, array_analytical, array_kmeans):
    """
    Plots a graph illustrating the historical trajectory and two sets of estimated modes.

    Arguments:
    t : int : cutoff point between historical series and the two sets of modes.
    vecteur : ndarray : historical data trajectory before t (length t).
    array_analytical : ndarray : optimal analytical trajectories (K1 trajectories of length T-t).
    array_kmeans : ndarray : K-means centroid trajectories (K2 trajectories of length T-t).
    """
    T = t + array_analytical.shape[1]  # total length
    time_full = np.linspace(0, 1, T)
    time_future = np.linspace(t / T, 1, T - t)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.set_facecolor('lightgrey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.grid(True, which='both', linewidth=2.0, alpha=1.0, color='white')

    # Historical trajectory
    plt.plot(time_full[:t], vecteur, label="AR(2) historical trajectory", color='black', linewidth=3)

    # Analytical trajectories
    for i, traj in enumerate(array_analytical):
        plt.plot(time_future, traj,
                 label=f"Analytical optimal quantizer" if i < 1 else None,
                 color='brown', alpha=0.8, linewidth=2.5)

    # K-means trajectories
    for i, traj in enumerate(array_kmeans):
        plt.plot(time_future, traj,
                 label=f"WTA forecasts" if i < 1 else None,
                 color='lightcoral', alpha=0.8, linewidth=2.5)

    plt.xlim(0, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

########################################################################
# MAIN
########################################################################

def main():
    # paramètres du modèle AR(2)
    alpha, beta = 0.5, 0.4
    sigma = 0.05
    
    n_samples = 200
    y = generate_AR2_series(n_samples=n_samples,
                            alpha=alpha, beta=beta,
                            sigma=sigma, seed=42)
    
    # choix du cutoff T et de l'horizon H
    T = 100  # moment de coupure
    H = 50   # horizon futur
    K = 5    # nombre de centroïdes
    
    # on genere N trajectoires futures
    N = 2000
    W_future = simulate_future_trajectories(y, T, H, N,
                                            alpha=alpha, beta=beta,
                                            sigma=sigma, seed=123)
    # W_future.shape = (N, H)

    # 1) Lloyd "maison"
    centers_lloyd, _ = lloyds_algorithm(W_future, K=K, max_iter=100, tol=1e-5, seed=123)
    
    # 2) K-Means scikit-learn
    centers_kmeans, _ = quantize_trajectories_kmeans(W_future, K=K, seed=999)
    
    vecteur = y[:T]  # shape = (T,)

    plot_trajectories(
        t = T,
        vecteur = vecteur,
        array_analytical = centers_lloyd,   # "analytical" = lloyd
        array_kmeans     = centers_kmeans   # "kmeans"     = scikit-learn
    )

if __name__ == "__main__":
    main()
