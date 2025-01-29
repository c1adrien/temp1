# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:11:51 2025

@author: rehmi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def davies_harte_fbm(n, H, random_state=None):
    """
    Génère un chemin de fBm (mouvement brownien fractionnaire)
    de longueur n, exponent H, via la méthode Davies-Harte.
    Retourne un tableau 1D : [B^H(0), B^H(1), ..., B^H(n-1)].
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    def Gamma(k, H):
        if k == 0:
            return 1.0
        return 0.5 * ((k+1)**(2*H) - 2*(k**(2*H)) + (k-1)**(2*H))
    
    G = [Gamma(k, H) for k in range(n)]
    GExtended = np.zeros(2*n)
    GExtended[:n] = G
    GExtended[n:2*n-1] = G[1:][::-1]  
    
    fftG = np.fft.fft(GExtended)
    Re_fftG = np.real(fftG)
    Re_fftG[Re_fftG < 0] = 0.0  
    
    Z = np.random.randn(2*n) + 1j * np.random.randn(2*n)
    W = np.fft.ifft(Z * np.sqrt(Re_fftG)).real
    fGn = W[:n]
    
    fbm = np.cumsum(fGn)
    return fbm

def simulate_future_trajectories_fbm(real_path, T, H, N, Hurst=0.7, seed=123):
    np.random.seed(seed)
    yT = real_path[T]  # la valeur réelle à l'instant T
    trajectories = np.zeros((N, H))
    
    for i in range(N):
        # On génère un fBm (longueur T+H)
        fbm_temp = davies_harte_fbm(T+H, Hurst)
        # On recale le chemin pour que la valeur à T coïncide avec yT
        shift = yT - fbm_temp[T]
        fbm_temp_shifted = fbm_temp + shift
        
        # On prend la portion [T..T+H-1]
        # => ce sont les H points "futurs"
        future_part = fbm_temp_shifted[T:T+H]
        trajectories[i,:] = future_part
    
    return trajectories

def quantize_trajectories(trajectories, K, seed=123):
    """
    Applique un k-means (K clusters) sur 'trajectories' (shape: (N, H))
    et renvoie les centroïdes de dimension H, ainsi que les labels.
    """
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=seed)
    kmeans.fit(trajectories)
    centers = kmeans.cluster_centers_  # shape : (K, H)
    labels = kmeans.labels_
    return centers, labels

def main():
    n_samples = 1500
    Hurst = 0.9  # exponent de Hurst > 0.5 => corrélation persistante
    real_path = davies_harte_fbm(n_samples, H=Hurst, random_state=42)
    
    T = 1000
    H = 150
    # => La "vraie" trajectoire réelle s'étend de t=0 à t=299,
    #    on va regarder le futur à partir de t=200 jusqu'à t=249
    
    N = 2000
    future_trajs = simulate_future_trajectories_fbm(real_path, T, H, N, Hurst=Hurst, seed=123)
    
    K = 5
    centers, labels = quantize_trajectories(future_trajs, K=K, seed=123)
    
    plt.figure(figsize=(10,6))
    plt.plot(range(T), real_path[:T], label="Trajectoire réelle (passé)", color="blue")
    plt.plot(range(T, T+H), real_path[T:T+H], label="Trajectoire réelle (futur)", color="black", linewidth=2)
    for k in range(K):
        plt.plot(range(T, T+H), centers[k,:],
                 label=f"Centroïde {k+1}",
                 linestyle="--", linewidth=1.5)
    plt.title(f"Mouvement brownien fractionnaire (H={Hurst})\n"
              f"Quantization des trajectoires futures (T={T}, H={H}, K={K})")
    plt.xlabel("Temps")
    plt.ylabel("Valeur du processus")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Valeur réelle à T={T}: {real_path[T]:.4f}")
    print(f"Exemple d'un centroïde (premiers points) : {centers[0,:5]} ...")

if __name__ == "__main__":
    main()
