import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import norm
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

#the model we used in the differents notebooks.
class tMCL(nn.Module):
    def __init__(
        self, 
        cond_dim,       # dimension d'entrée (== 1, le point initial)
        nb_step_simulation,  # longueur des trajectoires générées
        n_hypotheses, 
        device,  
        loss_type = "relaxted_wta" 
    ):
        super().__init__()
       
        self.loss_type = loss_type
        self.nb_step_simulation = nb_step_simulation
        self.n_hypotheses = n_hypotheses

        self.backbone = nn.Sequential(
                nn.Linear(cond_dim, 200),
                nn.Linear(200, 200), #maybe not useful
                nn.ReLU(),
                nn.Linear(200, 200)).to(device) #the backbone 
        
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(200, nb_step_simulation)
            ).to(device)
            for _ in range(n_hypotheses)
        ])
       
       
    def forward(self, distr_args):
       
        # (A) Passer dans la backbone => [B, hidden_dim]
        features = self.backbone(distr_args)
        pred_list = []
        for head in self.prediction_heads:
            out = head(features)  # => [B, nb_step_simulation]
            pred_list.append(out)
        prediction_list = torch.stack(pred_list, dim=1)  # => [B, K, nb_step_simulation]
        return prediction_list




    def log_prob(self, target, distr_args):
        target = target.squeeze(-1)  # [B, nb_step_simulation]
        prediction_list = self.forward(distr_args)  # [B, K, nb_step_simulation]
        pairwise_mse = torch.sum((prediction_list - target.unsqueeze(1))**2, dim=-1)  # [B, K]

        if self.loss_type == "relaxted_wta":
            epsilon = 0.05
            n_hypotheses = pairwise_mse.shape[1]
            winner, _ = pairwise_mse.min(dim=1)
            mcl_loss = (1 - epsilon * n_hypotheses / (n_hypotheses - 1)) * winner + \
                       (epsilon / (n_hypotheses - 1)) * pairwise_mse.sum(dim=1)
            mcl_loss_mean = mcl_loss.mean()
            
        else:
            
        
            mcl_loss, _ = pairwise_mse.min(dim=1)
        
            mcl_loss_mean = mcl_loss.mean()

       
        return mcl_loss_mean
    
# Définir les fonctions propres de Karhunen-Loève pour le mouvement brownien
def kl_eigenfunctions(t, T, m):
    """Calcule les m premières fonctions propres et valeurs propres pour le mouvement brownien."""
    eigenfunctions = []
    eigenvalues = []
    for n in range(1, m + 1):
        eigenfunc = np.sqrt(2 / T) * np.sin(np.pi * (n - 0.5) * t / T)
        eigenval = T**2 / (np.pi**2 * (n - 0.5)**2)
        eigenfunctions.append(eigenfunc)
        eigenvalues.append(eigenval)
    return np.array(eigenfunctions), np.array(eigenvalues)

# Générer les points de quantification optimaux pour une loi normale
def generate_quantization_points(N, dist_mean=0, dist_var=1):
    """
    Génère les points optimaux pour quantifier une loi normale centrée.
    """
    std_dev = np.sqrt(dist_var)
    return norm.ppf(np.linspace(1 / (2 * N), 1 - 1 / (2 * N), N)) * std_dev

# Reconstruire les trajectoires quantifiées
def reconstruct_quantized_trajectories(N_levels, eigenfunctions, eigenvalues):
    """
    Reconstruit les trajectoires quantifiées à partir des points de quantification produit.
    """
    m = len(N_levels)
    quantization_grids = [generate_quantization_points(N, dist_var=1) for N in N_levels]
    multi_indices = list(product(*[range(len(grid)) for grid in quantization_grids]))
    
    trajectories = []
    for index in multi_indices:
        trajectory = sum(
            np.sqrt(eigenvalues[n]) * quantization_grids[n][index[n]] * eigenfunctions[n]
            for n in range(m)
        )
        trajectories.append(trajectory)
    
    return np.array(trajectories)


#---- useful for toy example -----
def generate_brownian_motion(batch_size, nb_step_simulation, nb_timesteps_discretisation=300, fixed_start_point=(0.1,0)):
    """
    Génère des trajectoires de mouvement brownien.

    :param batch_size: Nombre de trajectoires à générer
    :param nb_step_simulation: Nombre d'étapes de simulation pour chaque trajectoire
    :param nb_timesteps_discretisation: Nombre de points de discrétisation sur [0,1]
    :param fixed_start_point: Tuple (time, x0) fixe (None pour choisir aléatoirement le point de départ)
    :return: Tenseur de dimension (batch_size, nb_step_simulation, 1) contenant les trajectoires
    """
   

    # Initialisation des trajectoires
    trajectories = []
    time_scale = 1.0 / nb_timesteps_discretisation

    # Déterminer le point de départ si fixé
    if fixed_start_point is not None:
        start_time = fixed_start_point[0]
        current_x0 = fixed_start_point[1]

    for _ in range(batch_size):
        # Si le point de départ n'est pas fixé, en générer un pour chaque trajectoire
        if fixed_start_point is None:
            start_time = np.random.randint(0, nb_timesteps_discretisation - nb_step_simulation + 1)/ nb_timesteps_discretisation
            current_x0 = np.random.normal(0, np.sqrt(start_time))
     

        increments = np.random.normal(loc=0.0, scale=np.sqrt(time_scale), size=nb_step_simulation)

        
        trajectory = np.cumsum(np.insert(increments, 0, current_x0))  

        
        trajectories.append(trajectory[1:].reshape(-1, 1))  # On enlève current_x0 de la liste finale

    
    trajectories_tensor = torch.tensor(np.stack(trajectories), dtype=torch.float32)

    return trajectories_tensor #1 step in the trajectory is a increase of 1/nb_timesteps_discretisation in time


def plot_brownian_trajectories(trajectories, nb_step_simulation,nb_timesteps_discretisation):
    """
    Affiche les trajectoires de mouvement brownien.

    :param trajectories: Tenseur de dimension (batch_size, nb_step_simulation, 1) contenant les trajectoires
    :param nb_step_simulation: Nombre d'étapes de simulation pour chaque trajectoire
    """
    # Vérification de la forme des trajectoires
    if len(trajectories.shape) != 3 or trajectories.shape[2] != 1:
        raise ValueError("Les trajectoires doivent avoir la forme (batch_size, nb_step_simulation, 1)")

    # Conversion des trajectoires en numpy pour le plotting
    trajectories_np = trajectories.squeeze(-1).numpy()

    # Création de la figure
    plt.figure(figsize=(10, 6))

    # Tracé de chaque trajectoire
    time_grid = [i * (1 / nb_timesteps_discretisation) for i in range(nb_step_simulation )]

    for idx, trajectory in enumerate(trajectories_np):
        plt.plot(time_grid, trajectory, label=f'Trajectoire {idx+1}')

    # Personnalisation du graphique
    plt.title("Trajectoires de Mouvement Brownien")
    plt.xlabel("Temps (étapes discrètes)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    # Affichage
    plt.show()




def generate_brownian_bridge_samples(batch_size, nb_discretization_points, interval_length, a=0, b=1):
    """
    Génère des trajectoires complètes de pont brownien sur [0, 1] et extrait un sous-trajet par trajectoire, de longueur fixe.

    :param batch_size: Nombre de sous-trajets à générer
    :param nb_discretization_points: Nombre total de points de discrétisation sur l'intervalle [0, 1]
    :param interval_length: Nombre de points dans chaque sous-trajet (bouts)
    :param a: Valeur initiale du pont brownien à t=0
    :param b: Valeur finale du pont brownien à t=1
    :return:
        - Tenseur de dimension (batch_size, interval_length, 1) contenant les sous-trajets
        - Tenseur de dimension (batch_size, 2) contenant les intervalles temporels [t0, t1] pour chaque sous-trajet
    """
    if interval_length > nb_discretization_points:
        raise ValueError("`interval_length` doit être inférieur ou égal à `nb_discretization_points`.")
    
    samples = []
    time_intervals = []

    for _ in range(batch_size):
        # Générer une trajectoire complète de pont brownien
        t_full = np.linspace(0, 1, nb_discretization_points)
        time_step = 1 / (nb_discretization_points - 1)
        
        increments = np.random.normal(0, np.sqrt(time_step), size=nb_discretization_points)
        brownian_path = a + np.cumsum(increments)
        correction = (t_full - t_full[0]) / (t_full[-1] - t_full[0]) * (b - brownian_path[-1])
        full_bridge = brownian_path + correction

        # Découper un sous-trajet aléatoire
        start_idx = np.random.randint(0, nb_discretization_points - interval_length + 1)
        end_idx = start_idx + interval_length

        sample = full_bridge[start_idx:end_idx]
        t0, t1 = t_full[start_idx], t_full[end_idx - 1]

        samples.append(sample.reshape(-1, 1))
        time_intervals.append([t0, t1])

    # Conversion des échantillons et intervalles en tenseurs PyTorch
    samples_tensor = torch.tensor(np.stack(samples), dtype=torch.float32)
    time_intervals_tensor = torch.tensor(np.stack(time_intervals), dtype=torch.float32)

    return samples_tensor, time_intervals_tensor

def plot_brownian_bridge_samples(samples, time_intervals):
    """
    Affiche les sous-trajets extraits d'un pont brownien.

    :param samples: Tenseur de dimension (batch_size, interval_length, 1) contenant les sous-trajets
    :param time_intervals: Tenseur de dimension (batch_size, 2) contenant les intervalles temporels [t0, t1] pour chaque sous-trajet
    """
    if len(samples.shape) != 3 or samples.shape[2] != 1:
        raise ValueError("Les échantillons doivent avoir la forme (batch_size, interval_length, 1).")
    
    if len(time_intervals.shape) != 2 or time_intervals.shape[1] != 2:
        raise ValueError("Les intervalles temporels doivent avoir la forme (batch_size, 2).")
    
    samples_np = samples.squeeze(-1).numpy()
    time_intervals_np = time_intervals.numpy()
    
    plt.figure(figsize=(10, 6))
    
    for idx, (sample, time_interval) in enumerate(zip(samples_np, time_intervals_np)):
        t0, t1 = time_interval
        time_grid = np.linspace(t0, t1, len(sample))
        plt.plot(time_grid, sample)
    
    plt.title("Bouts de Ponts Browniens (Découpages fixes)")
    plt.xlabel("Temps")
    plt.ylabel("Position")

    plt.grid(True)
    plt.show()