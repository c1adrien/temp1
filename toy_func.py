import numpy as np
import torch
import matplotlib.pyplot as plt
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
import torch.optim as optim

#the model we used in the differents notebooks.
class StochasticProcessSampler:
    """
    A unified sampler for generating stochastic process trajectories including:
    - Brownian Motion
    - Brownian Bridge
    - Autoregressive Process (AR(p))
    """
    
    def __init__(self, process_type, batch_size, nb_discretization_points, interval_length, additional_params=None):
        self.process_type = process_type
        self.batch_size = batch_size
        self.nb_discretization_points = nb_discretization_points
        self.interval_length = interval_length
        self.additional_params = additional_params if additional_params else {}

    def generate_samples(self):
        """
        Generates samples based on the specified stochastic process.
        """
        if self.process_type == "brownian_motion":
            return self._generate_brownian_motion()
        elif self.process_type == "brownian_bridge":
            return self._generate_brownian_bridge()
        elif self.process_type == "ARp":
            return self._generate_ARp()
        else:
            raise ValueError(f"Unsupported process type: {self.process_type}")

    def _generate_brownian_motion(self):
        """
        Generates batch_size Brownian motion trajectories, extracts sub-trajectories.
        """
        t_full = np.linspace(0, 1, self.nb_discretization_points)
        dt = 1 / (self.nb_discretization_points - 1)
        samples = []
        start_indices = []
        full_trajectories = []

        for _ in range(self.batch_size):
            increments = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=self.nb_discretization_points - 1)
            W = np.concatenate([[0], np.cumsum(increments)])
            
            start_idx = np.random.randint(0, self.nb_discretization_points - self.interval_length + 1)
            end_idx = start_idx + self.interval_length
            
            sample = W[start_idx:end_idx]
            samples.append(sample.reshape(-1, 1))
            start_indices.append(t_full[start_idx])
            full_trajectories.append(W.reshape(-1, 1))

        return torch.tensor(np.stack(samples), dtype=torch.float32), torch.tensor(start_indices, dtype=torch.float32)

    def _generate_brownian_bridge(self):
        """
        Generates batch_size Brownian Bridge trajectories.
        """
        a = self.additional_params.get("a", 0)
        b = self.additional_params.get("b", 1)

        samples = []
        start_indices = []
        t_full = np.linspace(0, 1, self.nb_discretization_points)
        dt = 1 / (self.nb_discretization_points - 1)
        
        for _ in range(self.batch_size):
            increments = np.random.normal(0, np.sqrt(dt), size=self.nb_discretization_points - 1)
            W = np.concatenate([[0], np.cumsum(increments)])
            
            bridge = a + W - t_full * (W[-1] - b + a)
            bridge[0] = a
            bridge[-1] = b
            
            start_idx = np.random.randint(0, self.nb_discretization_points - self.interval_length + 1)
            end_idx = start_idx + self.interval_length
            
            sample = bridge[start_idx:end_idx]
            samples.append(sample.reshape(-1, 1))
            start_indices.append(t_full[start_idx])
        
        return torch.tensor(np.stack(samples), dtype=torch.float32), torch.tensor(start_indices, dtype=torch.float32)

    def _generate_ARp(self):
        """
        Generates batch_size AR(p) process samples, extracts sub-trajectories.
        """
        coefficients = self.additional_params.get("coefficients", [0.7, -0.3])
        sigma = self.additional_params.get("sigma", 0.05)
        init_values = self.additional_params.get("init_values", None)

        p = len(coefficients)
        samples = []
        start_indices = []
        t_full = np.linspace(0, 1, self.nb_discretization_points)

        for _ in range(self.batch_size):
            y = np.zeros(self.nb_discretization_points)
            if init_values is None:
                y[:p] = np.random.randn(p) * sigma
            else:
                if len(init_values) != p:
                    raise ValueError("Length of init_values must match AR(p) order.")
                y[:p] = init_values

            for t in range(p, self.nb_discretization_points):
                noise = np.random.randn() * sigma
                y[t] = sum(coefficients[i] * y[t - (i + 1)] for i in range(p)) + noise
            
            start_idx = np.random.randint(0, self.nb_discretization_points - self.interval_length + 1)
            end_idx = start_idx + self.interval_length
            sample = y[start_idx:end_idx]
            
            samples.append(sample.reshape(-1, 1))
            start_indices.append(t_full[start_idx])

        return torch.tensor(np.stack(samples), dtype=torch.float32), torch.tensor(start_indices, dtype=torch.float32)

# Visualization function
def plot_sampled_trajectories(samples, start_indices, title, nb_discretization_points=100):
    samples_np = samples.squeeze(-1).numpy()
    start_indices_np = start_indices.numpy()
    dt = 1 / (nb_discretization_points - 1)  # Espacement temporel entre les points
    
    plt.figure(figsize=(5, 2))
    for sample, start_time in zip(samples_np, start_indices_np):
        end_time = start_time + len(sample) * dt  # Calcul de la fin de la sous-trajet
        sample_time_grid = np.linspace(start_time, end_time, len(sample))  # Temps correspondant au sous-trajet
        plt.plot(sample_time_grid, sample)
    
    plt.title(title)
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.grid(True)
    plt.show()





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
    




def train_tMCL(
    model, 
    process_type,
    num_epochs, 
    batch_size, 
    nb_discretization_points, 
    interval_length,
    device, 
    learning_rate=0.001,
    additional_params=None
):
    """
    Entraîne le réseau de neurones tMCL avec des trajectoires spécifiques (Brownian Motion, Brownian Bridge, AR(p)).

    :param model: Instance du modèle tMCL
    :param process_type: Type de processus à entraîner ('brownian_motion', 'brownian_bridge', 'ARp')
    :param num_epochs: Nombre d'époques d'entraînement
    :param batch_size: Taille du batch pour la génération de trajectoires
    :param nb_discretization_points: Nombre total de points de discrétisation
    :param interval_length: Longueur des trajectoires simulées
    :param device: Dispositif d'exécution (CPU ou GPU)
    :param learning_rate: Taux d'apprentissage
    :param additional_params: Dictionnaire d'hyperparamètres propres au processus (coefficients, sigma, init_values, etc.)
    :return: Modèle entraîné
    """
    
    if additional_params is None:
        additional_params = {}

    # Définir l'optimiseur
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Basculer le modèle en mode entraînement
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        if process_type == "brownian_motion":
            trajectories, start_indices = StochasticProcessSampler(
                process_type, batch_size, nb_discretization_points, interval_length
            ).generate_samples()

            trajectories = trajectories.to(device)
            start_indices = start_indices.to(device)

            distr_args = trajectories[:, 0, :]  # Condition : x0
            target_list = trajectories[:, :, :]

        elif process_type == "brownian_bridge":
            trajectories, start_indices = StochasticProcessSampler(
                process_type, batch_size, nb_discretization_points, interval_length
            ).generate_samples()

            trajectories = trajectories.to(device)
            start_indices = start_indices.to(device)

            t0 = start_indices.unsqueeze(-1)  # [batch_size, 1]
            distr_args = torch.cat([trajectories[:, 0, :], t0], dim=-1)  # Condition : [x0, t0]
            target_list = trajectories[:, :, :]

        elif process_type == "ARp":
            p = additional_params.get("p", 2)
            coefficients = additional_params.get("coefficients", [0.8, 0.02])
            sigma = additional_params.get("sigma", 0.002)
            init_values = additional_params.get("init_values", None)

            trajectories, start_indices = StochasticProcessSampler(
                process_type, batch_size, nb_discretization_points, interval_length,
                additional_params={"coefficients": coefficients, "sigma": sigma, "init_values": init_values}
            ).generate_samples()

            trajectories = trajectories.to(device)
            start_indices = start_indices.to(device)

            distr_args = trajectories[:, :p, :].reshape(batch_size, -1)  # Condition : p derniers points
            target_list = trajectories[:, p:, :]

        else:
            raise ValueError(f"Process type {process_type} not supported.")

        # Réinitialiser les gradients
        optimizer.zero_grad()

        # Calculer les prédictions et la perte
        loss = model.log_prob(target_list, distr_args)

        # Rétropropagation
        loss.backward()

        # Mettre à jour les poids
        optimizer.step()

        # Afficher les progrès
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model



def plot_brownian_bridge(interval_length, nb_discretization_points, m, N_levels, a, b, t_condition, trained_model):
    def kl_eigenfunctions(t, T, m):
        eigenfunctions = []
        eigenvalues = []
        for n in range(1, m + 1):
            eigenfunc = np.sqrt(2 / T) * np.sin(np.pi * n * t / T)
            eigenval = (T / (n * np.pi))**2
            eigenfunctions.append(eigenfunc)
            eigenvalues.append(eigenval)
        return np.array(eigenfunctions), np.array(eigenvalues)

    def generate_quantization_points(N, dist_mean=0, dist_var=1):
        std_dev = np.sqrt(dist_var)
        return norm.ppf(np.linspace(1 / (2 * N), 1 - 1 / (2 * N), N)) * std_dev

    def reconstruct_quantized_trajectories_conditioned(N_levels, eigenfunctions, eigenvalues, t, x, b):
        m = len(N_levels)
        quantization_grids = [generate_quantization_points(N, dist_var=1) for N in N_levels]
        multi_indices = list(product(*[range(len(grid)) for grid in quantization_grids]))
        
        linear_interpolation = (1 - (t - t[0]) / (t[-1] - t[0])) * x + ((t - t[0]) / (t[-1] - t[0])) * b
        
        trajectories = []
        for index in multi_indices:
            trajectory = sum(
                np.sqrt(eigenvalues[n]) * quantization_grids[n][index[n]] * eigenfunctions[n]
                for n in range(m)
            )
            trajectories.append(trajectory + linear_interpolation)
        
        return np.array(trajectories)

    T = t_condition + interval_length / nb_discretization_points
    num_steps = nb_discretization_points
    t_full = np.linspace(0, T, num_steps)
    
    dt = T / len(t_full)
    W = np.cumsum(np.sqrt(dt) * np.random.randn(len(t_full)))
    W = np.insert(W, 0, 0)[:-1]
    bridge_0_T = W - (t_full / T) * W[-1] + (1 - t_full / T) * a + (t_full / T) * b
    
    t_observed = t_full[t_full <= t_condition]
    bridge_0_t = bridge_0_T[:len(t_observed)]
    
    B_t = bridge_0_t[-1]
    
    t_quantized = t_full[t_full >= t_condition]
    eigenfunctions, eigenvalues = kl_eigenfunctions(t_quantized - t_condition, T - t_condition, m)
    
    trajectories_quantized = reconstruct_quantized_trajectories_conditioned(
        N_levels, eigenfunctions, eigenvalues, t_quantized, B_t, b
    )
    
    predictions_neural = trained_model.forward(torch.tensor([[float(B_t), t_condition]]))[0].detach().numpy()
    
    plt.figure(figsize=(14, 7))
    plt.plot(t_observed, bridge_0_t, label="Observed trajectory on [0, t_condition]", color='blue')
    
    for trajectory in trajectories_quantized:
        plt.plot(t_quantized[:len(predictions_neural[0])], trajectory[:len(predictions_neural[0])], alpha=0.6, color='orange')
    
    for trajectory in predictions_neural:
        plt.plot(t_quantized[:len(predictions_neural[0])], trajectory, alpha=0.6, color='green')
    
    plt.title(f"Quantization of Conditioned Brownian Bridge on [t, T] with $B(t)={B_t:.2f}$, $B(T)={b}$")
    plt.xlabel("Time $t$")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
    
    return {
        "t_observed": t_observed,
        "bridge_0_t": bridge_0_t,
        "t_quantized": t_quantized,
        "trajectories_quantized": trajectories_quantized,
        "predictions_neural": predictions_neural
    }






def plot_brownien(T, t_condition, pred_length, num_steps, m, N_levels, trained_model):
    prediction_length = pred_length/num_steps
    def kl_eigenfunctions(t, T, m):
        eigenfunctions = []
        eigenvalues = []
        for n in range(1, m + 1):
            eigenfunc = np.sqrt(2 / T) * np.sin(np.pi * (n - 0.5) * t / T)
            eigenval = T**2 / (np.pi**2 * (n - 0.5)**2)
            eigenfunctions.append(eigenfunc)
            eigenvalues.append(eigenval)
        return np.array(eigenfunctions), np.array(eigenvalues)

    def generate_quantization_points(N, dist_mean=0, dist_var=1):
        std_dev = np.sqrt(dist_var)
        return norm.ppf(np.linspace(1 / (2 * N), 1 - 1 / (2 * N), N)) * std_dev

    def reconstruct_quantized_trajectories(N_levels, eigenfunctions, eigenvalues):
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

    t_full = np.linspace(0, T, num_steps)
    t_prediction = np.linspace(0, prediction_length, int(prediction_length * num_steps / T))
    t_adjusted = np.linspace(t_condition, T, len(t_prediction))

    eigenfunctions, eigenvalues = kl_eigenfunctions(t_prediction, prediction_length, m)
    t_observed = np.linspace(0, t_condition, int(t_condition * num_steps / T))
    trajectory_0_t = np.cumsum(np.sqrt(1 / len(t_observed)) * np.random.randn(len(t_observed)))

    B_t = trajectory_0_t[-1]

    trajectories_quantized = reconstruct_quantized_trajectories(N_levels, eigenfunctions, eigenvalues) 
    predictions = trained_model.forward(torch.tensor([[float(B_t)]])) 
    predictions_neural = predictions[0].detach().numpy()

    trajectories_conditioned = trajectories_quantized + B_t

    plt.figure(figsize=(14, 7))
    plt.plot(t_observed, trajectory_0_t, label="Observed trajectory on [0, t_condition]", color='blue')
    for trajectory in trajectories_conditioned:
        plt.plot(t_adjusted, trajectory, alpha=0.6, color='orange')
    for trajectory in predictions_neural:
        plt.plot(t_adjusted, trajectory, alpha=0.6, color='green')
    
    plt.title("Quantization of Brownian Motion on [t_condition, T] Conditioned on [0, t_condition]")
    plt.xlabel("Time $t$")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
    
    return {
        "t_observed": t_observed,
        "trajectory_0_t": trajectory_0_t,
        "t_adjusted": t_adjusted,
        "trajectories_conditioned": trajectories_conditioned,
        "predictions_neural": predictions_neural
    }





def plot_ARp_quantization(batch_size, nb_discretization_points, interval_length, coefficients, sigma, t_condition, trained_model, nb_clusters=10):
    def generate_ARp_samples(batch_size, nb_discretization_points, interval_length, coefficients, sigma=0.05, init_values=None):
        """
        Génère des trajectoires complètes d'un processus AR(p) et extrait des sous-trajets.

        :param batch_size: Nombre de sous-trajets à générer
        :param nb_discretization_points: Nombre total de points de discrétisation
        :param interval_length: Nombre de points dans chaque sous-trajet (bouts)
        :param coefficients: Liste des coefficients [alpha1, alpha2, ..., alpha_p] pour le processus AR(p)
        :param sigma: Écart-type du bruit blanc
        :param seed: Graine pour la reproductibilité
        :param init_values: Valeurs initiales optionnelles (liste de longueur p) ou None pour aléatoire
        :param fixed_start: Si True, les sous-trajectoires commencent toutes à un même instant initial
        :return:
            - Tenseur de dimension (batch_size, interval_length, 1) contenant les sous-trajets
        """
        p = len(coefficients)
        if interval_length > nb_discretization_points:
            raise ValueError("`interval_length` doit être inférieur ou égal à `nb_discretization_points`.")
        
    
        
        samples = []
        
        for _ in range(batch_size):
            # Générer une trajectoire complète du processus AR(p)
            y = np.zeros(nb_discretization_points)
            
            # Initialisation des valeurs
            if init_values is None:
                y[:p] = np.random.randn(p) * sigma
            else:
                if len(init_values) != p:
                    raise ValueError("La longueur de init_values doit être égale à p.")
                y[:p] = init_values
            
            for t in range(p, nb_discretization_points):
                noise = np.random.randn() * sigma
                y[t] = sum(coefficients[i] * y[t - (i + 1)] for i in range(p)) + noise
            
            # Sélectionner les sous-trajets
            if init_values:
                start_idx = 0  # Tout commence au même instant
            else:
                start_idx = np.random.randint(0, nb_discretization_points - interval_length + 1)
            
            end_idx = start_idx + interval_length
            sample = y[start_idx:end_idx]
            
            samples.append(sample.reshape(-1, 1))
        
        # Conversion en tenseur PyTorch
        samples_tensor = torch.tensor(np.stack(samples), dtype=torch.float32)
        
        return samples_tensor
    
    nb_steps = nb_discretization_points
    y = np.zeros(nb_steps)
    p = len(coefficients)
    y[:p] = np.random.randn(p) * sigma
    
    for t in range(p, nb_steps):
        noise = np.random.randn() * sigma
        y[t] = sum(coefficients[i] * y[t - (i + 1)] for i in range(p)) + noise
    
    y_observed = y[:t_condition]
    y_future = y[t_condition:]
    
    init_values = y_observed[-p:]
    samples = generate_ARp_samples(1000, nb_discretization_points, interval_length + p, coefficients, sigma, list(init_values))
    samples_np = samples.squeeze(-1)
    
    kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(samples_np)
    kmeans_centroids = kmeans.cluster_centers_
    
    init_values = list(init_values)
    init_values = [float(x) for x in init_values]
    x_condition = torch.tensor([init_values])
    predictions_neural = trained_model.forward(x_condition)[0].detach().numpy()
    
    time_full = np.linspace(0, 1, nb_steps)
    time_observed = time_full[:t_condition]
    time_future = time_full[t_condition:t_condition + predictions_neural.shape[1]]
    
    plt.figure(figsize=(14, 7))
    plt.plot(time_observed, y_observed, label="Trajectoire observée", color='blue', linewidth=2)
    
    for centroid in kmeans_centroids:
        plt.plot(time_future, centroid[p:len(time_future) + p], alpha=0.6, color='orange', label="K-means Centroid")
    
    for trajectory in predictions_neural:
        plt.plot(time_future, trajectory, alpha=0.6, color='green', label="WTA Prediction")
    
    plt.title(f"Quantification et WTA du processus AR(p) à partir de t={t_condition}")
    plt.xlabel("Temps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
    
    return {
        "time_observed": time_observed,
        "y_observed": y_observed,
        "time_future": time_future,
        "kmeans_centroids": kmeans_centroids,
        "predictions_neural": predictions_neural
    }
# nb_discretization_points =300
# interval_length = 150
# batch_size = 500
# sampler = StochasticProcessSampler("brownian_motion", batch_size=batch_size,
#                                     nb_discretization_points=nb_discretization_points, 
#                                     interval_length=interval_length)
# samples, start_indices = sampler.generate_samples()
# #plot_sampled_trajectories(samples, start_indices, "Sous-trajectoires de Mouvement Brownien", nb_discretization_points)

# sampler = StochasticProcessSampler("brownian_bridge", batch_size=batch_size,
#                                     nb_discretization_points=nb_discretization_points, 
#                                     interval_length=interval_length)

# samples, start_indices = sampler.generate_samples()
# #plot_sampled_trajectories(samples, start_indices, "Sous-trajectoires de Pont Brownien", nb_discretization_points)

# sampler = StochasticProcessSampler("ARp", batch_size=batch_size, 
#                                    nb_discretization_points=nb_discretization_points, 
#                                    interval_length=interval_length, 
#                                    additional_params={"coefficients": [0.5, 0.3], "sigma": 0.05})
# samples, start_indices = sampler.generate_samples()
# #plot_sampled_trajectories(samples, start_indices, "Sous-trajectoires AR(p)", nb_discretization_points)