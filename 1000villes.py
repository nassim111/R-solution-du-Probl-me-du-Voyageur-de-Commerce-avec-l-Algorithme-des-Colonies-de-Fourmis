import numpy as np
import pandas as pd

# Read coordinates from CSV file
df = pd.read_csv('large.csv', header=None)
cities_coordinates = df.values

# Calculate distance matrix
d = np.linalg.norm(cities_coordinates[:, np.newaxis, :] - cities_coordinates, axis=-1)

iteration = 3000
n_ants = 1000
n_citys = len(cities_coordinates)

# Rest of your code remains unchanged


    # Initialisation des paramètres
m = n_ants
n = n_citys
e = 0.5         # Taux d'évaporation des phéromones
alpha = 1       # Facteur de phéromone
beta = 2        # Facteur de visibilité

    # Calcul de la visibilité (distance inverse)
visibility = 1 / d
visibility[np.isinf(visibility)] = 0
#visibility[visibility == inf] = 0

    # Initialisation des phéromones
pheromone = 0.1 * np.ones((m, n))

    # Initialisation des routes des fourmis
rute = np.ones((m, n + 1))

for ite in range(iteration):

        rute[:, 0] = 1  # Position initiale de chaque fourmi, la ville '1'

        for i in range(m):

            temp_visibility = np.array(visibility)

            for j in range(n - 1):

                combine_feature = np.zeros(n)
                cum_prob = np.zeros(n)

                cur_loc = int(rute[i, j] - 1)

                temp_visibility[:, cur_loc] = 0

                p_feature = np.power(pheromone[cur_loc, :], beta)
                v_feature = np.power(temp_visibility[cur_loc, :], alpha)

                p_feature = p_feature[:, np.newaxis]
                v_feature = v_feature[:, np.newaxis]

                combine_feature = np.multiply(p_feature, v_feature)

                total = np.sum(combine_feature)
                probs = combine_feature / total
                cum_prob = np.cumsum(probs)

                r = np.random.random_sample()
                city = np.nonzero(cum_prob > r)[0][0] + 1

                rute[i, j + 1] = city

            left = list(set([i for i in range(1, n + 1)]) - set(rute[i, :-2]))[0]
            rute[i, -2] = left

        rute_opt = np.array(rute)

        dist_cost = np.zeros((m, 1))

        for i in range(m):
            s = 0
            for j in range(n - 1):
                s = s + d[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1]
            dist_cost[i] = s

        dist_min_loc = np.argmin(dist_cost)
        dist_min_cost = dist_cost[dist_min_loc]

        best_route = rute[dist_min_loc, :]
        pheromone = (1 - e) * pheromone

        for i in range(m):
            for j in range(n - 1):
                dt = 1 / dist_cost[i]
                pheromone[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1] += dt


# Print results
print('Cas de', len(cities_coordinates), 'villes :')
print()
print('Meilleur chemin:', best_route)
print('Longueur du meilleur chemin:', int(dist_min_cost[0]) + d[int(best_route[-2]) - 1, 0])
