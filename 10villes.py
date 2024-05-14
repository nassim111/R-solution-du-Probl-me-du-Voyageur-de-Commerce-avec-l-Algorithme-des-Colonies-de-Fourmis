import numpy as np
from numpy import inf
cities_coordinates = np.array([
                                [7.751835449506971898e-02,1.701756081581240476e-02],
                                [8.920754898569652758e-01,7.373514682305275514e-01],
                                [3.454501876496263169e-02,4.184648528935739353e-01],
                                [4.914751692674588224e-01,6.439409755506043664e-01],
                                [9.045597035614028059e-04,6.336019048365898465e-01],
                                [8.891986053886618002e-01,7.461217257399166414e-02],
                                [3.393974714022980343e-01,7.646842668898091722e-01],
                                [6.308181956673577506e-01,2.336425069907764884e-01],
                                [3.307065478033500705e-01,8.515144334870929921e-01],
                                [8.889897385795041407e-01,7.094063238673585792e-01]
                                
                             ])
d = np.zeros((len(cities_coordinates), len(cities_coordinates)))

for i in range(len(cities_coordinates)):
        for j in range(len(cities_coordinates)):
            # Calcul de la distance euclidienne entre les villes
            d[i, j] = np.linalg.norm(cities_coordinates[i] - cities_coordinates[j])

iteration = 2000
n_ants = 20
n_citys = len(cities_coordinates)

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

print('Cas de', len(cities_coordinates), 'villes :')
    #print('Chemin de toutes les fourmis vers la fin:')
    #print(rute_opt)
print()
print('Meilleur chemin:', best_route)
print('Longueur du meilleur chemin:', int(dist_min_cost[0]) + d[int(best_route[-2]) - 1, 0])
