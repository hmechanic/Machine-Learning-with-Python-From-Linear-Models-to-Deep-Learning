import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt

X = np.loadtxt("toy_data.txt")

# TODO: Your code here


# Problem 2
def run_kmeans():
    for k in range(1, 5):
        min_cost = None
        best_seed = None
        
        for seed in range(0, 5):
            mixture, post = common.init(X, k, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            
            if min_cost is None or cost > min_cost:
                min_cost = cost
                best_seed = seed
            
        mixture, post = common.init(X, k, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = f"K-means for K={k}, seed={best_seed}, cost={min_cost}"
        print(title)
        common.plot(X, mixture, post, title)
        (X, mixture, post, title)
        

# run_kmeans()


BIC = []

def run_EM():
    for k in [1, 12]:#range(1, 7):
        min_cost = None
        best_seed = None
        
        for seed in range(0, 5):
            mixture, post = common.init(X, k, seed)
            mixture, post, cost = em.run(X, mixture, post)
            print(f"K-means for K={k}, seed={seed}, cost={cost}")
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_seed = seed
        
        mixture, post = common.init(X, k, best_seed)
        mixture, post, cost = em.run(X, mixture, post)
        BIC.append(common.bic(X, mixture, cost))
        title = f"em for K={k}, seed={best_seed}, cost={min_cost}"
        print(title)
        common.plot(X, mixture, post, title)
        (X, mixture, post, title)
        
# run_EM()

# mixture, post = common.init(X, 4, 4)
# mixture, post, cost = em.run(X, mixture, post)
# title = f"em for K={4}, seed={4}, cost={cost}"
# print(title)
# common.plot(X, mixture, post, title)

# # Plotting the best number of k clusters 
# K_values = list(range(1, len(BIC) + 1))

# plt.figure(figsize=(6,4))
# plt.plot(K_values, BIC, marker="o", linestyle="-")
# plt.xlabel("Number of clusters (K)")
# plt.ylabel("BIC score")
# plt.title("BIC vs Number of Clusters")
# plt.grid(True)

# # mark the best K (max BIC)
# best_k = K_values[BIC.index(max(BIC))]
# best_bic = max(BIC)
# plt.scatter(best_k, best_bic, color="red", zorder=5, label=f"Best K = {best_k}")
# plt.legend()

# plt.show()


### Problen section 7


X = np.loadtxt("netflix_incomplete.txt")

# run_EM()


# Predictions

mixture, post = common.init(X, 12, 1)
mixture, post, cost = em.run(X, mixture, post)

X_pred = em.fill_matrix(X, mixture)

X_gold = np.loadtxt("netflix_complete.txt")

common.rmse(X_gold, X_pred)