import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

def make_modular_network(N, average_degree, community_number, mu):
    assert N % community_number == 0, 'N must be devisible by community_number'
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if j < (N/community_number)*(i//(N/community_number)+1):
                if np.random.rand() < ((N-(i/N))/N)*average_degree*(1-mu)/(N/community_number):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
            else:
                if np.random.rand() < ((N-(i/N))/N)*average_degree*(mu)/(N-(N/community_number)):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
    return G

if __name__ == '__main__':
    G_array = make_modular_network(128, 16, 4, 0.1)
    G=nx.from_numpy_matrix(G_array)
    plt.figure(figsize=(5,5))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G,pos)

    plt.axis("off")
    plt.savefig("example.png")
    plt.show()
