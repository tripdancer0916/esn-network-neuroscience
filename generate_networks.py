import numpy as np

def make_modular_network(N, average_degree, community_number, mu):
    assert N % community_number == 0, 'N must be devisible by community_number'
    G = np.zeros((N, N))
    size = N/community_number
    for i in range(N):
        com_index = i//size
        k_in_prev = 0
        k_out_prev = 0
        for j in range(int(size*com_index)):
            if G[i][j] != 0:
                k_out_prev += 1
        for j in range(int(size*com_index), int(size*(com_index+1))):
            if G[i][j] != 0:
                k_in_prev += 1
        for j in range(i, N):
            if j < size*((i//size)+1):
                if np.random.rand() < (average_degree*(1-mu)-k_in_prev)/(size-(i-(size*com_index))+1):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
            else:
                if np.random.rand() < (average_degree*(mu)-k_out_prev)/(N-(size*((i//size)+1))+1):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
    return G


def make_recurrent_layered_network(N, average_degree, community_number, mu):
    assert N % community_number == 0, 'N must be devisible by community_number'
    size = N / community_number
    G = np.zeros((N, N))
    for i in range(N):
        com_index = i // size
        k_in_prev = 0
        k_out_prev = 0
        for j in range(int(size * com_index)):
            if G[i][j] != 0:
                k_out_prev += 1
        for j in range(int(size * com_index), int(size * (com_index + 1))):
            if G[i][j] != 0:
                k_in_prev += 1
        for j in range(i, N):
            if j < size * ((i // size) + 1):
                if int(com_index) == 0 or int(com_index) == community_number - 1:
                    if np.random.rand() < (average_degree * (1 - (mu * 0.5)) - k_in_prev) / (
                            size - (i - (size * com_index)) + 1):
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()
                else:
                    if np.random.rand() < (average_degree * (1 - mu) - k_in_prev) / (
                            size - (i - (size * com_index)) + 1):
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()
            elif j < size * ((i // size) + 2):
                if int(com_index) == 0 or int(com_index) == community_number - 1:
                    if np.random.rand() < (average_degree * (mu * 0.5) - k_out_prev) / size:
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()
                else:
                    if np.random.rand() < (average_degree * (mu) - k_out_prev) / size:
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()

    return G


def make_feedforward_layered_network(N, average_degree, community_number, mu):
    assert N % community_number == 0, 'N must be devisible by community_number'
    size = N/community_number
    G = np.zeros((N, N))
    for i in range(N):
        com_index = i//size
        k_in_prev = 0
        k_out_prev = 0
        for j in range(int(size*com_index)):
            if G[j][i] != 0:
                k_out_prev += 1
        for j in range(int(size*com_index), int(size*(com_index+1))):
            if G[i][j] != 0:
                k_in_prev += 1
        for j in range(i, N):
            if j < size*((i//size)+1):
                if int(com_index) == 0 or int(com_index) == community_number-1:
                    if np.random.rand() < (average_degree*(1-(mu*0.5))-k_in_prev)/(size-(i-(size*com_index))+1):
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()
                else:
                    if np.random.rand() < (average_degree*(1-mu)-k_in_prev)/(size-(i-(size*com_index))+1):
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()
            elif j < size*((i//size)+2):
                if int(com_index) == 0 or int(com_index) == community_number-1:
                    if np.random.rand() < (average_degree*(mu*0.5)-k_out_prev)/size:
                        G[i][j] = np.random.randn()
                        # G[j][i] = np.random.randn()
                else:
                    if np.random.rand() < (average_degree*(mu)-k_out_prev)/size:
                        G[i][j] = np.random.randn()
                        # G[j][i] = np.random.randn()

    return G


def make_bypass_network(N, average_degree, community_number, mu, eta):
    assert N % community_number == 0, 'N must be devisible by community_number'
    size = N/community_number
    G = np.zeros((N, N))
    for i in range(N):
        com_index = i//size
        k_in_prev = 0
        k_out_prev = 0
        for j in range(int(size*com_index)):
            if G[j][i] != 0:
                k_out_prev += 1
        for j in range(int(size*com_index), int(size*(com_index+1))):
            if G[i][j] != 0:
                k_in_prev += 1
        for j in range(i, N):
            if j < size*((i//size)+1):
                if int(com_index) == 0 or int(com_index) == community_number-1:
                    if np.random.rand() < (average_degree*(1-(mu*0.5))-k_in_prev)/(size-(i-(size*com_index))+1):
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()
                else:
                    if np.random.rand() < (average_degree*(1-mu)-k_in_prev)/(size-(i-(size*com_index))+1):
                        G[i][j] = np.random.randn()
                        G[j][i] = np.random.randn()
            elif j < size*((i//size)+2):
                if int(com_index) == 0 or int(com_index) == community_number-1:
                    if np.random.rand() < (average_degree*((mu-eta)*0.5)-k_out_prev)/size:
                        G[i][j] = np.random.randn()
                        # G[j][i] = np.random.randn()
                else:
                    if np.random.rand() < (average_degree*(mu-eta)-k_out_prev)/size:
                        G[i][j] = np.random.randn()
                        # G[j][i] = np.random.randn()
            else:
                if int(com_index) == 0 or int(com_index) == community_number-1:
                    if np.random.rand() < (average_degree*(eta*0.5)-k_out_prev)/size:
                        G[i][j] = np.random.randn()
                        # G[j][i] = np.random.randn()
                else:
                    if np.random.rand() < (average_degree*(eta)-k_out_prev)/size:
                        G[i][j] = np.random.randn()
                        # G[j][i] = np.random.randn()

    return G
