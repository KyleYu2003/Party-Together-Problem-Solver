import networkx as nx
from student_utils import *


def mtsp_dp(G):
    """
    TSP solver using dynamic programming.
    Input:
        G: a NetworkX graph representing the city.
        This directed graph is equivalent to an undirected one by construction.
    Output:
        tour: a list of nodes traversed by your car.

    All nodes are represented as integers.

    You must solve the problem using dynamic programming.

    The tour must begin and end at node 0.
    It can only go through edges that exist in the graph.
    It must visit every node in G exactly once.
    """
    n = G.number_of_nodes()
    all_nodes = list(G.nodes)
    start_node = 0
    INF = float('inf')

    # Draw
    # draw_gragh(G)

    # Initialize DP table
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1 << start_node][start_node] = 0

    # Fill DP table
    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u):
                for v in range(n):
                    if (not (mask & 1)):
                        continue
                    if not mask & (1 << v):
                        new_mask = mask | (1 << v)
                        if G.has_edge(all_nodes[u], all_nodes[v]):
                            dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + G[all_nodes[u]][all_nodes[v]]['weight'])

    # Reconstruct the tour
    end_state = (1 << n) - 1
    last_node = start_node
    min_tour_cost = INF
    for i in range(n):
        if G.has_edge(all_nodes[i], start_node) and dp[end_state][i] + G[all_nodes[i]][start_node]['weight'] < min_tour_cost:
            min_tour_cost = dp[end_state][i] + G[all_nodes[i]][start_node]['weight']
            last_node = i

    tour = []
    mask = end_state
    while mask:
        tour.append(all_nodes[last_node])
        prev_mask = mask ^ (1 << last_node)
        for i in range(n):
            if prev_mask & (1 << i) and G.has_edge(all_nodes[i], all_nodes[last_node]) and dp[mask][last_node] == dp[prev_mask][i] + G[all_nodes[i]][all_nodes[last_node]]['weight']:
                last_node = i
                break
        mask = prev_mask

    tour.append(start_node)
    tour.reverse()
    return tour
