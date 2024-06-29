import time
import random
from heapq import heappush, heappop
import networkx as nx
from student_utils import *
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def ptp_solver(G: nx.DiGraph, H: list, alpha: float):
    """
    PTP solver.
    Input:
        G: a NetworkX graph representing the city.
        This directed graph is equivalent to an undirected one by construction.
        H: a list of home nodes that you must visit.
        alpha: the coefficient of calculating cost.
    Output:
        tour: a list of nodes traversed by your car.
        pick_up_locs_dict: a dictionary of (pick-up-locations, friends-picked-up) pairs
        where friends-picked-up is a list/tuple containing friends who get picked up at
        that specific pick-up location. Friends are represented by their home nodes.

    All nodes are represented as integers.

    The tour must begin and end at node 0.
    It can only go through edges that exist in the graph.
    Pick-up locations must be in the tour.
    Everyone should get picked up exactly once.
    """

    # Initialize the tour with just the starting point 0
    T = [0]

    # Compute shortest paths between all nodes
    all_pairs_shortest_path_length = dict(nx.floyd_warshall(G))

    def compute_cost(T, pick_up_locs_dict):
        driving_cost = 0
        walking_cost = 0

        # Calculate driving cost
        for i in range(1, len(T)):
            driving_cost += alpha * all_pairs_shortest_path_length[T[i-1]][T[i]]

        # Calculate walking cost
        for pick_up_loc, friends in pick_up_locs_dict.items():
            for friend in friends:
                walking_cost += all_pairs_shortest_path_length[friend][pick_up_loc]

        return driving_cost + walking_cost

    def tsp(route):
        # Create the routing index manager and model.
        manager = pywrapcp.RoutingIndexManager(len(route), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return all_pairs_shortest_path_length[route[from_node]][route[to_node]]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution: 
            # Get the solution route.
            index = routing.Start(0)
            tour = []
            while not routing.IsEnd(index):
                tour.append(route[manager.IndexToNode(index)])
                index = solution.Value(routing.NextVar(index))
            tour.append(route[manager.IndexToNode(index)])
            
            return tour
        
        # Should not return None
        return None

    # Initialize pick up locations dict
    pick_up_locs_dict = {}

    # Assign initial pickup locations
    for friend in H:
        pick_up_locs_dict[friend] = [friend]

    # Create initial tour with pick-up locations and node 0
    T = [0] + list(pick_up_locs_dict.keys()) + [0]

    # Improve the tour and pick-up locations iteratively
    while True:
        best_cost = compute_cost(T, pick_up_locs_dict)
        improved = False

        # Create a priority queue for friends and their candidate pickup locations
        friend_queue = []
        for friend in H:
            current_pickup = [k for k, v in pick_up_locs_dict.items() if friend in v][0]
            for candidate_pickup in G.nodes:
                if candidate_pickup == current_pickup:
                    continue
                cost = compute_cost([0, candidate_pickup, 0], {candidate_pickup: [friend]})
                heappush(friend_queue, (cost, friend, candidate_pickup))

        # Attempt to improve by reassigning friends to different pickup locations
        while friend_queue:
            _, friend, candidate_pickup = heappop(friend_queue)
            # the generator has and only has one pickup 
            current_pickup = [k for k, v in pick_up_locs_dict.items() if friend in v][0]
            new_pick_up_locs_dict = {k: v[:] for k, v in pick_up_locs_dict.items()}
            new_pick_up_locs_dict[current_pickup].remove(friend)
            if not new_pick_up_locs_dict[current_pickup]:
                del new_pick_up_locs_dict[current_pickup]
            if candidate_pickup in new_pick_up_locs_dict:
                new_pick_up_locs_dict[candidate_pickup].append(friend)
            else:
                new_pick_up_locs_dict[candidate_pickup] = [friend]
            new_T = [0] + list(new_pick_up_locs_dict.keys()) + [0]
            
            # As the walk cost from home to pick up point is fixed, we just need to solve tsp on the pick up points
            new_T = tsp(new_T)
            new_cost = compute_cost(new_T, new_pick_up_locs_dict)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_T = new_T
                best_pick_up_locs_dict = new_pick_up_locs_dict
                improved = True
                break

        if not improved:
            break

        T = best_T
        pick_up_locs_dict = best_pick_up_locs_dict

    # Ensure the tour only contains existing edges
    final_tour = [T[0]]
    for i in range(1, len(T)):
        if G.has_edge(T[i-1], T[i]):
            final_tour.append(T[i])
        else:
            shortest_path = nx.shortest_path(G, source=T[i-1], target=T[i], weight="weight")
            final_tour.extend(shortest_path[1:])

    # Ensure the tour ends at 0
    if final_tour[-1] != 0:
        final_tour.append(0)

    return final_tour, pick_up_locs_dict

if __name__ == "__main__":
    file = 'inputs/8.in'
    print(f"\nReading file {os.path.basename(file)}...")
    G, H, alpha = input_file_to_instance(file)
    print(f"n = {G.number_of_nodes()}, |H| = {len(H)}, alpha = {alpha}")
    print('Graph constructed...')

    start_time = time.time()
    ptp_tour, ptp_pick_up_lcs_dict = ptp_solver(G, H, alpha)
    ptp_time = time.time() - start_time
    print('Tour generated by PTP solver...')
    print("Writing solution to output file...")
    write_ptp_solution_to_out(ptp_tour, ptp_pick_up_lcs_dict, os.path.basename(file))
    print("Output written.")
    print('Analyzing the solution...')

    ptp_is_legitimate, ptp_driving_cost, ptp_walking_cost = analyze_solution(G, H, alpha, ptp_tour, ptp_pick_up_lcs_dict)
    ptp_cost = ptp_driving_cost + ptp_walking_cost

    print(f"Your PTP solution is{' NOT' if not ptp_is_legitimate else ''} legitimate.")
    print(f"Total cost of your PTP solution: {ptp_cost:.6f}")
    print(f"Total driving cost of your PTP solution: {ptp_driving_cost:.6f}")
    print(f"Total walking cost of your PTP solution: {ptp_walking_cost:.6f}")
    print(f"Running time of your PTP solver: {ptp_time:.6f} seconds")

    print(f"Tour: {ptp_tour}")
    print(f"Pick-up locations dictionary: {ptp_pick_up_lcs_dict}")
