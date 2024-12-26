# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    route_to_stops.clear()
    trip_to_route.clear()
    stop_trip_count.clear()
    fare_rules.clear()
    merged_fare_df = None

    # Create trip_id to route_id mapping
    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    # Map route_id to a list of stops in order of their sequence
    for _, row in df_stop_times.iterrows():
        tripID = row['trip_id']
        routeID = trip_to_route.get(tripID)
        if routeID:
            route_to_stops[routeID].append(row['stop_id'])

    # Ensure each route only has unique stops
    for routeID, stops in route_to_stops.items():
        route_to_stops[routeID] = list(dict.fromkeys(stops))
    
    # Count trips per stop
    for _, row in df_stop_times.iterrows():
        stop_trip_count[row['stop_id']]+=1

    # Create fare rules for routes
    fare_rules = df_fare_rules.groupby('route_id').apply(lambda x: x.to_dict('records')).to_dict()

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_attributes, df_fare_rules, on='fare_id')
    

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    # pass  # Implementation here
    tripCount = defaultdict(int)
    for trip_id, rid in trip_to_route.items():
        tripCount[rid]+=1

    busiest_routes = sorted(tripCount.items(), key= lambda x: x[1], reverse=True)[:5]

    return [(int(route_id), trip_count) for route_id, trip_count in busiest_routes]

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    # pass  # Implementation here
    frequentStops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return [(int(stop_id), trip_count) for stop_id, trip_count in frequentStops]


# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    # pass  # Implementation here
    stopRouteCount = defaultdict(set)

    for rID, stops in route_to_stops.items():
        for stopID in stops:
            stopRouteCount[stopID].add(rID)
    
    busiestStops = {sID: len(routes) for sID, routes in stopRouteCount.items()}

    top5 = sorted(busiestStops.items(), key=lambda x: x[1], reverse=True)[:5]

    return [(int(stop_id), route_count) for stop_id, route_count in top5]

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    # pass  # Implementation here
    pairRoutes = defaultdict(set)

    for rID, stops in route_to_stops.items():
        for i in range(len(stops)-1):
            stopPair = (stops[i], stops[i+1])
            pairRoutes[stopPair].add(rID)
    
    unique = [(pair, list(routes)[0]) for pair, routes in pairRoutes.items() if len(routes) == 1]

    unique = [(pair, routeID, stop_trip_count[pair[0]] + stop_trip_count[pair[1]]) for pair, routeID in unique]

    top5 = sorted(unique, key=lambda x: x[2], reverse=True)[:5]

    return [(pair, route_id) for pair, route_id, _ in top5]

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # pass  # Implementation here
    G = nx.DiGraph()

    for rID, stops in route_to_stops.items():
        edges = [(stops[i], stops[i+1]) for i in range(len(stops)-1)]
        G.add_edges_from(edges, route=rID)

    pos = nx.spring_layout(G)

    edgeX = []
    edgeY = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edgeX += [x0, x1, None]
        edgeY += [y0, y1, None]

    edgeTrace = go.Scatter(
        x=edgeX, y=edgeY, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines'
    )

    nodeX = []
    nodeY = []
    nodeText = []
    for node in G.nodes():
        x, y = pos[node]
        nodeX.append(x)
        nodeY.append(y)
        nodeText.append(f"Stop ID: {node}")

    nodeTrace = go.Scatter(
        x=nodeX, y=nodeY, text=nodeText, mode='markers+text', textposition="top center", hoverinfo='text',
        marker=dict(
            showscale=False, color='#1f77b4', size = 10, line_width=2
        )
    )
    
    fig = go.Figure(data=[edgeTrace, nodeTrace],
                    layout=go.Layout(
                        title='<br>Stop-Route Graph',
                        titlefont_size = 16,
                        showlegend=False,
                        hovermode = 'closest',
                        margin = dict(b=0, l=0, r=0, t=0),
                        xaxis = dict(showgrid=False, zeroline=False),
                        yaxis = dict(showgrid=False, zeroline=False)
                    )
                    )
    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    # pass  # Implementation here
    directRoutes = []

    for rID, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            startIndex = stops.index(start_stop)
            endIndex = stops.index(end_stop)
            if startIndex < endIndex:
                directRoutes.append(int(rID))

    return directRoutes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    pyDatalog.load("""
                   DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
                   OptimalRoute(R1, R2, X, Y, Z) <= DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y) & (R1!=R2)
                   """)

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    # visualize_stop_route_graph_interactive(route_to_stops)
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # pass  # Implementation here
    for rID, stops in route_to_stops.items():
        for stopID in stops:
            pyDatalog.assert_fact('RouteHasStop', rID, stopID)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    # pass  # Implementation here
    result = pyDatalog.ask(f"DirectRoute(R, {start}, {end})")

    if result:
        rID = [int(r[0]) for r in result.answers]
        return sorted(rID)
    
    return []

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # pass  # Implementation here
    result = pyDatalog.ask(f"OptimalRoute(R1,R2, {start_stop_id},{end_stop_id},{stop_id_to_include})")
    if result:
        paths = result.answers
        formattedPaths = [(int(r1), int(stop_id_to_include), int(r2)) for r1, r2 in paths]
        return formattedPaths
    
    return []

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # pass  # Implementation here
    result = pyDatalog.ask(f"OptimalRoute(R1, R2, {start_stop_id}, {end_stop_id}, {stop_id_to_include})")
    if result:
        paths = result.answers
        formattedPaths = [(int(r2), int(stop_id_to_include), int(r1)) for r1, r2 in paths]
        return formattedPaths
    
    return []

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # pass  # Implementation here
    fPath = forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
    if fPath:
        return fPath
    
    bPath =  backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
    if bPath:
        return bPath
    
    path1 = forward_chaining(start_stop_id, stop_id_to_include, stop_id_to_include, max_transfers // 2)
    path2 = backward_chaining(stop_id_to_include, end_stop_id, stop_id_to_include, max_transfers // 2)
    if path1 and path2:
        return path1 + path2
    
    return []

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    # pass  # Implementation here
    return merged_fare_df[merged_fare_df['price'] <= initial_fare]

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    # pass  # Implementation here
    summary = defaultdict(lambda: {'min_price': float('inf'), 'stops': []})
    for _, row, in pruned_df.iterrows():
        rId = row['route_id']
        
        summary[rId]['min_price'] = min(summary[rId]['min_price'], row['price'])
        summary[rId]['stops'].append(row['origin_id'])
        summary[rId]['stops'].append(row['destination_id'])
        summary[rId]['stops'] = list(dict.fromkeys(summary[rId]['stops']))

    return summary

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    # pass  # Implementation here
    queue = deque([(start_stop_id,0,[],initial_fare)])
    visited = set()

    while queue:
        current, tfs, path,fLeft = queue.popleft()

        if tfs > max_transfers or fLeft < 0:
            continue
        
        if current == end_stop_id:
            return path

        for rID, rInfo in route_summary.items():
            if current in rInfo['stops']:
                for next in rInfo['stops']:
                    if next not in visited:
                        visited.add(next)
                        queue.append((next, tfs+1, path + [(rID, next)], fLeft - rInfo['min_price']))
    return []