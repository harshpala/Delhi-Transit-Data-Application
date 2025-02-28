# Delhi-Transit-Data-Application
# 🚌 **Delhi Transit Data Application**

## 📚 **Project Overview**
This project focuses on building a transit data application to navigate Delhi's extensive bus network using GTFS static data from Delhi's Open Transit Data (OTD). The application efficiently handles route, trip, and stop data to provide clear and accurate transit information.

---

## 🚀 **Objective**
- Load and preprocess GTFS static transit data.
- Build a knowledge base (KB) for route and stop reasoning.
- Implement reasoning algorithms for direct and optimal route finding.
- Compare reasoning approaches (Brute-Force vs. FOL-based).
- Plan optimal routes using Forward and Backward Chaining.
- Analyze performance in terms of execution time, memory usage, and scalability.

---

## 🗂️ **Dataset**
The project uses the following GTFS static data files which can be downloaded from [Delhi’s Open Transit Data (OTD)](https://otd.delhi.gov.in/data/static/):
- **routes.txt**
- **trips.txt**
- **stop_times.txt**
- **stops.txt**
- **fare_rules.txt**

### **Key Terms:**
- **Start Stop:** Starting point of a journey.
- **End Stop:** Destination point of a journey.
- **Intermediate/Via Stop:** Stops for interchanges or route continuation.
- **Interchange:** Switching from one route to another.

### Static data file structure for Delhi buses
![Static data file structure for Delhi buses](figure.png)
---

## 🧠 **Knowledge Base Creation**
- Load static data files into structured Python data types.
- Create the following dictionaries:
  - `route_to_stops = {route_id: [list_of_stop_ids]}`
  - `trip_to_route = {trip_id: [list_of_route_ids]}`
  - `stop_trip_count = {stop_id: trip_count}`

### **Knowledge Base Queries:**
1. Top 5 busiest routes based on trip count.
2. Top 5 stops with the most frequent trips.
3. Top 5 busiest stops based on unique routes passing through them.
4. Top 5 stop pairs connected by exactly one direct route.

### **Graph Representation:**
Visualize the knowledge base using Plotly with route-to-stop mappings.

---

## 🤖 **Reasoning Algorithms** 
### **Direct Route Finding:**
#### **1. Brute-Force Approach:**
- Procedural logic to identify direct routes between stops.
- **Time Complexity:** O(R * S)
- **Memory Complexity:** Low

#### **2. FOL-Based Reasoning (PyDatalog):**
- Declarative approach using predicates and terms.
- **Time Complexity:** O(R + S)
- **Memory Complexity:** Moderate

#### **Comparison:**
- Brute-Force: Efficient for smaller datasets.
- FOL-Based: Scales better for larger datasets.

---

## 🗺️ **Optimal Route Planning** 
### **1. Forward Chaining:**
- Start from the start stop and propagate forward.
- Track valid paths adhering to constraints.
- **Time Complexity:** O(T * P)
- **Memory Complexity:** Higher due to path storage.

### **2. Backward Chaining:**
- Start from the end stop and work backward.
- Eliminate invalid paths early.
- **Time Complexity:** O(T * P)
- **Memory Complexity:** Lower than Forward Chaining.

#### **Comparison:**
- Forward Chaining: Better for dense networks.
- Backward Chaining: Better for sparsely connected stops.

---

## 🎯 **PDDL Implementation & Fare-Constrained Planning**
1. **PDDL Implementation:**
   - Define initial and goal states.
   - Implement actions: `board_route`, `transfer_route`.
2. **Fare-Constrained Planning:**
   - Optimize routes based on fare and maximum interchanges.
   - Use pruning techniques to improve computational efficiency.

---

## 📊 **Results & Analysis**
- **Direct Route Finding:** Brute-Force vs. FOL-Based Reasoning
  - Brute-Force: Faster for small datasets.
  - FOL-Based: More scalable.
- **Optimal Route Finding:** Forward vs. Backward Chaining
  - Forward: Suitable for dense networks.
  - Backward: Efficient for sparse connections.

---

## 📝 **Conclusion**
- Brute-Force is effective for small datasets, while FOL reasoning scales well.
- Forward chaining is ideal for dense connections; backward chaining excels in sparse ones.
- Fare-constrained planning ensures practical real-world applications.

---


## 📬 **Contact**
For queries or collaboration, feel free to reach out!

---

Happy Coding! 🚀

*Note: Analysis results and performance comparisons are based on analysis of the attached code and GTFS files.*

