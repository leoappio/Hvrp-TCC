import pulp
import math

def read_hvrp_instance(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    coords = {}
    demands = {}
    reading_coords = False
    reading_demands = False

    for line in lines:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            reading_coords = True
            continue
        if line == "DEMAND_SECTION":
            reading_coords = False
            reading_demands = True
            continue
        if line.startswith("DEPOT_SECTION") or line == "EOF":
            break

        if reading_coords:
            parts = line.split()
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            coords[node_id] = (x, y)
        elif reading_demands:
            parts = line.split()
            node_id = int(parts[0])
            demand = int(parts[1])
            demands[node_id] = demand

    return coords, demands


def euclidean(i, j, coords):
    x1, y1 = coords[i]
    x2, y2 = coords[j]
    return round(math.hypot(x1 - x2, y1 - y2), 2)



coords, demands = read_hvrp_instance("tests/E-n20-k5.vrp")
nodes = list(coords.keys())
depot = 1
clients = [i for i in nodes if i != depot]
edges = [(i, j) for i in nodes for j in nodes if i != j]

vehicle_types = {
    "A": {"capacity": 20, "fixed_cost": 20, "var_cost": 1},
    "B": {"capacity": 30, "fixed_cost": 35, "var_cost": 1},
    "C": {"capacity": 40, "fixed_cost": 50, "var_cost": 1},
    "D": {"capacity": 70, "fixed_cost": 120, "var_cost": 1},
    "E": {"capacity": 120, "fixed_cost": 225, "var_cost": 1},
}

model = pulp.LpProblem("HVRP_Model", pulp.LpMinimize)


# y[i,j,k] = 1 se arco (i,j) é usado pelo veículo tipo k
y = pulp.LpVariable.dicts("y", [(i, j, k) for (i, j) in edges for k in vehicle_types], cat="Binary")


# a[i,k] = 1 se o cliente i é o último da rota com veículo k
a = pulp.LpVariable.dicts("a", [(i, k) for i in clients for k in vehicle_types], cat="Binary")


# b[i,k] = 1 se o cliente i é visitado por veículo k e não é o último
b = pulp.LpVariable.dicts("b", [(i, k) for i in clients for k in vehicle_types], cat="Binary")


# v[i,k] = carga acumulada no veículo k ao chegar em i
v = pulp.LpVariable.dicts("v", [(i, k) for i in nodes for k in vehicle_types], lowBound=0, cat="Continuous")


# funçao objetivo
model += (
    pulp.lpSum(
        a[i, k] * (vehicle_types[k]["fixed_cost"] + vehicle_types[k]["var_cost"] * euclidean(i, depot, coords))
        for i in clients for k in vehicle_types
    ) +
    pulp.lpSum(
        y[i, j, k] * vehicle_types[k]["var_cost"] * euclidean(i, j, coords)
        for (i, j) in edges for k in vehicle_types
    )
)


# cada cliente atendido uma unica vez
for i in clients:
    model += pulp.lpSum(a[i, k] + b[i, k] for k in vehicle_types) == 1


# conservacao de fluxo
for i in clients:
    for k in vehicle_types:
        model += pulp.lpSum(y[j, i, k] for j in nodes if j != i) == a[i, k] + b[i, k]
        model += pulp.lpSum(y[i, j, k] for j in nodes if j != i) == b[i, k]


# veículo sai e retorna do depósito no máximo uma vez
for k in vehicle_types:
    model += pulp.lpSum(y[depot, j, k] for j in clients) <= 1
    model += pulp.lpSum(y[i, depot, k] for i in clients) <= 1


# restrições de capacidade com carga acumulada
for k in vehicle_types:
    Q = vehicle_types[k]["capacity"]
    for i in clients:
        model += v[i, k] >= demands[i] * (a[i, k] + b[i, k])
        model += v[i, k] <= Q * (a[i, k] + b[i, k])
    for i, j in edges:
        if i != depot and j != depot:
            model += v[j, k] >= v[i, k] + demands[j] - Q * (1 - y[i, j, k])

