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


def configure_hvrp_problem(problem_number: int):
    vehicles_3_5 = {
        "A": {"capacity": 20, "fixed_cost": 20, "var_cost": 1},
        "B": {"capacity": 30, "fixed_cost": 35, "var_cost": 1},
        "C": {"capacity": 40, "fixed_cost": 50, "var_cost": 1},
        "D": {"capacity": 70, "fixed_cost": 120, "var_cost": 1},
        "E": {"capacity": 120, "fixed_cost": 225, "var_cost": 1},
    }

    vehicles_4_6 = {
        "A": {"capacity": 60, "fixed_cost": 1000, "var_cost": 1},
        "B": {"capacity": 80, "fixed_cost": 1500, "var_cost": 1},
        "C": {"capacity": 150, "fixed_cost": 3000, "var_cost": 1},
    }

    depot_coord_default = (30, 40)
    depot_coord_20_20 = (20, 20)

    vehicles = vehicles_3_5 if problem_number in [3, 5] else vehicles_4_6
    depot_coord = depot_coord_default if problem_number in [3, 4] else depot_coord_20_20

    return vehicles, depot_coord


problem_number = 4

coords, demands = read_hvrp_instance("tests/E-n20-k5.vrp")

vehicle_types, new_depot_coord = configure_hvrp_problem(problem_number)
coords[1] = new_depot_coord

nodes = list(coords.keys())
depot = 1
clients = [i for i in nodes if i != depot]
edges = [(i, j) for i in nodes for j in nodes if i != j]

model = pulp.LpProblem("HVRP_Model", pulp.LpMinimize)

y = pulp.LpVariable.dicts("y", [(i, j, k) for (i, j) in edges for k in vehicle_types], cat="Binary")
a = pulp.LpVariable.dicts("a", [(i, k) for i in clients for k in vehicle_types], cat="Binary")
b = pulp.LpVariable.dicts("b", [(i, k) for i in clients for k in vehicle_types], cat="Binary")
v = pulp.LpVariable.dicts("v", [(i, k) for i in nodes for k in vehicle_types], lowBound=0, cat="Continuous")

# Função objetivo
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


# (4.2) Cada cliente é visitado exatamente uma vez
for i in clients:
    model += pulp.lpSum(a[i, k] + b[i, k] for k in vehicle_types) == 1


# (4.3) Fluxo de entrada = a + b
# (4.4) Fluxo de saída = b
for i in clients:
    for k in vehicle_types:
        model += pulp.lpSum(y[j, i, k] for j in nodes if j != i) == a[i, k] + b[i, k]
        model += pulp.lpSum(y[i, j, k] for j in nodes if j != i) == b[i, k]


# (4.5) Partida do depósito igual ao número de rotas finalizadas com a[i,k]
for k in vehicle_types:
    model += pulp.lpSum(y[depot, j, k] for j in clients) == pulp.lpSum(a[i, k] for i in clients)


# (4.6) Capacidade acumulada entre i → j
for k in vehicle_types:
    Q = vehicle_types[k]["capacity"]
    for i, j in edges:
        if i != depot:
            model += (
                v[j, k] >= v[i, k]
                + (demands[j] * (a[j, k] + b[j, k]) if j in clients else 0)
                - Q * (a[i, k] + b[i, k] - y[i, j, k])
            )


# (4.7) Carga mínima no cliente i considerando entrada de arcos
for i in clients:
    for k in vehicle_types:
        model += (
            v[i, k] >= demands[i] * (a[i, k] + b[i, k])
            + pulp.lpSum(demands[j] * y[j, i, k] for j in nodes if j != i)
        )


# (4.8) Carga máxima no cliente i
for i in clients:
    for k in vehicle_types:
        Q = vehicle_types[k]["capacity"]
        model += v[i, k] <= Q * (a[i, k] + b[i, k])

print('ok')