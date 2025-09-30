import math
import pulp
import networkx as nx
import matplotlib.pyplot as plt


def read_hvrp_instance(filepath: str):
    coords, demands = {}, {}
    vehicle_types = {}
    reading_coords = reading_demands = reading_vtypes = False
    depot = None

    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("NODE_COORD_SECTION"):
                reading_coords, reading_demands, reading_vtypes = True, False, False
                continue
            if ln.startswith("DEMAND_SECTION"):
                reading_coords, reading_demands, reading_vtypes = False, True, False
                continue
            if ln.startswith("VEHICLE_TYPES"):
                reading_coords, reading_demands, reading_vtypes = False, False, True
                continue
            if ln.startswith("END_VEHICLE_TYPES"):
                reading_vtypes = False
                continue
            if ln.startswith("DEPOT_SECTION"):
                reading_coords = reading_demands = reading_vtypes = False
                continue
            if ln == "EOF":
                break

            if reading_coords:
                parts = ln.split()
                if len(parts) != 3:
                    continue
                i, x, y = parts
                coords[int(i)] = (float(x), float(y))
            elif reading_demands:
                parts = ln.split()
                if len(parts) != 2:
                    continue
                i, q = parts
                demands[int(i)] = int(q)
            elif reading_vtypes:
                if ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) != 4:
                    continue
                k, cap, fixed, var = parts
                vehicle_types[k] = {
                    "capacity": float(cap),
                    "fixed_cost": float(fixed),
                    "var_cost": float(var),
                }
            else:
                try:
                    val = int(ln)
                    if val != -1:
                        depot = val
                except ValueError:
                    pass

    if depot is None:
        raise ValueError("DEPOT_SECTION ausente ou inválida no arquivo .vrp")
    return coords, demands, depot, vehicle_types


def euclidean(i, j, coords):
    (x1, y1), (x2, y2) = coords[i], coords[j]
    return math.hypot(x1 - x2, y1 - y2)


def build_hvrp_model(instance_path: str) -> pulp.LpProblem:
    coords, demands, depot, Kp = read_hvrp_instance(instance_path)

    nodes = sorted(coords.keys())
    clients = [i for i in nodes if i != depot]
    M_types = sorted(Kp.keys())

    K_i = {i: [k for k, par in Kp.items() if demands[i] <= par["capacity"]] for i in clients}

    def cdist(i, j, k):
        return Kp[k]["var_cost"] * euclidean(i, j, coords)

    Y_index = []
    Y_out_dep = {k: [] for k in M_types}
    Y_in_dep = {k: [] for k in M_types}
    Y_ijk_by_k = {k: [] for k in M_types}

    for k in M_types:
        for j in clients:
            if k in K_i[j]:
                tpl = (depot, j, k)
                Y_index.append(tpl)
                Y_out_dep[k].append(tpl)
                Y_ijk_by_k[k].append(tpl)

        for i in clients:
            if k in K_i[i]:
                tpl = (i, depot, k)
                Y_index.append(tpl)
                Y_in_dep[k].append(tpl)
                Y_ijk_by_k[k].append(tpl)

        for i in clients:
            if k not in K_i[i]:
                continue
            for j in clients:
                if i == j or (k not in K_i[j]):
                    continue
                tpl = (i, j, k)
                Y_index.append(tpl)
                Y_ijk_by_k[k].append(tpl)

    cijk = {(i, j, k): cdist(i, j, k) for (i, j, k) in Y_index}

    Cik = {}
    for i in clients:
        for k in K_i[i]:
            Cik[(i, k)] = Kp[k]["fixed_cost"] + cdist(i, depot, k)

    mdl = pulp.LpProblem("HVRP4_MIP", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("y", Y_index, lowBound=0, upBound=1, cat="Binary")

    a_index = [(i, k) for i in clients for k in K_i[i]]
    b_index = [(i, k) for i in clients for k in K_i[i]]
    a = pulp.LpVariable.dicts("a", a_index, lowBound=0, upBound=1, cat="Binary")
    b = pulp.LpVariable.dicts("b", b_index, lowBound=0, upBound=1, cat="Binary")

    v = pulp.LpVariable.dicts("v", ((i, k) for i in clients for k in M_types), lowBound=0, cat="Continuous")

    mdl += (
        pulp.lpSum(Cik[(i, k)] * a[(i, k)] for (i, k) in a_index)
        + pulp.lpSum(cijk[(i, j, k)] * y[(i, j, k)] for (i, j, k) in Y_index)
    ), "Objective"

    for i in clients:
        mdl += pulp.lpSum(a[(i, k)] for k in K_i[i]) + pulp.lpSum(b[(i, k)] for k in K_i[i]) == 1, f"visit[{i}]"


    for i in clients:
        for k in K_i[i]:
            in_arcs = [(j, i, k) for (j, i2, k2) in Y_ijk_by_k[k] if i2 == i and j != i]
            mdl += pulp.lpSum(y[(j, i, k)] for (j, _, _) in in_arcs) == a[(i, k)] + b[(i, k)], f"in_deg[{i},{k}]"

    for i in clients:
        for k in K_i[i]:
            out_arcs = [(i, j, k) for (i2, j, k2) in Y_ijk_by_k[k] if i2 == i and j != i]
            mdl += pulp.lpSum(y[(i, j, k)] for (_, j, _) in out_arcs) == b[(i, k)], f"out_deg[{i},{k}]"

    for k in M_types:
        mdl += pulp.lpSum(y[(depot, j, k)] for (depot, j, _) in Y_out_dep[k]) == \
               pulp.lpSum(a[(i, k)] for i in clients if k in K_i[i]), f"depot_balance[{k}]"

    for k in M_types:
        Qk = Kp[k]["capacity"]
        for (i, j, kk) in Y_ijk_by_k[k]:
            if kk != k or i == depot:
                continue
            if (i in clients and k in K_i[i]) and (j in clients and k in K_i[j]):
                mdl += v[(j, k)] >= v[(i, k)] + demands[j] * (a[(j, k)] + b[(j, k)]) \
                       - Qk * (a[(i, k)] + b[(i, k)] - y[(i, j, k)]), f"mtz_flow[{i},{j},{k}]"

    for i in clients:
        for k in K_i[i]:
            incoming = [(j, i, k) for (j, i2, k2) in Y_ijk_by_k[k] if i2 == i and j != i]
            mdl += v[(i, k)] >= demands[i] * (a[(i, k)] + b[(i, k)]) + \
                   pulp.lpSum(demands[j] * y[(j, i, k)] for (j, _, _) in incoming), f"mtz_lb[{i},{k}]"

    for i in clients:
        for k in K_i[i]:
            Qk = Kp[k]["capacity"]
            mdl += v[(i, k)] <= Qk * (a[(i, k)] + b[(i, k)]), f"cap[{i},{k}]"

    return mdl


def draw_solution(mdl, coords, depot, vehicle_types, thr=0.5):
    positions = {i: coords[i] for i in coords}

    arcs_by_type = {k: [] for k in vehicle_types.keys()}
    for var in mdl.variables():
        if not (var.name.startswith("y_") and var.varValue and var.varValue > thr):
            continue
        inside = var.name[var.name.find("(") + 1: var.name.rfind(")")]
        i_str, j_str, k_str = inside.split(",_")
        i = int(i_str)
        j = int(j_str)
        k = k_str.strip("'")
        arcs_by_type.setdefault(k, []).append((i, j))

    routes = []
    for k, arcs in arcs_by_type.items():
        if not arcs:
            continue
        succ = {}
        for (i, j) in arcs:
            succ[i] = j

        starts = [j for (i, j) in arcs if i == depot]
        veh_num = 1
        for s in starts:
            route = [depot, s]
            cur = s
            visited = set(route)
            while cur in succ and succ[cur] != depot:
                nxt = succ[cur]
                if nxt in visited:
                    break
                route.append(nxt)
                visited.add(nxt)
                cur = nxt
            route.append(depot)
            routes.append((f"{k}{veh_num}", route))
            veh_num += 1

    G = nx.DiGraph()
    for _, route in routes:
        for i, j in zip(route[:-1], route[1:]):
            G.add_edge(i, j)

    plt.figure(figsize=(7, 7))
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple",
              "tab:brown", "tab:red", "tab:pink", "tab:olive", "tab:cyan"]
    color_map = {}
    for idx, (veh, _) in enumerate(routes):
        color_map[veh] = colors[idx % len(colors)]

    nx.draw_networkx_nodes(G, pos=positions, nodelist=[depot],
                           node_color="red", node_size=650, label="Depósito")
    nx.draw_networkx_nodes(G, pos=positions, nodelist=[i for i in coords if i != depot],
                           node_color="lightblue", node_size=520)

    for veh, route in routes:
        edges = list(zip(route[:-1], route[1:]))
        nx.draw_networkx_edges(G, pos=positions, edgelist=edges,
                               edge_color=color_map[veh], width=2.2,
                               arrows=True, arrowsize=15, label=veh)

    labels = {i: str(i) for i in coords}
    nx.draw_networkx_labels(G, pos=positions, labels=labels, font_size=9)

    from matplotlib.lines import Line2D
    handles = []
    for veh, _ in routes:
        handles.append(Line2D([0], [0], color=color_map[veh], lw=2, label=veh))
    handles.insert(0, Line2D([0], [0], marker='o', color='w', label='Depósito',
                             markerfacecolor='red', markersize=10))
    if handles:
        plt.legend(handles=handles, loc="best")

    plt.axis("off")
    plt.tight_layout()
    plt.show()
