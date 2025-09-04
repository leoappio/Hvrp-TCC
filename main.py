import math
import pulp
import networkx as nx
import matplotlib.pyplot as plt


def read_hvrp_instance(filepath: str):
    coords = {}
    demands = {}
    vehicle_types = {}
    fleet_count = {}
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
                i, x, y = ln.split()
                coords[int(i)] = (float(x), float(y))
            elif reading_demands:
                i, q = ln.split()
                demands[int(i)] = int(q)
            elif reading_vtypes:
                if ln.startswith("#"):
                    continue
                k, cap, fixed, var, cnt = ln.split()
                vehicle_types[k] = {
                    "capacity": float(cap),
                    "fixed_cost": float(fixed),
                    "var_cost": float(var),
                }
                fleet_count[k] = int(cnt)
            else:
                try:
                    val = int(ln)
                    if val != -1:
                        depot = val
                except ValueError:
                    pass

    if depot is None:
        raise ValueError("DEPOT_SECTION ausente ou inválida no arquivo .vrp")

    return coords, demands, depot, vehicle_types, fleet_count


def euclidean(i, j, coords):
    (x1, y1), (x2, y2) = coords[i], coords[j]
    return math.hypot(x1 - x2, y1 - y2)


def build_hvrp_model(instance_path: str) -> pulp.LpProblem:
    coords, demands, depot, Kp, Kcount = read_hvrp_instance(instance_path)

    nodes = sorted(coords.keys())
    clients = [i for i in nodes if i != depot]
    edges = [(i, j) for i in nodes for j in nodes if i != j]

    K_i = {i: [k for k, par in Kp.items() if demands[i] <= par["capacity"]]
           for i in clients}

    dij = {(i, j): euclidean(i, j, coords) for (i, j) in edges}
    cijk = {(i, j, k): Kp[k]["var_cost"] * dij[i, j] for (i, j) in edges for k in Kp}

    mdl = pulp.LpProblem("HVRP_MIP", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("y", ((i, j, k) for (i, j) in edges for k in Kp),
                              lowBound=0, upBound=1, cat="Binary")
    x = pulp.LpVariable.dicts("x", ((i, k) for i in clients for k in K_i[i]),
                              lowBound=0, upBound=1, cat="Binary")
    v = pulp.LpVariable.dicts("v", ((i, k) for i in nodes for k in Kp),
                              lowBound=0, cat="Continuous")
    r = {k: pulp.LpVariable(f"r_{k}", lowBound=0, upBound=Kcount.get(k, 0), cat="Integer") for k in Kp}

    mdl += (
        pulp.lpSum(Kp[k]["fixed_cost"] * y[depot, j, k] for j in clients for k in Kp) +
        pulp.lpSum(cijk[i, j, k] * y[i, j, k] for (i, j) in edges for k in Kp)
    )

    for i in clients:
        mdl += pulp.lpSum(x[i, k] for k in K_i[i]) == 1, f"assign[{i}]"

    for i in clients:
        for k in K_i[i]:
            mdl += pulp.lpSum(y[j, i, k] for j in nodes if j != i) == x[i, k], f"in_deg[{i},{k}]"
            mdl += pulp.lpSum(y[i, j, k] for j in nodes if j != i) == x[i, k], f"out_deg[{i},{k}]"

    for k in Kp:
        mdl += pulp.lpSum(y[depot, j, k] for j in clients) == r[k], f"routes_start[{k}]"
        mdl += pulp.lpSum(y[i, depot, k] for i in clients) == r[k], f"routes_end[{k}]"

    for k in Kp:
        for i in nodes:
            if (i, i) in [(a, b) for (a, b) in edges]:
                mdl += y[i, i, k] == 0, f"no_loop[{i},{k}]"

    for k in Kp:
        for (i, j) in edges:
            if (i in clients) and (k in K_i[i]):
                mdl += y[i, j, k] <= x[i, k], f"link_out[{i},{j},{k}]"
            if (j in clients) and (k in K_i[j]):
                mdl += y[i, j, k] <= x[j, k], f"link_in[{i},{j},{k}]"

        for j in clients:
            if k in K_i[j]:
                mdl += y[depot, j, k] <= x[j, k], f"link_dep_out[{j},{k}]"
        for i in clients:
            if k in K_i[i]:
                mdl += y[i, depot, k] <= x[i, k], f"link_dep_in[{i},{k}]"

    for k, par in Kp.items():
        Qk = par["capacity"]
        mdl += v[depot, k] == 0, f"load_depot[{k}]"

        for (i, j) in edges:
            if j != depot:
                mdl += v[j, k] >= v[i, k] + demands[j] - Qk * (1 - y[i, j, k]), f"load_flow[{i},{j},{k}]"

        for i in clients:
            if k in K_i[i]:
                mdl += v[i, k] >= demands[i] * x[i, k], f"load_min[{i},{k}]"
                mdl += v[i, k] <= Qk * x[i, k], f"load_max[{i},{k}]"
            else:
                for j in nodes:
                    if j != i:
                        mdl += y[i, j, k] == 0, f"forbid_out[{i},{j},{k}]"
                        mdl += y[j, i, k] == 0, f"forbid_in[{j},{i},{k}]"

    return mdl


def draw_solution(mdl, coords, depot, vehicle_types, thr=0.5):
    positions = {i: coords[i] for i in coords}

    arcs_by_type = {k: [] for k in vehicle_types.keys()}
    for var in mdl.variables():
        if not (var.name.startswith("y_") and var.varValue and var.varValue > thr):
            continue

        inside = var.name[var.name.find("(")+1:var.name.rfind(")")]
        i_str, j_str, k_str = inside.split(",_")
        i = int(i_str)
        j = int(j_str)
        k = k_str.strip("'")
        arcs_by_type.setdefault(k, []).append((i, j))

    routes = []
    for k, arcs in arcs_by_type.items():
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
    plt.legend(handles=handles, loc="best")

    plt.axis("off")
    plt.tight_layout()
    plt.show()
