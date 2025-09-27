import pulp
from main import build_hvrp_model, draw_solution

mdl = build_hvrp_model("problems/problem-18.vrp")
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=86400)
mdl.solve(solver)

print("Status:", pulp.LpStatus[mdl.status])
print("FO:", pulp.value(mdl.objective))

for v in mdl.variables():
    if v.varValue and v.varValue > 1e-6:
        print(v.name, "=", v.varValue)


coords = {
    1:(30,40),
    2:(37,52),
    3:(49,49),
    4:(52,64),
    5:(20,26),
    6:(40,30),
    7:(21,47),
    8:(17,63),
    9:(31,62),
    10:(52,33),
    11:(51,21),
    12:(42,41),
    13:(31,32),
    14:(5,25),
    15:(12,42),
    16:(36,16),
    17:(52,41),
    18:(27,23),
    19:(17,33),
    20:(13,13),
    21:(57,58),
}

vehicle_types = {"A": {}, "B": {}, "C": {},"D": {},"E": {}}

draw_solution(mdl, coords=coords, depot=1, vehicle_types=vehicle_types)