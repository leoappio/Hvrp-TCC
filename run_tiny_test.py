import pulp
from main import build_hvrp_model, draw_solution

mdl = build_hvrp_model("problems/tiny_hvrp.vrp")
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=30)
mdl.solve(solver)

print("Status:", pulp.LpStatus[mdl.status])
print("FO:", pulp.value(mdl.objective))

for v in mdl.variables():
    if v.varValue and v.varValue > 1e-6:
        print(v.name, "=", v.varValue)

draw_solution(mdl, coords={1:(30,40),2:(35,44),3:(40,42),4:(32,36)}, depot=1, vehicle_types={"A":{}, "B":{}})
