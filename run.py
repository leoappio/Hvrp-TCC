import pulp
from main import build_hvrp_model, draw_solution, read_hvrp_instance

problem = "problems/problem-4-10-A.vrp"
timeLimit = 3600

mdl = build_hvrp_model(problem)
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=timeLimit)
mdl.solve(solver)

print("Status:", pulp.LpStatus[mdl.status])
print("FO:", pulp.value(mdl.objective))

coords, demands, depot, vehicle_types = read_hvrp_instance(problem)
draw_solution(mdl, coords=coords, depot=depot, vehicle_types=vehicle_types)
