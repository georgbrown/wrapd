Output data and meshes will be sent here by default.

final_mesh.node -- Final solution (TetMesh file, vertex data)
final_mesh.ele  -- Final solution (TetMesh file, element data)
final_mesh.obj  -- Final solution (OBJ file, vertices and faces)

x_errors.txt -- ||X-X*|| / ||X_init-X*|| at each ADMM iteration, where X, X*, and X_init
                are the current, solution, and initial positions, respectively.

objectives.txt -- Total objective at each ADMM iteration.
accumulated_time_s.txt -- Total accumulated runtime at each ADMM iteration
reweighted.txt -- Identifies which ADMM iterations had a reweighting operation (1=yes, 0=no)
