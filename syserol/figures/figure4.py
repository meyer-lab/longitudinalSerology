from syserol.simulated import generate_simulated, imputeSim
from syserol.tensor import perform_contTF,  flatten_to3D
from syserol.figures.figure1 import R2X_Plots


def makeFigure():
    sim_tensor, _ = generate_simulated()
    imputeSim(sim_tensor, .5)
    sim_3d = flatten_to3D(sim_tensor)

    f = R2X_Plots(sim_tensor, sim_3d, fig4=True)
    
    return f