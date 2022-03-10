from syserol.simulated import generate_simulated, imputeSim
from syserol.tensor import perform_CMTF,  flatten_to3D
from syserol.figures.figure1 import makeFigure as fig1

def makeFigure():
    sim_tensor, _ = generate_simulated()
    imputeSim(sim_tensor, .5)
    sim_3d = flatten_to3D(sim_tensor)

    f = fig1(sim_tensor, sim_3d)
    
    return f