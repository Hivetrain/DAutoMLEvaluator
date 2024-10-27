from deap import base, creator, gp, tools
from dml.ops import create_pset

class DeapToolboxFactory:
    @staticmethod
    def create_toolbox():
        pset = create_pset()
        
        # Create fitness and individual types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Create toolbox
        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        return toolbox, pset
