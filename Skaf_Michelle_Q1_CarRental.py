# CAR RENTAL PROBLEM: REALLOCATION OF CARS
# SEARH THE WORDS "TO BE FILLED IN BY STUDENTS"
import math

from pyomo.environ import *  # Import Pyomo environment


def main():
    
    # Define the problem instance
    # List of origin locations
    Loc_Orig = ['Loc 1', 'Loc 2']
    
    # List of destination locations
    Loc_Dest = ['Loc 3', 'Loc 4', 'Loc 5', 'Loc 6']
    
    # Dictionary specifying supply
    orig_supply = {'Loc 1':16, 
                   'Loc 2':18}

    # Dictionary specifying needs (lower bound)
    dest_need_LB = {'Loc 3':5, 
                    'Loc 4':5, 
                    'Loc 5':5, 
                    'Loc 6':5} 

    # TO BE FILLED IN BY STUDENTS: CREATE THE DICTIONARY THAT DESCRIBES THE DESTINATION LOCATIONS
    # AND THEIR CORRESPONDING UPPER BOUNDS FOR THEIR NEEDS
    # Dictionary specifying needs (upper bound)
    dest_need_UB = { 'Loc 3':10,
                     'Loc 4':10,
                     'Loc 5':10,
                     'Loc 6':10}



    # Dictionary specifying unit costs
    unit_cost = {('Loc 1', 'Loc 3'): 54.0,
                  ('Loc 1', 'Loc 4'): 17.0,
                  ('Loc 1', 'Loc 5'): 23.0,
                  ('Loc 1', 'Loc 6'): 30.0,
                  ('Loc 2', 'Loc 3'): 24.0,
                  ('Loc 2', 'Loc 4'): 18.0,
                  ('Loc 2', 'Loc 5'): 19.0,
                  ('Loc 2', 'Loc 6'): 31.0}

    model = ConcreteModel()


    # Define decision variables, one per possible (loc_orig, loc_dest) combination
    # Variable x[loc_orig, loc_dest] represents the shipping  volume from loc_orig to loc_dest
    model.x = Var(Loc_Orig, Loc_Dest, within=NonNegativeReals)


    # Define objective function
    # Min  x['Loc 1','Loc 3']*unit_cost['Loc 1','Loc 3'] + ....+x['Loc 1','Loc 6']*unit_cost['Loc 1','Loc 6']
    #      x['Loc 2','Loc 3']*unit_cost['Loc 2','Loc 3'] + ....+x['Loc 2','Loc 6']*unit_cost['Loc 2','Loc 6']  
    # Pay attention to the caps! Recall that the object names in Python are case sensitive.

    # TO BE FILLED IN BY STUDENTS: COMPLETE THE DEFINITION OF THE OBJECTIVE
    model.obj = Objective(
                expr = sum(model.x[loc_orig,loc_dest]*unit_cost[loc_orig,loc_dest] for loc_orig in Loc_Orig for loc_dest in Loc_Dest),
                sense = minimize)


    # Demand constraints
    # For demand at every destination loc_dest we define the constraints:
    #       dest_need_LB[loc_dest] <= x['Loc 1',loc_dest] + x['Loc 2',loc_dest] <= dest_need_UB[loc_dest] 
    # This is done in two steps: One for the lower bound (LB) and next for the upper bound (UB)
    def demand_LB_rule(model, loc_dest):
        return model.x['Loc 1',loc_dest] + model.x['Loc 2',loc_dest] >= dest_need_LB[loc_dest]  
    model.demand_LB_constraint = Constraint(Loc_Dest, rule=demand_LB_rule)

    # TO BE FILLED IN BY STUDENTS: ADD CONSTRAINTS FOR THE LOWER BOUND OF THE DEMAND NEEDS    
    def demand_UB_rule(model, loc_dest):
        return model.x['Loc 1',loc_dest] + model.x['Loc 2',loc_dest] <= dest_need_UB[loc_dest]
    model.demand_UB_constraint = Constraint(Loc_Dest, rule=demand_UB_rule)



    # TO BE FILLED IN BY STUDENTS: SUPPLY CONSTRAINTS 
    # Supply constraint
    # Compact definition of supply constraints
    # For every origin location we define the constraint:
    #   x[loc_orig,'Loc 3'] +...+ x[loc_orig,'Loc 6'] == orig_supply[loc_orig]
    def supply_rule(model, loc_orig):
            return model.x[loc_orig,'Loc 3'] +model.x[loc_orig,'Loc 4'] \
                + model.x[loc_orig,'Loc 5']+ model.x[loc_orig,'Loc 6'] == orig_supply[loc_orig]
    model.supply_constraint = Constraint(Loc_Orig, rule=supply_rule)

        
    # Solve the LP model
    opt = SolverFactory('glpk').solve(model)

    # Report results
    print('\n')
    print('RELOCATION PLAN:')
    print('Cost: {:.2f}'.format(value(model.obj)))
    print('\n')
    print('OPTIMAL SOLUTION:')
    for loc_orig in Loc_Orig:
        for loc_dest in Loc_Dest:
            if value(model.x[loc_orig,loc_dest])>0:
                print("x[{:^2s}, {:^2s}] = {:.0f}".format(loc_orig, loc_dest, value(model.x[loc_orig,loc_dest])))

    return   # End of the main function

# Call the main() function when running the script
if __name__ == "__main__":
    main()
