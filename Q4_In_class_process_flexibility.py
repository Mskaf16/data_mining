#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:35:43 2021

@author: michelleskaf
"""

# PROCESS FLEXIBILITY: DEDICATED, FULL FLEXIBLE, SHORT CHAIN, AND LONG CHAIN
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

###########################################################
# THESE LINES ARE COMING FROM THE LONG CHAIN DETERMINISTIC MODEL SCRIPT
import pandas as pd

from pyomo.environ import *  # Import Pyomo environment
###########################################################

def get_confidence_interval_for_mean(values):
    # Returns the 95% CI
    mu_v = np.mean(values)
    sigma_v = np.std(values)
    coef = 1.96
    n = len(values)

    mse=sigma_v/math.sqrt(n)
    CI_LB=mu_v - coef*mse
    CI_UB=mu_v + coef*mse

    return (CI_LB,CI_UB)



def get_sample_prob_minx(values,x):
    success = [v for v in values if v >= x]
    fraction = len(success)/len(values)
    return fraction



def simulate_performance(n_trials, plant_capacity, mu, sigma):
    # Simulation of performance

    sales_dedicated = []
    sales_full_flex = []
    sales_short_chain = []

    ############### NEW LINE: Prepare for the Long Chain configuration
    sales_long_chain = []
    ###############

    ###########################################################
    # THESE LINES ARE COMING FROM THE LONG CHAIN dETERMINISTIC MODEL SCRIPT

    # Next line reads the Excel file that describes the network structure
    df = pd.read_excel("LongChainNetwork.xlsx", header=0, index_col=0)
    # header=0: Row 0 is used for the column labels of the parsed DataFrame.
    # index_col=0: Column 0 is used as the row labels of the DataFrame. 

    # Define the list of plant names 
    P = list(df.index.map(str))

    # Define the list of demand names
    D = list(df.columns.map(str))

    # Define the network A as a dictionary with row labels in P and column labels in D
    A = {(p, d):df.at[p,d] for p in P for d in D}


    model = ConcreteModel()


    # Define decision variables, one per possible (plant,demand) combination
    # Variable x[p,d] represents the production volume of demand type d at plant p
    model.x = Var(P, D, within=NonNegativeReals)


    # Define objective function
    # Max  x['P1','D1']*A['P1','D1'] + X['P1','D2']*A['P1','D2'] +...+ x['P1','D6']*A['P1','D6'] + ... +
    #      x['P6','D1']*A['P6','D1'] + X['P6','D2']*A['P6','D2'] +...+ x['P6','D6']*A['P6','D6']  
    model.obj = Objective(
                expr = sum(model.x[p,d]*A[p,d] for p in P for d in D),
                sense = maximize )


    # Plant capacity constraints
    # Set plant capacity value
    plant_capacity=100

    # Compact definition of plant capacity constraints
    # For every plant p in P we define the constraint:
    #   x[p,'D1']**A[p,'D1'] + x[p,'D2']*A[p,'D2'] +...+ x[p,'D6']*A[p,'D6'] <= plant_capacity
    def plant_capacity_rule(model, p):
            return sum(model.x[p,d]*A[p,d] for d in D) <= plant_capacity
    model.plant_capacity = Constraint(P, rule=plant_capacity_rule)



    # Demand constraints
    # These constraints are configured so that their RHS may change 

    # Define an indexed Param component, which looks like a Python dictionary 
    # The argument "mutable=True" anticipates that the parameter may change 
    # between successive calls to the solver
    model.demand = Param(D, mutable=True)

    # Compact definition of demand constraints
    # For every demand type d in D we define the constraint:
    #   x['P1',d]*A['P1',d] + x['P2',d]*A['P2',d] +...+ x['P6',d]*A['P6',d] <= model.demand[d]
    def max_demand_rule(model, d):
        return sum(model.x[p,d]*A[p,d] for p in P) <= model.demand[d]
    model.demand_upper_bound = Constraint(D, rule=max_demand_rule)


    ###########################################################
    

    for i in range(n_trials):
        
        # The indented block below is all part of the "for" loop
        
        # Generate demand values for one particular trial
        # First we get a list of 6 random draws from the (normal) demand random variable
        demand_samples = stats.norm.rvs(loc=mu, scale=sigma, size=6)
        # The negative demands are replaced by 0s
        demand_samples=[max(d,0) for d in demand_samples]
    
        # Dedicated case
        sales_per_dedicated_plant = [min(plant_capacity,d) for d in demand_samples]
        # Add the new sales value to the list of sales for the dedicated config
        sales_dedicated.append(sum(sales_per_dedicated_plant))
    
        # Full flaxible case
        total_sales_full_flex=min(plant_capacity*6, sum(demand_samples))
        # Add the new sales value to the list of sales for the full flex config
        sales_full_flex.append(total_sales_full_flex)


        # Short chain
        total_sales_short_chain = \
                min(plant_capacity*2, demand_samples[0]+demand_samples[1]) + \
                min(plant_capacity*2, demand_samples[2]+demand_samples[3]) + \
                min(plant_capacity*2, demand_samples[4]+demand_samples[5])
        # Add the new sales value to the list of sales for the short chain config
        sales_short_chain.append(total_sales_short_chain)
 

        ###########################################################
        # NEXT LINES ARE COMING FROM THE LONG CHAIN dETERMINISTIC MODEL SCRIPT

        # Long chain        
        # Solve the LP model
        # Populate the RHS of the demand constraints with specific demand values
        for i in range(len(demand_samples)):
            model.demand[D[i]]=demand_samples[i]
        # Solve the LP model
        opt = SolverFactory('glpk').solve(model)
        # Add the new sales value to the list of sales for the long chain config
        sales_long_chain.append(value(model.obj))      
        ###########################################################
                          

    return (sales_dedicated, sales_full_flex, sales_short_chain, sales_long_chain)


def main():

    n_trials = 1000 # number of trials to run in the simulation
    
    plant_capac=100

    # Demand: Parameters of the normal distribution
    mu = 100
    sigma = 40

    # Next command sets the seed to generate the same sequence of pseudorandom 
    # numbers every time you run the script and hence it allows to replicate results
    # The argument is an arbitrary integer
    np.random.seed(2021)


    # NEXT LINE IS MODIFIED TO GET THE OUTPUT FROM THE LONG CHAIN
    (sales_dedicated, sales_full_flex, sales_short_chain, sales_long_chain) = simulate_performance(n_trials,plant_capac,mu,sigma)

    # Create histograms of the distribution of sales (probability density function)
    plt.hist(sales_dedicated,bins=300,density=True)
    plt.title('Dedicated config: Distribution pdf')
    # The following line saves the figure in the same folder where the python script is located
    # plt.savefig('Sales_dedicated_pdf.pdf', format='pdf')
    plt.show()

    plt.hist(sales_short_chain,bins=300,density=True)
    plt.title('Short chain config: Distribution pdf')
    # plt.savefig('Sales_short_chain_pdf.pdf', format='pdf')
    plt.show()
    
    plt.hist(sales_full_flex,bins=300,density=True)
    plt.title('Full flex config: Distribution pdf')
    # plt.savefig('Sales_full_flex_pdf.pdf', format='pdf')
    plt.show()


    ###########################################################
    #  CREATE THE HISTOGRAM FOR THE LONG CHAIN OUTPUT 
    plt.hist(sales_long_chain,bins=300,density=True)
    plt.title('Long chain config: Distribution pdf')
    # plt.savefig('Sales_long_chain_pdf.pdf', format='pdf')
    plt.show()



    # Report results
    print('\n')
    # Dedicated
    print('DEDICATED CONFIG.:')
    print('Sales sample mean: {:.2f}'.format(np.mean(sales_dedicated)))
    conf_int=get_confidence_interval_for_mean(sales_dedicated)
    print('95% CI for expected sales: ({:.2f}, {:.2f})'.format(conf_int[0], conf_int[1]))
    prob_600 = get_sample_prob_minx(sales_dedicated, 600)
    print('P(Sales=600)= {:.2%} \n'.format(prob_600))


    # Short chain
    print('SHORT CHAIN CONFIG.:')
    print('Sales sample mean: {:.2f}'.format(np.mean(sales_short_chain)))
    conf_int=get_confidence_interval_for_mean(sales_short_chain)
    print('95% CI for expected sales: ({:.2f}, {:.2f})'.format(conf_int[0], conf_int[1]))
    prob_600 = get_sample_prob_minx(sales_short_chain, 600)
    print('P(Sales=600)= {:.2%} \n'.format(prob_600))


    # Full flexible
    print('FULL FLEXIBLE CONFIG.:')
    print('Sales sample mean: {:.2f}'.format(np.mean(sales_full_flex)))
    conf_int=get_confidence_interval_for_mean(sales_full_flex)
    print('95% CI for expected sales: ({:.2f}, {:.2f})'.format(conf_int[0], conf_int[1]))
    prob_600 = get_sample_prob_minx(sales_full_flex, 600)
    print('P(Sales=600)= {:.2%} \n'.format(prob_600))

    ##### ADD LINES FOR THE LONG CHAIN OUTPUT
    # Long chain
    print('LONG CHAIN CONFIG.:')
    print('Sales sample mean: {:.2f}'.format(np.mean(sales_long_chain)))
    conf_int=get_confidence_interval_for_mean(sales_long_chain)
    print('95% CI for expected sales: ({:.2f}, {:.2f})'.format(conf_int[0], conf_int[1]))
    prob_600 = get_sample_prob_minx(sales_long_chain, 600)
    print('P(Sales=600)= {:.2%} \n'.format(prob_600))

    return


# Call the main() function when running the script
if __name__ == "__main__":
    main()