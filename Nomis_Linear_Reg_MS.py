import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_excel("Nomis_B_Data_Extract.xlsx", index_col=0)


# For linear regression, the input parameters have to be numpy arrays
# The following lines should get relevant columns from the the dataframe and turn them into
# numpy arrays (by using the to_numpy() function)
AcceptYN = df['Accept?'].to_numpy()
APR = df['APR'].to_numpy()


# Create the Linear Regression model 
modelLinReg = LinearRegression()
# Arguments: (independent, dependent) variables 
# In order to fit the model, the independent x-values go as a column vector, 
# and the dependent y-values go as a row vector 
# The function ravel() transpose an array
modelLinReg.fit(APR.reshape(-1,1), AcceptYN.reshape(-1,1))  
r_sq = modelLinReg.score(APR.reshape(-1,1), AcceptYN.reshape(-1,1))

print('\n Attraction of the APR based on Linear Regression:\n')
print("{:>20s}{:>6.3f}".format('Intercept: ', modelLinReg.intercept_[0]))
print("{:>20s}{:>6.3f}\n".format('APR coefficient: ', modelLinReg.coef_[0][0]))
print("{:>20s}{:>6.3f}\n".format('R^2: ', r_sq))



# Plot jointly the regression results and the related confidence intervals for the Accept? values 
# as a measure of goodness-of-fit

# The following lines prepare bins for accept/reject decisions based on different values of APR
# First, we set the threshold values for the bins (e.g., cut_bins = [4, 4.5, 5, 5.5, ...])
minAPR = math.floor(min(APR))
maxAPR = math.ceil(max(APR))
cut_bins = np.arange(minAPR, maxAPR+0.5, 0.5).tolist()

# Second, we create a new dataframe only based on APR values used for segmenting the database
# For each APR segment (say, between APR=4 and APR=4.5) we compute summary statistics related to the
# contained Accept? values
mid_point_bins = [cut_bins[i]+(cut_bins[i+1]-cut_bins[i])*0.5 for i in range(len(cut_bins)-1)]
df['APR bin'] = pd.cut(df['APR'], bins = cut_bins, include_lowest=True)
APR_bins_with_stats = df.groupby('APR bin').agg({'Accept?': ['count', 'mean', 'std']})
APR_bins_with_stats.columns = ['count', 'mean', 'std']
APR_bins_with_stats['mid point bins'] = mid_point_bins

# For each bin we compute the mean and half-width 95% CIs (1.96*MSE) values for the Accept? values contained therein
APRmeanbins = APR_bins_with_stats[['mean']].to_numpy() 
# The error is one side of the 95% CI for the expected Accept? decision
y_error = 1.96*APR_bins_with_stats[['std']].to_numpy()/np.sqrt(APR_bins_with_stats[['count']].to_numpy())



# Build the chart 
plt.figure(1, figsize=(5, 4))
plt.clf()

# Plot the Accept?  Yes/No decisions
plt.scatter(APR.ravel(), AcceptYN, color="black", s=1)

# Plot the fitted curve for the P(Accept|APR)
lower_x = min(cut_bins)
upper_x = max(cut_bins)
x = np.linspace(lower_x, upper_x, 400)
print(x)
prob_accept = modelLinReg.predict(x.reshape(-1,1)) # TO BE COMPLETED BY STUDENTS: P(accept|APR=x) based on linear regression output
prob_accept = prob_accept.ravel() 

plt.plot(x, prob_accept, color="red", linewidth=1)
# Plot the 95% CI for the predicted APR for each bin
plt.errorbar(mid_point_bins, APRmeanbins, yerr = y_error[:,0], fmt ='o', elinewidth=1, capsize=2, markersize=2)

plt.ylabel("Accept?", fontsize=10)
plt.xlabel("APR", fontsize=10)
plt.title("Predictive Model - Linear Regression")
plt.xticks(range(int(lower_x), int(upper_x)), fontsize=8)
plt.yticks([-0.25, 0, 0.25, 0.5, 0.75, 1], fontsize=8)
plt.ylim(-0.30, 1.05)
plt.xlim(int(lower_x), int(upper_x)-1)
plt.tight_layout()
plt.savefig('LinearRegApprox.pdf', format='pdf')
plt.show()

####### LOGISTIC REG EXTRA

from sklearn.linear_model import LogisticRegression
modelLogReg = LogisticRegression()
# Arguments: (independent, dependent) variables 
# In order to fit the model, the independent x-values go as a column vector, 
# and the dependent y-values go as a row vector 
# The function ravel() transpose an array
modelLogReg.fit(APR.reshape(-1,1), AcceptYN.reshape(-1,1))  
r_sq = modelLogReg.score(APR.reshape(-1,1), AcceptYN.reshape(-1,1))
print(r_sq)
modelLogReg.predict_proba

lower_x = min(cut_bins)
upper_x = max(cut_bins)
x = np.linspace(lower_x, upper_x, 400)
print(x)
prob_accept = modelLogReg.predict_proba(x.reshape(-1,1))[:,1] # TO BE COMPLETED BY STUDENTS: P(accept|APR=x) based on linear regression output
prob_accept = prob_accept.ravel() 

plt.plot(x, prob_accept, color="red", linewidth=1)
# Plot the 95% CI for the predicted APR for each bin
plt.errorbar(mid_point_bins, APRmeanbins, yerr = y_error[:,0], fmt ='o', elinewidth=1, capsize=2, markersize=2)

plt.ylabel("Accept?", fontsize=10)
plt.xlabel("APR", fontsize=10)
plt.title("Predictive Model - Logistic Regression(Alternet Solution)")
plt.xticks(range(int(lower_x), int(upper_x)), fontsize=8)
plt.yticks([-0.25, 0, 0.25, 0.5, 0.75, 1], fontsize=8)
plt.ylim(-0.30, 1.05)
plt.xlim(int(lower_x), int(upper_x)-1)
plt.tight_layout()
plt.savefig('LogisticRegApprox.pdf', format='pdf')
plt.show()