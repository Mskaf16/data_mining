import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

# Load the data and set the parameters
pd.options.mode.chained_assignment = None
df = pd.read_excel("./Nomis_B_Data_Extract.xlsx", index_col=0)
df_acepted = df[df['Accept?'] == 1]
APR_acepted = df_acepted['APR'].to_numpy()
revenue_acepted = df_acepted['Net revenue per month'].to_numpy()
color = 'tab:blue'
bucket_size = 1/2
minAPR = math.floor(min(APR_acepted))
maxAPR = math.ceil(max(APR_acepted))
cut_bins = np.arange(minAPR, maxAPR+bucket_size, bucket_size).tolist()
mid_point_bins = [cut_bins[i]+(cut_bins[i+1]-cut_bins[i])*bucket_size for i in range(len(cut_bins)-1)]



# Figure 1: $ per Cust/Month/Accepted Quote vs. APR
df_acepted['APR bin'] = pd.cut(df_acepted['APR'], bins = cut_bins, include_lowest=True)
APR_bins_with_stats = df_acepted.groupby('APR bin').agg({'Net revenue per month': ['count', 'mean', 'std']})
APR_bins_with_stats.columns = ['count', 'mean', 'std']
APR_bins_with_stats['mid point bins'] = mid_point_bins

average_revenue = APR_bins_with_stats['mean']

plt.xlabel('Quoted APR')
plt.ylabel('$ per Cust/Month/Accepted Quote', color=color)
plt.title('Chart: 1. “$ per Cust/Month/Accepted Quote” vs. “Quoted APR”')
plt.xlim(int(minAPR), int(maxAPR))
offset = bucket_size / 2
ind = np.arange(minAPR+offset, maxAPR+offset, bucket_size)[:len(average_revenue)]
width_list = [ bucket_size * 0.8 for i in range(len(ind)) ]

plt.bar(ind, average_revenue, color=color, width=width_list)
plt.yticks(np.arange(0, max(average_revenue)*1.20, 10))
plt.tick_params(axis='y', labelcolor=color)
plt.grid(axis='y', color='grey', alpha=0.3)
plt.savefig('Figure 1.pdf', format='pdf')
plt.show()



# Figure 2: Conversion rate vs. APR
df['APR bin'] = pd.cut(df['APR'], bins = cut_bins, include_lowest=True)
APR_bins_with_stats = df.groupby('APR bin').agg({'Accept?': ['count', 'mean', 'std']})
APR_bins_with_stats.columns = ['count', 'mean', 'std']
APR_bins_with_stats['mid point bins'] = mid_point_bins

# The following line gets an array of floats
conversion_rate = APR_bins_with_stats[['mean']].to_numpy() 
# The following line converts the array of floats into a list needed to plot below
conversion_rate = [ i[0] for i in conversion_rate]

plt.xlabel('Quoted APR')
plt.ylabel('Conversion Rate', color=color)
plt.title('Chart: 2.“Conversion Rate” vs. “Quoted APR”')
plt.xlim(int(minAPR), int(maxAPR))
offset = bucket_size / 2
ind = np.arange(minAPR+offset, maxAPR+offset, bucket_size)[:len(conversion_rate)]
width_list = [ bucket_size * 0.8 for i in range(len(ind)) ]

plt.bar(ind, conversion_rate, color=color, width=width_list)
plt.yticks(np.arange(0, max(conversion_rate)*1.10, 0.1))
plt.tick_params(axis='y', labelcolor=color)
plt.grid(axis='y', color='grey', alpha=0.3)
plt.savefig('Figure 2.pdf', format='pdf')
plt.show()





# Figure 3: Number of quotes per bin vs. APR
# The following line should get a numpy array of integers
APRcountbins = APR_bins_with_stats[['count']].to_numpy()
# The following line converts the array of integers into a list needed to plot below
APRcountbins = [ i[0] for i in APRcountbins]

plt.xlabel('Quoted APR')
plt.ylabel('Number of Quotes', color=color)
plt.title('Chart: 3."Number of Quotes" vs. “Quoted APR”')
plt.xlim(int(minAPR), int(maxAPR))
offset = bucket_size / 2
ind = np.arange(minAPR+offset, maxAPR+offset, bucket_size)[:len(APRcountbins)]
width_list = [ bucket_size * 0.8 for i in range(len(ind)) ]

plt.bar(ind, APRcountbins, color=color, width=width_list)

plt.yticks(np.arange(0, max(APRcountbins)*1.20, 50))
plt.ylim(0, max(APRcountbins)*1.10)

plt.tick_params(axis='y', labelcolor=color)
plt.grid(axis='y', color='grey', alpha=0.3)

plt.savefig('Figure 3.pdf', format='pdf')
plt.show()

