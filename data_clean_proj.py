import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecademylib3_seaborn
import glob

files = glob.glob('state*.csv')

file_list = []
for i in files:
  reader = pd.read_csv(i)

  file_list.append(reader)

census_data = pd.concat(file_list)

print(census_data.head())
print(census_data.dtypes)

census_data.Income = census_data.Income.replace('[\$]', '', regex = True)
census_data.Income = pd.to_numeric(census_data.Income)


gender_split = census_data.GenderPop.str.split('_', expand = True)

census_data['Male_Pop'] = gender_split[0]
census_data['Female_Pop'] = gender_split[1]

census_data = census_data[['State', 'TotalPop', 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific', 'Income', 'Male_Pop', 'Female_Pop']]

census_data.Male_Pop = census_data.Male_Pop.str.replace('M', '')
census_data.Female_Pop = census_data.Female_Pop.str.replace('F', '')

census_data.Male_Pop = pd.to_numeric(census_data.Male_Pop)
census_data.Female_Pop = pd.to_numeric(census_data.Female_Pop)

print(census_data.head())
print(census_data.dtypes)

print(census_data.fillna(0))
census_data = census_data.drop_duplicates()
# print(census_data.duplicated().value_counts())
census_data.Hispanic = census_data.Hispanic.str.replace('%', '')
census_data.Hispanic = pd.to_numeric(census_data.Hispanic)
plt.hist(census_data.Hispanic)

plt.show()
plt.clf()

plt.scatter(census_data.Female_Pop, census_data.Income)
plt.show()
plt.clf()