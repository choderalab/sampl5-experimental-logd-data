import pandas as pd
import pymc
from moremodels import DispensingErrorModel

# Input data from the preprocessing step.
table = pd.read_excel('processed.xlsx', sheetname='Filtered Data')
x = DispensingErrorModel(table)

# Do a MAP fit to get better starting values
# map = pymc.MAP(x, verbose=True)
# map.fit(verbose=True)

mc = pymc.MCMC(x,db='txt', dbname='sampl5_test.txt')
mc.sample(3000, save_interval=1000)

mc.db.close()


