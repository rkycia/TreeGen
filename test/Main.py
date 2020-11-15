## @package Main
#  Example of usage of Generator and ProbabilityTree.

import numpy as np
import pandas as pd
#from pandas import ExcelWriter
#from pandas import ExcelFile
import matplotlib.pyplot as plt
#import seaborn as sns
#import networkx as nx

#import is ok from test directory
#adjust it accordingly: 
import sys
sys.path.append('../')
import TreeGen.Generator as G


print("####################Read test data####################")
#perpare data:
df = pd.read_excel('Data.xls', sheetname='Data')
#df.head()
print(df.columns)

pSelected = df.loc[:, 'P1.1':'P6.8'].copy()

#example data
p1Selected=df.loc[:, 'P1.1':'P1.2'].copy()


#cleanining data
#p1Selected.fillna(0,inplace=1)
print(p1Selected.isnull().any())
print(p1Selected.isnull().sum())

p1Selected=p1Selected.dropna()  #remove rows with NaN

print(p1Selected.isnull().any())
print(p1Selected.isnull().sum())

#####Code starts here:

print("####################Construct Tree####################")
    
tree = G.ProbabilityTree( p1Selected , verbose= True)

print("####################Print Tree####################")
tree.printTree()

print("####################Tree Properties####################")
print( tree.getMaxRecord(True) )
print( tree.oracle([1,2], True))

print("####################Tree graph####################")
gr =tree.drawTree(verbose=True, show = False)
      

            
print("####################Generate record####################")
gen = G.Generator(tree)

#print(gen.getRecord(verbose = True))

"""    
genData = gen.getRecord()
for i in range (1000):
    df = gen.getRecord()
    #print( df )
    genData = genData.append( df )
"""
gen =G.Generator(tree)
print("gen = ",type(gen))
genData = gen.getRecords(1000)
        
print( genData.head() )


print("####################Compare results with data####################")

print("statistics of data:")
countData = p1Selected['P1.1'].value_counts()
probabilityData = countData.copy() / countData.sum()
print( probabilityData )

plt.bar(probabilityData.index, probabilityData.values, label="Data", color = 'r', alpha=0.5)

print("statistics of generated data:")
countGen = genData['P1.1'].value_counts()
probabilityGen = countGen.copy() / countGen.sum()
print( probabilityGen )


plt.xlabel("values")
plt.ylabel("probability/frequency")
plt.bar(probabilityGen.index, probabilityGen.values, label="Generated", color = 'g', alpha=0.5)
plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show()

print("difference:")
difference = (probabilityData - probabilityGen).copy()
print( difference )
plt.xlabel("values")
plt.ylabel("difference of probabilities")
plt.bar(difference.index, difference.values)
plt.show()

print("sum abs(difference) = ", np.abs(difference).sum())

print("####################Decrease of error when increase generation statistics####################")

def diff(generator, data, label ='P1.1', power10 = 4):
    x = [10**i for i in range(1,power10)]
    y1 = [generator.getRecords(k) for k in x ]
    countData = data[label].value_counts()
    probabilityData = countData.copy() / countData.sum()
    
    y2 = [ y[label].value_counts() for y in y1 ]
    y3 = [ y.copy()/y.sum() - probabilityData for y in y2 ]
    y = [np.abs(y).sum() for y in y3]
    
    return x,y

ret = diff(gen,  p1Selected, power10 =6 )
plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of records")
plt.ylabel("error")
plt.scatter(ret[0],ret[1])
plt.show()



