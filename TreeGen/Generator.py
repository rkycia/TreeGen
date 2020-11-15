## @package Generator
#  Module contains the class Generator that is Monte Carlo generator from the ProbabilityTree structure.



from  TreeGen.ProbabilityTree import *



##Generator class.
#Class containing functionality of the Monte Carlo generator that uses the Probability Tree to generate data with given probability.
class Generator:
    """Generator class.
    
    Class containing functionality of the Monte Carlo generator that uses the Probability Tree to generate data with given probability.
    """
    
    ##Constructor. Initializes generator with given Probability Tree.
    #@param ProbabilityTree Probability tree for generator initialization.
    def __init__( self, tree : ProbabilityTree = None ):
        """Constructor. Initializes generator with given Probability Tree.
        
        @param ProbabilityTree Probability tree for generator initialization.
        
        """
        
        ## @var tree 
        #  Probability tree
        self.tree = tree
        
        import random
        
        ## @var random  
        # Random number generator.
        self.rand = random
    
    ##Sets seed for internal random number generator
    def setSeed( self, seed ):
       """Sets seed for internal random number generator
       
       """
       self.rand.seed( seed )
    
    ##Generates a single record recursively walking through the tree.  
    #@param verbose Tells what it does.
    #@return Data Frame containing a single generated record.
    def getRecord( self, verbose : bool = False ):
        """Generates a single record recursively walking through the tree.
        
        @param verbose Tells what it does.
        @return Data Frame containing a single generated record.
        
        """
        #recursion over tree
        node = self.tree.getTree()
        tmp_list = []
        while node != None:
            element = self.__getElement( node, verbose )
            tmp_list.append( (element[0], [element[1].value]) )
            node = element[1].nextNode
            if verbose:
                print( tmp_list )
        
        #convert to data frame
        dataFrame =  pd.DataFrame.from_items( tmp_list ) 
        
        if verbose:
            print("Generated record: \n", dataFrame )
        
        return(dataFrame)
        
    ##For a given node the function generates an element with given probability.   
    #@param Node  Starting node.
    #@param verbose Tells what it does.   
    #@note It uses random number generator.
    def __getElement( self, node : Node = None, verbose : bool = False ):
        """For a given node the function generates an element with given probability.
        
        @param Node  Starting node.
        @param verbose Tells what it does.
        
        @note It uses random number generator.
        
        """
        if node != None:
            
            node = node
            columnName = node.columnName
            randomNumber = self.rand.random()
            if verbose:
                print( "Column name = ", columnName )
                print( "Random number = ", randomNumber )
        
            #create Cumulative Distribution Function and check which element to choose.
            cdf = 0.0
            fdata = node.data[0]
            for data in node.data:
                cdf = cdf + data.probability
                if verbose:
                    print( "value = ", data.value, ", probability = ", data.probability  )
                    print( "cdf = ", cdf )
                if randomNumber <= cdf:
                    fdata = data
                    break
            return ( columnName, fdata )
        else:
            raise ValueError("Tree is empty!")
     
    ##@param n Number of records to generate.
    #@return Data Frame contining n generated records.       
    def getRecords( self, n : int  = 1 ):
        """
        
        @param n Number of records to generate.
        @return Data Frame contining n generated records.
        
        """
        if n < 0:
            return(pd.DataFrame([]))
        genData = self.getRecord()
        for _ in range(n-1):
            genData = genData.append( self.getRecord() )
        return(genData)
    
  
#testing
if __name__ == "__main__":
    """import numpy as np
    
    print("####################Read test data####################")
    #perpare data:
    df = pd.read_excel('Data.xls', sheetname='Data')
    #df.head()
    print(df.columns)
    pSelected = df.loc[:, 'P1.1':'P6.8'].copy()

    #example data
    #p1Selected=df.loc[:, 'P1.1':'P1.4'].copy()
    #p1Selected=df.loc[:, 'P1.1':'P1.3'].copy()
    p1Selected=df.loc[:, 'P1.1':'P1.2'].copy()

    #cleanining data
    #p1Selected.fillna(0,inplace=1)
    print(p1Selected.isnull().any())
    print(p1Selected.isnull().sum())

    p1Selected=p1Selected.dropna()  #remove rows with NaN

    print(p1Selected.isnull().any())
    print(p1Selected.isnull().sum())
    
    print("####################Construct Tree####################")
    tree = ProbabilityTree( p1Selected , verbose= True)
    #tree = Tree( p1Selected )

    print("####################Generate records####################")
    gen = Generator(tree)
    genData = gen.getRecords(1000)
    print( genData.head() )
    
    
    print("####################Compare results with data####################")
    print("statistics of data:")
    countData = p1Selected['P1.1'].value_counts()
    probabilityData = countData.copy() / countData.sum()
    print( probabilityData )

    #sns.kdeplot(probabilityData, shade=True, color="r", label="Data")
    #sns.distplot(probabilityData, color="r", label="Data")
    #plt.bar(probabilityData.index, probabilityData.values, label="Data", color = 'w', hatch = '+' )
    plt.bar(probabilityData.index, probabilityData.values, label="Data", color = 'r', alpha=0.5)

    print("statistics of generated data:")
    countGen = genData['P1.1'].value_counts()
    probabilityGen = countGen.copy() / countGen.sum()
    print( probabilityGen )

    #sns.kdeplot(probabilityGen, shade=True, color="g", label="Generated")
    #sns.distplot(probabilityGen, color="g", label="Generated")
    #plt.bar(probabilityGen.index, probabilityGen.values, label="Generated", color = 'w', hatch = '/')
    plt.bar(probabilityGen.index, probabilityGen.values, label="Generated", color = 'g', alpha=0.5)
    plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.show()

    print("difference:")
    difference = (probabilityData - probabilityGen).copy()
    print( difference )
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

    ret = diff(gen,  p1Selected )
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("number of records")
    plt.ylabel("error")
    #plt.plot(ret[0],ret[1])
    plt.scatter(ret[0],ret[1])
    plt.savefig("errorDecrease.png")
    plt.show()"""
