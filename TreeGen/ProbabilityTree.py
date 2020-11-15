## @package ProbabilityTree
#  Module contains the class ProbabilityTree and helper classes Node and dataNode.
#
#  ProbabilityTree can acomodate data from DataFrame as a tree with frequencies of occurence of unique data in each column.
#  ProbabilityTree contains various useful function to operate on it. 
#  ProbabilityTree is an input for monte carlo generator.
 

#import numpy as np
import pandas as pd
#from pandas import ExcelWriter
#from pandas import ExcelFile
import matplotlib.pyplot as plt
#import seaborn as sns
import networkx as nx



##Contains node of probability tree.
class Node: 
    """ Contains node of probability tree.
    
    """
    ##Constructor
    #@param columnName The name of column in data frame.
    #@param data  List of dataNodes.
    def __init__(self, columnName = None, data = []): 
        """Constructor
        @param columnName The name of column in data frame.
        @param data  List of dataNodes.
        """
        self.columnName = columnName
        self.data = data


## Contains a frequency count of given unique value in the row of Data Frame.
class dataNode:
       
    
    ## Constructor
    #    @param value Unique value in the row of Data Frame.
    #    @param probability Frequency of occurence of value.
    #    @param nextNode contains next node reached from given node with given probability.
    def __init__(self, value = None, probability : float = 0.0, nextNode : Node = None):
        """Constructor
        @param value Unique value in the row of Data Frame.
        @param probability Frequency of occurence of value.
        @param nextNode contains next node reached from given node with given probability.
        """
        self.value = value
        self.probability = probability
        self.nextNode = nextNode        
   

## Contains probability tree created from Data Frame.     
class ProbabilityTree:
    """Contains probability tree created from Data Frame."""
    
    ##Helper function for constructing Probability Tree from Data Frame
    #@param dataFrame  Data Frame from which the tree is constructed.
    #@param verbose Tells what it does.
    #@param verboseShift Test shift at given level for verbose option.
    #It usues recursive call for constructing tree.
    def __buildTreeFromDataFrame(self, dataFrame : pd.DataFrame = None, verbose : bool = False, verobseShift : str = ""):
        """Helper function for constructing Probability Tree from Data Frame
        @param dataFrame  Data Frame from which the tree is constructed.
        @param verbose Tells what it does.
        @param verboseShift Test shift at given level for verbose option.
        It usues recursive call for constructing tree.
        """
        if dataFrame.columns.size == 0:
            return None
        else:
            #take first column name
            columnName = dataFrame.columns[0]
            if verbose:
                print(verobseShift + "Column name=", columnName)
            
            #calculate probability for column
            count = dataFrame[columnName].value_counts().sort_values(ascending = False)
            cumSum = count.sum()
            probability = count / float(cumSum)
            if verbose:
                print(verobseShift +"Probability:")
                print(probability)
            #construct list recursively
            tmp_list=[]
            for i in probability.index:
                if verbose:
                    print( verobseShift + "Creating subtree for:", i )
                    print( verobseShift + "Probability = ", probability[i] )
                    print( verobseShift + "Next data frame for subtree: \n", dataFrame.loc[dataFrame[columnName] == i].drop(columnName, axis = 1).copy().head() )
                tmp_list.append( dataNode( value = i, probability = probability[i], nextNode = self.__buildTreeFromDataFrame( dataFrame.loc[dataFrame[columnName] == i].drop(columnName, axis = 1).copy(), verbose, verobseShift = verobseShift + '\t' ) ) )
            #finalize tree construction
            node = Node( columnName = columnName, data = tmp_list )
            #return tree
            return node
    
    ##Constructor
    def __init__(self, dataFrame : pd.DataFrame = None, verbose : bool = False):
        """ Constructor
        
        """
        #save columns for duture reference
        self.columns = dataFrame.columns.copy()
        self.tree = self.__buildTreeFromDataFrame(dataFrame = dataFrame, verbose = verbose )
   

    ##@return The names and ordering of levels in tree.    
    #@note It is equivalend with the names and ordering of columns in the Data Frame used to initialize Probability Tree        
    def getColumns(self):
        """Returns the names and ordering of levels in tree.    
        @note It is equivalend with the names and ordering of columns in the Data Frame used to initialize Probability Tree        
        """
        return( self.columns )
    
    ##@return Probability tree - the first node.
    def getTree(self):
        """Returns probability tree - the first node
        """
        return(self.tree)
    
    ##Walks through the tree and prints its characteristics.
    def printTree(self):
        """Walks through the tree and prints its characteristics.
        """
        self.__printTreeHelper(self.tree)
       
    ##Helper function that walks recursively throught the tree and prints its characteristics.  
    #@param startNode Initial node to start the walk down.
    #@param prefix Shift of the printed text when moving from one level to the other.    
    def __printTreeHelper(self, startNode : Node = None, printPrefix : str = ''):
        """Helper function that walks recursively throught the tree and prints its characteristics.  
        @param startNode Initial node to start the walk down.
        @param prefix Shift of the printed text when moving from one level to the other.
        """
        node = startNode
        if node != None:
            print(printPrefix + "Column name = ", node.columnName)
            for data in node.data:
                print(printPrefix + "value = ", data.value, ", probability = ", data.probability)
            
            for data in node.data:
                print(printPrefix + "Entering ", node.columnName, " for  value ",  data.value, ":" )
                self.__printTreeHelper(data.nextNode, printPrefix + '\t')
        else:
            print(printPrefix +"None")
 
    
    ##Saves to file graps that visualize the tree.
    #@param filename PDF filename for saving graphs
    #@param graph_filename filename for saving graph data in NetworkX format
    #@param show display graphs in a window
    #@param verbose Tells what it does.
    #@warning It requires networkx library to work properly   
    #Since the tree is non-binary and its root contains many values in general, so 
    #the number of connected graphs is exactly the number of edges emanating from the root.
    def drawTree(self, filename : str = "ProbabilityTree.pdf", graph_filename : str = "graph.edgelist", show : bool= True, verbose : bool = False ):
        """ Saves to file graps that visualize the tree.
        @param filename PDF filename for saving graphs
        @param graph_filename filename for saving graph data in NetworkX format
        @param show display graphs in a window
        @param verbose Tells what it does.
        @warning It requires networkx library to work properly   
        
        Since the tree is non-binary and its root contains many values in general, so 
        the number of connected graphs is exactly the number of edges emanating from the root.    
        """
        graph = self.__drawTreeHelper( self.tree, verbose = verbose )
        
        if verbose:
            print("Edges: ",graph[0], "\n")
            print("Labels of edges: ", graph[1], "\n")
        
        G = nx.Graph()
        #G = nx.DiGraph() #not supported
        
        #set layout
        G.add_edges_from(graph[0])
        pos = nx.spring_layout(G)
        #pos = nx.circular_layout(G)
        #pos = nx.random_layout(G)
        #pos = nx.shell_layout(G)
        #pos = nx.spectral_layout(G)
        
        #save graph
        nx.write_edgelist(G, path="grid.edgelist", delimiter=":")

        #save graphs to multipage PDF
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(filename)
        plt.axis('off')
        
        #full graph:
        fig = plt.figure()
        nx.draw(G,pos,edge_color='black',width=1,linewidths=1, node_size=100, node_color='green',alpha=0.9, font_size =8, labels={node:node for node in G.nodes()})
        #nx.draw_networkx_edge_labels(G,pos,edge_labels=graph[1], font_color='black', font_size =8) #not used for better clarity of the picture
        plt.savefig(pp, format='pdf')
        if show:
            plt.show()
        plt.close(fig)
        #connected components
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for s in S:
            fig = plt.figure()
            nx.draw(s,pos,edge_color='black',width=1,linewidths=1, node_size=100, node_color='green',alpha=0.9, font_size =8, labels={node:node for node in s.nodes()})
            nx.draw_networkx_edge_labels(G,pos,edge_labels=graph[1], font_color='black', font_size =8)
            plt.axis('off')
            plt.savefig(pp, format='pdf')
            if show:
                plt.show()
            plt.close(fig)
        pp.close()
             
        
        return( G )
        
        
    ##Heleper function for drawTree().   
    #@param startNode Node from which the walk starts down.
    #@param label Label attached to the next level of recursion to the name of vertices.
    #@param verbose Tells what it does.    
    #It walks recursively through the tree and gather information about edges and their labels.
    def __drawTreeHelper(self, startNode : Node = None, label :str = "", verbose : bool = False):
        """Heleper function for drawTree(). 
        
        It walks recursively through the tree and gather information about edges and their labels.
        """                
        node = startNode
    
        edgeList = []
        edgeLabelList = {}
        
        inNodeLabel = node.columnName
        for n in node.data:
            if n.nextNode != None:
                outNodeLabel = n.nextNode.columnName
                labelValueIn = ", v={}|".format(n.value)
                labelProbability = "p={:.3f}".format(n.probability)
                #save all edges at given level
                for nn in n.nextNode.data:
                    labelValueOut = ", v={}|".format(nn.value)
                    edgeList = edgeList + [[label + inNodeLabel+labelValueIn, label + inNodeLabel+labelValueIn+ outNodeLabel+labelValueOut]]                
                    edgeLabelList[(label + inNodeLabel+labelValueIn, label + inNodeLabel+labelValueIn+ outNodeLabel+labelValueOut)] = labelProbability
        
                #recursive call
                elements = self.__drawTreeHelper( n.nextNode,  label + inNodeLabel+labelValueIn,  verbose )
                edgeList = edgeList + elements[0]
                edgeLabelList.update(elements[1])
            else:
                #we arein the leaf
                outNodeLabel = "Leaf"
                labelValueIn = ", v={}|".format(n.value)
                labelValueOut = ", v={}".format(n.value)
                labelProbability = "p={:.3f}".format(n.probability)
                edgeList = edgeList + [[label + inNodeLabel+labelValueIn, label + inNodeLabel+labelValueIn+ outNodeLabel+labelValueOut]]                
                edgeLabelList[(label + inNodeLabel+labelValueIn, label + inNodeLabel+labelValueIn+ outNodeLabel+labelValueOut)] = labelProbability                
                
        return edgeList, edgeLabelList
    
    ##Helper function for getMaxRecords working recursively on the tree.
    #@param node Node from which walk down starts.
    def __getMaxRecordHelper( self, node : Node, verbose : bool = False ):
        """Helper function for getMaxRecords working recursively on the tree.
        
        @param node Node from which walk down starts.
        """
        node = node
        valueList = []
        probabilityList = []
        if node != None:
            valueList = [node.data[0].value]
            probabilityList = [node.data[0].probability]
            if verbose:
                print("At node ", node.columnName)
                print("value is ", node.data[0].value)
                print("with probability ", node.data[0].probability)
                print("List of vaues: ", valueList)
                print("List of probabilities: ", probabilityList)
            #recursive call
            if node.data[0].nextNode != None:
                ret = self.__getMaxRecordHelper( node.data[0].nextNode, verbose )
                valueList = valueList + ret[0]
                probabilityList = probabilityList + ret[1]
        return valueList, probabilityList

    ## @verbose Tells what it does.
    #@return The tuple of list of values and the list of probabilities for the path containing highest probabiities. The order of lists is the same as returned by getColumns.
    def getMaxRecord( self, verbose : bool =  False ):
        """
        @verbose Tells what it does.
        @return The tuple of list of values and the list of probabilities for the path containing highest probabiities. The order of lists is the same as returned by getColumns.
        
        """
        node = self.tree
        return self.__getMaxRecordHelper( node, verbose ) 
    
    ##Checks if the record (list) is in the row. Ordering should be the same as returned by getColumns.  
    #@param record List of values to check. It can be shorter that the level of the tree but not larger.
    #@param verbose Tells what it does.
    #@return List of probabilities correspoding to the elements in the record list.
    def oracle(self, record : list, verbose : bool = False ):
        """Checks if the record (list) is in the row. Ordering should be the same as returned by getColumns.
        
        @param record List of values to check. It can be shorter that the level of the tree but not larger.
        @param verbose Tells what it does.
        @return List of probabilities correspoding to the elements in the record list.
        
        """
        if len(record) > len( self.columns ):
            raise AttributeError("Record is larger than number of columns: {}".format(self.columns))
        else:
            return(self.__oracleHelper(record, self.tree, verbose))
       
    ##Helper function for oracle function. Wlaks recursively through the tree and gather information.  
    #@param record List of values to check. It can be shorter that the level of the tree but not larger.
    #@param node Node from which the recursion starts.
    #@param verbose Tells what it does.
    def __oracleHelper(self, record : list, node : Node, verbose : bool = False ):
        """Helper function for oracle function. Wlaks recursively through the tree and gather information.
        
        @param record List of values to check. It can be shorter that the level of the tree but not larger.
        @param node Node from which the recursion starts.
        @param verbose Tells what it does.
        
        """
        if record == []:
            return []
        
        node = node
        probabilityList = []
        if node != None:
            if verbose:
                print("Visiting node: ", node.columnName)
            found = False
            for d in node.data:
                print("\t visiting data: ", d.value )
                if record[0] == d.value:
                    found = True
                    probabilityList = [d.probability]
                    if verbose:
                        print("At node ", node.columnName)
                        print("found value ", record[0])
                        print("with probability ", d.probability)
               
                    #recursive call
                    if  d.nextNode != None:
                        ret = self.__oracleHelper( record[1:], d.nextNode, verbose )
                        probabilityList = probabilityList + ret
                    break
            if found:    
                return( probabilityList )
            else:
                if verbose:
                    print("No element {} found. Set probability to 0.0. Finishing.".format(record[0]))
                return([0.0])
        
#testing
if __name__ == "__main__":
    """ 
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
    
    print("####################Print Tree####################")
    tree.printTree()
    
    print("####################Tree Properties####################")
    print( tree.getMaxRecord(True) )
    print( tree.oracle([1,2], True))

    print("####################Tree graph####################")
    gr =tree.drawTree(verbose=True)
    """
