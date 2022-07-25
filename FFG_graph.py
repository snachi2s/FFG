"""
Created FFG in networkx with reference to toy problem 
Basic msg passing algorithm with two edge inputs 
message passing function -> only has the product of messages
sum product function -> only has the sum of messages
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

ffg = nx.Graph()
factor_nodes = ['f1', 'f2', 'f3', 'f4', 'f5']
variable_edges = ['X1', 'X2', 'X3', 'l1', 'l2']
ffg.add_nodes_from(factor_nodes)
ffg.add_edges_from([('f1', ''), ('f2', 'f1'),
                    ('f1', 'f3'), ('f3', 'f4'), ('f4', 'f5')])

edge_labels = {('f1', ''): 'X1', ('f2', 'f1'): 'l1',
               ('f1', 'f3'): 'X2', ('f3', 'f4'): 'X3', ('f4', 'f5'): 'l2'}

# display the graph

# pos=nx.spring_layout(ffg)
# nx.draw(ffg, pos, with_labels = True)
# nx.draw_networkx_edge_labels(ffg, pos, edge_labels=edge_labels)
# plt.show()

# message passing algorithm
def message_passing(ffg, max_iter=10, tol=1e-3):
    """
    message passing algorithm
    """
    messages = {}
    for edge in ffg.edges():
        print(edge)
        messages[(edge[0], edge[1])] = np.array([0.2, 0.8])
        messages[(edge[1], edge[0])] = np.array([0.1, 0.9])
    for i in range(max_iter):
        for edge in ffg.edges():
            neighbors = list(ffg.neighbors(edge[0]))
            neighbors.remove(edge[1])
            product = np.ones(2)
            for neighbor in neighbors:
                product *= messages[(neighbor, edge[0])]
            messages[(edge[0], edge[1])] = product
            messages[(edge[1], edge[0])] = product
        diff = 0
        for edge in ffg.edges():
            diff += np.sum(np.abs(messages[(edge[0], edge[1])
                                           ] - messages[(edge[1], edge[0])]))
        if diff < tol:
            break
    return messages


print("product part")
sm = message_passing(ffg, max_iter=10, tol=1e-3)
print(sm)



def sum_product(ffg, max_iter=100, tol=1e-3):
    """
    sum-product algorithm
    """
    messages = {}
    for edge in ffg.edges():
        messages[(edge[0], edge[1])] = np.ones(2)
        messages[(edge[1], edge[0])] = np.ones(2)
    for i in range(max_iter):
        for edge in ffg.edges():
            neighbors = list(ffg.neighbors(edge[0]))
            neighbors.remove(edge[1])
            sum = np.zeros(2)
            for neighbor in neighbors:
                sum += messages[(neighbor, edge[0])]
            messages[(edge[0], edge[1])] = sum
            messages[(edge[1], edge[0])] = sum
        diff = 0
        for edge in ffg.edges():
            diff += np.sum(np.abs(messages[(edge[0], edge[1])] - messages[(edge[1], edge[0])]))
        if diff < tol:
            break
    return messages


print("Sum part")
sp = sum_product(ffg, max_iter=10, tol=1e-3)
print(sp)
