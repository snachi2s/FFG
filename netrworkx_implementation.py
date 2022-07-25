"""
1. create a graph with 5 nodes and 4 vertices
2. run message passing algorithm in that graph
3. run sum-product algorithm in that graph
4. print the output
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# create a graph with 5 nodes and 4 vertices
G = nx.Graph()
G.add_nodes_from([1,2,3,4,5])
G.add_edges_from([(1,2),(1,3),(2,4),(3,4)])

# run message passing algorithm in that graph
def message_passing(G, max_iter=100, tol=1e-3):
    """
    message passing algorithm
    """
    # initialize the messages
    messages = {}
    for edge in G.edges():
        messages[(edge[0], edge[1])] = np.ones(2)
        messages[(edge[1], edge[0])] = np.ones(2)
    # iterate
    for i in range(max_iter):
        # update messages
        for edge in G.edges():
            # get the neighbors of edge[0]
            neighbors = list(G.neighbors(edge[0]))
            # remove edge[1] from neighbors
            neighbors.remove(edge[1])
            # compute the product of messages from neighbors to edge[0]
            product = np.ones(2)
            for neighbor in neighbors:
                product *= messages[(neighbor, edge[0])]
            # update the message from edge[0] to edge[1]
            messages[(edge[0], edge[1])] = product
            # update the message from edge[1] to edge[0]
            messages[(edge[1], edge[0])] = product
        # check convergence
        diff = 0
        for edge in G.edges():
            diff += np.sum(np.abs(messages[(edge[0], edge[1])] - messages[(edge[1], edge[0])]))
        if diff < tol:
            break
    # return the messages
    return messages

# run sum-product algorithm in that graph
def sum_product(G, max_iter=100, tol=1e-3):
    """
    sum-product algorithm
    """
    # initialize the messages
    messages = {}
    for edge in G.edges():
        messages[(edge[0], edge[1])] = np.ones(2)
        messages[(edge[1], edge[0])] = np.ones(2)
    # iterate
    for i in range(max_iter):
        # update messages
        for edge in G.edges():
            # get the neighbors of edge[0]
            neighbors = list(G.neighbors(edge[0]))
            # remove edge[1] from neighbors
            neighbors.remove(edge[1])
            # compute the sum of messages from neighbors to edge[0]
            sum = np.zeros(2)
            for neighbor in neighbors:
                sum += messages[(neighbor, edge[0])]
            # update the message from edge[0] to edge[1]
            messages[(edge[0], edge[1])] = sum
            # update the message from edge[1] to edge[0]
            messages[(edge[1], edge[0])] = sum
        # check convergence
        diff = 0
        for edge in G.edges():
            diff += np.sum(np.abs(messages[(edge[0], edge[1])] - messages[(edge[1], edge[0])]))
        if diff < tol:
            break
    # return the messages
    return messages

# print the output
print("message passing algorithm:")
print(message_passing(G))
print("sum-product algorithm:")
print(sum_product(G))

# draw the graph
nx.draw(G, with_labels=True)
plt.show()
