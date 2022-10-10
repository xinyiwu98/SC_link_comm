import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from numpy import random
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sknetwork.clustering import modularity
from collections import Counter
from scipy.special import comb
from collections import defaultdict



arr = lambda x : np.array(list(x))
    
#constructing boundary maps B_1 & B_2 and the adjacency matrices C, D, E1 and A_rw_hat

#contruct B_1 for synthetic networkx graph object
def generate_B1(G):
    B1 = np.zeros((len(G),len(G.edges)))
    
    for index, (u,v) in enumerate(G.edges):
        if u < v :
            B1[u-1,index] = -1
            B1[v-1,index] = 1
        else:
            B1[u-1,index] = 1
            B1[v-1,index] = -1
    return B1

#get all triangle lists of the networkx graph object
def get_triangles_list(G):
    triangle_list = []
    for index, (u,v) in enumerate(G.edges):
        for node in G:
            # check edge (u, node)
            if node != u:
                edge_u_node = (min(node, u), max(node, u))
                if edge_u_node in G.edges:
                    # check edge (v, node)
                    if node != v:
                        edge_v_node = (min(node, v), max(node, v))
                        if edge_v_node in G.edges:
                            triangle = [u, v, node]
                            triangle.sort()
                            if triangle not in triangle_list:
                                triangle_list.append(triangle) 
    return triangle_list

#contruct B_2 for networkx graph object: clique complex
def generate_B2(G):
    triangles = get_triangles_list(G)
    B2 = np.zeros((len(G.edges),len(triangles)))
    for index, [a,b,c] in enumerate(triangles):
        e1 = list(G.edges).index((a,b))
        e2 = list(G.edges).index((a,c))
        e3 = list(G.edges).index((b,c))
        B2[e1,index] = 1
        B2[e2,index] = -1
        B2[e3,index] = 1
    return B2

#contruct B_1 for networkx graph object based on actual dataset
def generate_B1ca(G):
    B1 = np.zeros((len(G),len(G.edges)))

    for index, (u,v) in enumerate(G.edges):
        index_u = np.where(np.array(G.nodes) == u)
        index_v = np.where(np.array(G.nodes) == v)
        B1[index_u,index] = -1
        B1[index_v,index] = 1
    return B1

#contruct B_2 for networkx graph object based on actual dataset
def generate_B2ca(G,triangle_list):
    B2 = np.zeros((len(G.edges),len(triangle_list)))
    for i in range(len(triangle_list_subset)):
        node_1 = triangle_list.iloc[i][0]
        node_2 = triangle_list.iloc[i][1]
        node_3 = triangle_list.iloc[i][2]
        index_1_candidate = np.where(edge_list[0]== node_1)[0]
        index_2f_candidate = np.where(edge_list[0]== node_2)[0]
        index_2b_candidate = np.where(edge_list[1]== node_2)[0]
        index_3_candidate = np.where(edge_list[1]== node_3)[0]
        index_12 = np.intersect1d(index_1_candidate, index_2b_candidate)[0]
        index_13 = np.intersect1d(index_1_candidate, index_3_candidate)[0]
        index_23 = np.intersect1d(index_2f_candidate, index_3_candidate)[0]
        B2[index_12,i] = 1
        B2[index_13,i] = -1
        B2[index_23,i] = 1
    return B2

#contruct B_2 for the largest connected component of networkx graph object based on actual dataset
def generate_B2ca_cc(G,triangle_list):
    B2 = np.zeros((len(G.edges),len(triangle_list)))
    for i in range(len(triangle_list)):
        node_1 = triangle_list.iloc[i][0]
        node_2 = triangle_list.iloc[i][1]
        node_3 = triangle_list.iloc[i][2]
        for index, (u,v) in enumerate(G.edges):

            if u == node_1 and v == node_2:
                index_12 = index
                B2[index_12,i] = 1
            elif u == node_1 and v == node_3:
                index_13 = index
                B2[index_13,i] = -1
            elif u == node_2 and v == node_3:
                index_23 = index
                B2[index_23,i] = 1
    return B2

#contruct baseline C
def generate_C(B1):
    C = np.abs(B1.T)@np.abs(B1)
    for i in range(C.shape[0]):
        C[i,i] = 0
    return C

#contruct baseline D
def generate_D(B1):
    deg = np.abs(B1)@np.ones(B1.shape[1])
    D = np.abs(B1.T)@np.diag(1/(np.maximum(deg,2)-1))@np.abs(B1)
    for i in range(D.shape[0]):
        D[i,i] = 0
    return D

#contruct baseline E1
def generate_E1(B1):
    deg = np.abs(B1)@np.ones(B1.shape[1])
    E = np.abs(B1.T)@np.diag(1/deg)@np.abs(B1)
    return E@E-E

#build A_rw_hat

def generate_B1_hat(B1,V):
    B1_hat = B1@np.transpose(V)
    return B1_hat

def generate_B2_hat(B2,V):
    B2_hat = V@B2
    return B2_hat

def generate_plus(M):
    s = M.shape
    zero = np.zeros(s)
    M_plus = np.maximum(M,zero)
    return M_plus

def generate_minus(M):
    s = M.shape
    zero = np.zeros(s)
    M_minus = np.maximum(-M,zero)
    return M_minus

def generate_A_l_hat(B1,V):
    B1_hat = generate_B1_hat(B1,V)
    B1_hat_plus = generate_plus(B1_hat)
    B1_hat_minus = generate_minus(B1_hat)
    A_l_hat = np.transpose(B1_hat_minus)@B1_hat_plus+np.transpose(B1_hat_plus)@B1_hat_minus
    return A_l_hat

def generate_A_u_hat(B2,V):
    B2_hat = generate_B2_hat(B2,V)
    B2_hat_plus = generate_plus(B2_hat)
    B2_hat_minus = generate_minus(B2_hat)
    A_u_hat = B2_hat_plus@np.transpose(B2_hat_minus)+B2_hat_minus@np.transpose(B2_hat_plus)
    return A_u_hat

def generate_A_s_hat(B1,B2):
    deg = np.sum(np.abs(B1),axis =1)
    d1 = np.abs(B1).T@deg
    d2 = 3*np.sum(np.abs(B2),axis = 1)
    d = d1+d2
    dd = np.hstack((d,d))
    A_s_hat = np.diag(dd)
    return A_s_hat

def generate_A_rw_hat(A_s_hat, A_l_hat,A_u_hat):
    return A_s_hat+A_l_hat+A_u_hat

#comparison metrics

def community_quality(node_label,edge_label,G):
    base = 0
    base_sim = 0
    enrich = 0
    enrich_sim= 0
    for index_n, n in enumerate(G.nodes):
        for index_m, m in enumerate(G.nodes):
            if index_n != index_m:
                base = base + 1
                if node_label[index_n] == node_label[index_m]:
                    base_sim = base_sim + 1
         
    overlap_comm = defaultdict(list)
    for index_n, n in enumerate(G.nodes):
        for index,(u,v) in enumerate(G.edges):
            if n == u or n == v:
                overlap_comm[n].append(edge_label[index])
    for index_n, n in enumerate(G.nodes):
        for index_m, m in enumerate(G.nodes):
            if len(set(overlap_comm[n]).intersection(set(overlap_comm[m])))> 0 and index_n != index_m:
                enrich = enrich + 1
                if node_label[index_n] == node_label[index_m]:
                    enrich_sim = enrich_sim + 1
    return (enrich_sim/enrich)/(base_sim/base)


def community_coverage(G,edge_class):
    count = Counter(edge_class)
    comm = [k for k in count if count[k] > 1]
    raw_cover = 0
    for n in list(G.nodes):
        for index, (u,v) in enumerate(G.edges):
            if n == u or n == v:
                if edge_class[index] in comm:
                    raw_cover = raw_cover + 1
                    break
    comm_cov = raw_cover/len(G.nodes)
    return comm_cov

def overlap_coverage(G,edge_class):
    count = Counter(edge_class)
    comm = [k for k in count if count[k] > 1]
    node_comm = np.zeros(len(G))
    for index_n, n in enumerate(G.nodes):
        j_comm = set()
        for index, (u,v) in enumerate(G.edges):
            if n == u or n == v:
                if edge_class[index] in comm:
                    j_comm.add(edge_class[index])
        node_comm[index_n] = len(j_comm)
    overlap_cov = np.average(node_comm)
    return overlap_cov

#old implementation of overlap quality: use degree of each node as proxy for number of communities that each node participate in
def overlap_quality(G,edge_class):
    count = Counter(edge_class)
    comm = [k for k in count if count[k] > 1]
    degrees = [val for (node, val) in G.degree()]
    node_comm = np.zeros(len(G))
    for index_n, n in enumerate(G.nodes):
        j_comm = set()
        for index, (u,v) in enumerate(G.edges):
            if n == u or n == v:
                if edge_class[index] in comm:
                    j_comm.add(edge_class[index])
        node_comm[index_n] = len(j_comm)
    return metrics.mutual_info_score(degrees,node_comm)

#current implementation of overlap quality: requires metadata
def overlap_quality_new(G,ground_truth,edge_class):
    count = Counter(edge_class)
    comm = [k for k in count if count[k] > 1]
    node_comm = np.zeros(len(G))
    for index_n, n in enumerate(G.nodes):
        j_comm = set()
        for index, (u,v) in enumerate(G.edges):
            if n == u or n == v:
                if edge_class[index] in comm:
                    j_comm.add(edge_class[index])
        node_comm[index_n] = len(j_comm)
    return metrics.mutual_info_score(ground_truth,node_comm)


