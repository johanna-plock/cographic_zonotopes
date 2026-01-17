import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

#+++ takes a file in the instance format described on https://timpasslib.aalto.fi/pesplib.html and converts it to a tuple (G,T,l,u,w,S) +++
#+++ (PESP with additional spanning tree for cycle matrix) +++
#+++ arguments: file path for a text file +++

def read_file(file_path):
    import os
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)

    G = nx.DiGraph()
    l = [] 
    u = [] 
    w = [] 
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('#') or not line.strip():
                continue
                
            parts = line.strip().split(';')
            if len(parts) < 6:
                continue
            parts = [int(p.strip()) for p in parts]
            
            _, start_node, end_node, lower_bound, upper_bound, weight = parts
            
            G.add_node(start_node)
            G.add_node(end_node)
            G.add_edge(start_node, end_node, id=len(l), sweight=upper_bound-lower_bound)
            
            l.append(lower_bound)
            u.append(upper_bound) 
            w.append(weight)
    T = 60

    S_undirected = nx.minimum_spanning_tree(G.to_undirected(), weight='sweight')
    S = nx.DiGraph()
    S.add_nodes_from(G.nodes())
    
    for (a,b) in S_undirected.edges():
        if G.has_edge(a,b):
            S.add_edge(a,b)
        elif G.has_edge(b,a):
            S.add_edge(b,a)

    return G,T,l,u,w,S

#+++ takes a file in the solution format described on https://timpasslib.aalto.fi/pesplib.html and put out the given solution x as a list +++
#+++ arguments: file path for a text file +++

def starting_point(file_path):
    import os
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)
    x = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('#') or not line.strip():
                continue
                
            parts = line.strip().split(';')
            if len(parts) < 2:
                continue
                
            parts = [int(p.strip()) for p in parts]
            start_point = parts[1]
            x.append(start_point)

    return x

#+++ computing the cycle matrix for a cographic zonotope +++
#+++ arguments: DiGraph G, spanning tree S+++

def cycle_matrix(G,S):
    if S == None:
        S = nx.random_spanning_tree(G, weight=None)
    m = G.number_of_edges()
    mu = len(G.edges) - len(S.edges)
    row = 0
    Gamma = [[0] * m for _ in range(mu)]

    edges_sorted = sorted(G.edges(data='id'), key=lambda x: x[2])
    for (a,b, _) in edges_sorted:
        if (a,b) not in S.edges:
            S1 = S.copy()
            S1.add_edge(a,b) 
            C = nx.find_cycle(S1, orientation = "ignore")

            if (a,b,"forward") in C:
                sign = 1
            else:
                sign = -1

            for (u,v,orient) in C:
                if orient == "forward":
                    i = G.edges[u, v]["id"]
                    Gamma[row][i] = sign
                else:
                    i = G.edges[u,v]["id"]
                    Gamma[row][i] = -sign
            row += 1
    return Gamma

#+++ computing the scaled cycle matrix and translation vector for a cographic zonotope +++
#+++ arguments: DiGraph G, lower/upper bound vectors l and u, period time T +++

def scaled_cycle_matrix(G,T,l,u,S):
    Gamma = cycle_matrix(G,S)
    translation = ((np.array(Gamma)@np.array(l))/T).tolist()
    d = (np.array(u) - np.array(l))/T
    Gammad = Gamma.copy()
    for i in range(len(Gammad)):
        for j in range(len(Gammad[0])):
            Gammad[i][j] = float(Gammad[i][j] * d[j])
    return Gammad, translation

#+++ computing the covector for an arborescence in a graph G +++
#+++ arguments: Digraph G, Arborescence A (in general not all edges in the same directions as in G) +++

def covector(G, A):
    C = [0] * len(G.edges)
    for (a, b) in G.edges:
        if (a, b) in A.edges:
            C[G.edges[a, b]['id']] = '+'
        elif (b, a) in A.edges:
            C[G.edges[a, b]['id']] = '-'
    return C

#+++ computing all neighboring tiles of a given tile in a fine zonotopal tiling +++
#+++ arguments: DiGraph G, Arborescence A corresponding to the tile +++

def finding_neighbors(G,A):
    Neighbors = []
    neighboring_covectors = []
    root = next(x for x in A.nodes() if A.in_degree(x)==0)

    for e in G.edges:
        (u,v) = e
        if (u,v) in A.edges or (v,u) in A.edges:
            continue

        A1 = A.copy()
        A1.add_edge(u,v)
        C = nx.find_cycle(A1, orientation = "ignore")
        Cedges = []
        for (a,b,_) in C:
            Cedges.append((a,b))

        for f in Cedges:
            if f == e:
                continue
            N = A1.copy()
            N.remove_edge(*f)
            if nx.is_arborescence(N):
                if all(nx.has_path(N, root, n) for n in N.nodes if n != root): 
                #diese Zeile kann nicht direkt weg, weil garantiert werden muss dass die arboreszenz immer noch root als wurzel hat
                    Neighbors.append(N)
                    neighboring_covectors.append(covector(G, N))
            else: 
                N.remove_edge(*e)
                N.add_edge(v,u)
                if all(nx.has_path(N, root, n) for n in N.nodes if n != root):
                    Neighbors.append(N)
                    neighboring_covectors.append(covector(G,N))
    return neighboring_covectors, Neighbors

#+++ computing an integer point lying in a given tile of a fine zonotopal tiling, or findng that there is no such point +++
#+++ arguments: DiGraph G, Arborescence A corresponding to the tile, lower/upper bound vectors l and u, period time T +++

def tile_to_point(G,A,T,l,u,S):
    node_index = {node: idx for idx, node in enumerate(G.nodes)}
    root = next(x for x in A.nodes() if A.in_degree(x)==0)
    pi = [0]*len(G.nodes)
    x = np.array([0]*len(G.edges))
    for (a,b) in A.edges:
        if (a,b) in G.edges:
            i = G.edges[a,b]["id"]
            x[i] = u[i]
        else:
            i = G.edges[b,a]["id"]
            x[i] = l[i]
    
    current_layer = list(A.successors(root))
    while current_layer != []:
        next_layer = []
        for v in current_layer:
            i = node_index[v]
            predv = next(A.predecessors(v))
            k = node_index[predv]
            if (predv,v) in G.edges:
                j = G.edges[predv,v]["id"]
                pi[i] = pi[k] + x[j]
            elif (v,predv) in G.edges:
                j = G.edges[v,predv]["id"]
                pi[i] = pi[k] - x[j]
            next_layer.extend(A.successors(v))
        current_layer = next_layer

    for (a,b) in G.edges:
        if (a, b) not in A.edges and (b, a) not in A.edges:
            i = G.edges[a, b]['id']
            j = node_index[a]
            k = node_index[b]
            x[i] = pi[k] - pi[j]

    for i in range(len(x)):
        x[i] = (x[i]-l[i]) % T + l[i]

    Gamma = np.array(cycle_matrix(G,S))
    if all(l[i] <= x[i] <= u[i] for i in range(len(x))):
        z = (Gamma@x)/T
        return z
    else: 
        return None
    
#+++ computing the tile of a fine zonotopal tiling that a given integer point lies in +++
#+++ arguments: DiGraph G, lower/upper bound vectors l and u, period time T, integer point z, root of all arborescences in the tiling+++ 

def point_to_tile(G,T,l,u,z,root,S):
    p = [0]*len(G.edges)
    i = 0
    edges_sorted = sorted(G.edges(data='id'), key=lambda x: x[2])
    for (a,b,id) in edges_sorted:
        if (a,b) not in S.edges:
            p[id] = z[i]
            i+=1

    AuxG = G.copy()
    for (a,b) in G.edges:
        id = G.edges[a,b]["id"]
        AuxG.edges[a,b]['weight'] = u[id] - p[id]*T
        AuxG.add_edge(b,a)
        AuxG.edges[b,a]["weight"] = -l[id] + p[id]*T
    
    shortestpaths = nx.single_source_bellman_ford_path(AuxG, source=root, weight= "weight")
    Arborescence = nx.DiGraph()
    Arborescence.add_nodes_from(G.nodes)
    for path in shortestpaths.values():
        for i in range(len(path)-1):
            Arborescence.add_edge(path[i],path[i+1])
    c = covector(G,Arborescence)
    return c, Arborescence

#+++ combining the previous functions to find a locally optimal solution x for edge labels +++
#+++ arguments: DiGraph G, lower/upper bound vectors l and u, period time T, integer point z, edge weights w +++

def optimal_feasible_sol(G,T,l,u,z,w,S):
    Gamma = np.array(cycle_matrix(G,S), dtype=float)
    w = np.array(w)
    z = np.array(z, dtype = float)

    M = gp.Model("mip1")
    M.Params.OutputFlag = 0
    vars = M.addMVar(len(G.edges), lb=l, ub=u, vtype=GRB.CONTINUOUS)
    
    M.setObjective(w @ vars, GRB.MINIMIZE) 

    M.addMConstr(Gamma, vars, "=", T*z)
    M.optimize()
    optBFS = vars.X
    
    return optBFS, M.ObjVal

#+++ combining the previous functions to find a locally optimal solution x for edge labels +++
#+++ arguments: DiGraph G, lower/upper bound vectors l and u, period time T, basic solution x, root of arborescences, edge weights w +++

def zonotopal_tiling_search(G,T,l,u,x,root,w,S):
    done = False
    Gamma = np.array(cycle_matrix(G,S))
    z = (Gamma@x)/T
    A = point_to_tile(G,T,l,u,z,root,S)[1]
    visited_tiles = [A.edges]
    plotlist = [A]
    currentopt, currentval = optimal_feasible_sol(G,T,l,u,z,w,S)

    max_iter = 200 
    iteration = 0

    while not done and iteration <= max_iter:
        print("Tile number", iteration+1)
        iteration += 1
        Neighbors = finding_neighbors(G,A)[1]
        integerpoints = []
        Neighborswithints = []
        for Arborescence in Neighbors:
            if Arborescence.edges not in visited_tiles:
                int = tile_to_point(G,Arborescence,T,l,u,S)
                if int is not None:
                    integerpoints.append(int)
                    Neighborswithints.append(Arborescence)

        if not integerpoints:
            return currentopt, currentval, plotlist

        optimalsol = []
        optimalval = []
        for int in integerpoints:
            y, value = optimal_feasible_sol(G,T,l,u,int,w,S)
            optimalsol.append(y)
            optimalval.append(value)
        
        i = optimalval.index(min(optimalval))
        xopt = optimalsol[i]
        if currentval < optimalval[i]:
            done = True
        else:
            A = Neighborswithints[i]
            visited_tiles.append(A.edges)
            plotlist.append(A)
            currentopt = xopt
            currentval = optimalval[i]
    return currentopt, currentval, plotlist

#+++ plotting the full zonotope and a list of tiles of it in matplotlib, showing also all integer points in red +++
#+++ arguments: DiGraph G, list of arborescences corresponding to tiles, lower/upper bound vectors l and u, period time T +++

def plot(G,List,T,l,u,S):
    Gamma, translation = scaled_cycle_matrix(G,T,l,u,S)
    Gamma = np.array(Gamma)

    unitvertices = np.array(list(itertools.product([0, 1], repeat=Gamma.shape[1])))
    zonotopevertices = unitvertices @ Gamma.T + np.array(translation)
    zonotope = ConvexHull(zonotopevertices)

    plt.figure(figsize=(6,6))
    plt.fill(zonotopevertices[zonotope.vertices,0], zonotopevertices[zonotope.vertices,1], alpha=0.3, edgecolor='k', linewidth=1.2)
    
    if List == []:
        pass
    else:
        for A in List:
            fixed = {}
            for i in range(len(covector(G,A))):
                if covector(G,A)[i] == "+":
                    fixed[i] = 1
                if covector(G,A)[i] == "-":
                    fixed[i] = 0
            tileunitvertices = np.array([p for p in unitvertices if all(p[i] == value for i, value in fixed.items())])
            tilevertices = tileunitvertices @ Gamma.T + np.array(translation)
            tile = ConvexHull(tilevertices)
            plt.fill(tilevertices[tile.vertices,0], tilevertices[tile.vertices,1], alpha=0.3, edgecolor='k', linewidth=1.2)

    plt.gca().set_aspect('equal', adjustable='box')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xs = np.arange(np.floor(xmin), np.ceil(xmax) + 1)
    ys = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
    X, Y = np.meshgrid(xs, ys)
    plt.scatter(X, Y, color='red', s=30)
    plt.show()
    return