import sys



def calc_e(t, q):
    '''
    Calculate the execution cost of assigning task t to processor q
    task(t) must come before processor(q)
    R = {} this must be global for all functions to see
    '''
    resol = R[t]          # find the resolution for task t in the set of resolutions R={t1:200, t2:500, ...}
    x = getExecTime(t,q,resol)
    return x

def calc_c(i,q,j,r):
    '''
    Calculate the communication cost of task (i,j) between processor (q,r)
    =0 when no connection between (i,j)
    =sys.maxint when no connection (q,r)
    '''
    if i.get_weight(j):      # task i adjacent to j
        if q.get_weight(r):  # proc q adjacent to r
            return  i.get_weight(j) / q.get_weight(r)
    else:
        return 0


def calc_f(node):
    '''
    Calculate the cost function: f(n) = g(n) + h(n)
    input:
            current node
                1) current processor q
                2) current task i
                3) Partial task assignment node.K = {task:processor, ...} <-- includes {i:q}
    output:
            f(n)

    g(n) is the Turnaround time of processor q == sum of execution time of tasks assigned in q + communication time
    h(n) is the heuristic cost
    '''

    q = node.current_processor
    i = node.current_task

    gn = q.TA                                # initially begins with Turnaround time of the processor
    for t in node.K:                         # for tasks assigned in the partial assignemnt
        if node.K[t] == q:                   # if assigned to current processor q
            gn += calc_e(t,q)                # add the execution cost
        elif node.K[t] in q.adjacent:        # if task is not assigned in q but neighbors of q
            gn += calc_c(i,q,t,node.K[t])    # adding communication cost

    hn = 0

    for t in node.UT:                            # for unassigned tasks t
        if t in i.adjacent:                      # if t is adjacent to i
            minAC = sys.maxint
            for r in q.adjacent:                 # for all neighboring processors
                AC = 0
                for tasks in node.K:                  #
                    if node.K[tasks] == q:            # all tasks in q that is communicating with unassigned adjacent t
                        AC += calc_c(tasks, q, t, r)
                if minAC > AC:
                    minAC = AC                   # find minimum communication time among neighboring processors
            hn += min(minAC, calc_e(t,q))        # h(n) = sum of all adjacent tasks -> minAC
    
    fn = gn + hn
    return fn



def astar(taskGraph, processorGraph):
    '''
    input: task graph and processor graph
    output: Complete assignment K, and update turnaround time(TA) for the processors accordingly

    Initially create a new graph astarGraph, and initial node

    '''
    n = 1                                #id of the search node
    closed_set = set()                   # set of nodes already evaluated
    open_set = set()                     # set of tentative nodes to be evaluated
    open_set.add(init_node)
    
    astarGraph = Graph()
    astarGraph.add_vertex(0)            #initial node


    #######     first level of the search tree     #########
    for task in taskGraph.vert_dict:
        for proc in processorGraph.vert_dict:         # assigning all the task in all the processor
            astarGraph.add_vertex(n)                  # create search node starting with n = 1
            
            search_Vertex = astarGraph.vert_dict[n]   # vertex of search tree
            search_Vertex.current_processor = proc    # set the current processor
            search_Vertex.current_task = task         # set the current task
            search_Vertex.K[task] = proc              # add partial assignment to K

            search_Vertex.UT = taskGraph.vert_dict    # Fixing the Unassigned Task set
            for assignedtask in K:                    
                if assignedtask in search_Vertex.UT:
                    del search_Vertex.UT[assignedtask]


            search_Vertex.f_score = calc_f(search_Vertex)
            open_set.add(search_Vertex)               # finally, add the searce tree vertex to the open list
            n += 1                                     # need to increment the search node id

    open_set.remove(init_node)



    ######     Recursive level of the search tree     #########
    #
    #
    #
    #
    #
    #
    #
    #
    ############################################################



    while open_set:
        #select the node with least f_score and remove it from the open list(nodes)
        n = None
        for node in open_set:
            if n is None:
                n = node
            elif f_score[node] < f_score[n]:
                n = node

        '''
        expand the children of x (each node will have 1)partial assignment{1:a, 2:b})
        find out a processor k with most load
        calculate f_score
        put them in the open list(nodes)

        '''
        open_set.remove(x)

        #When the algorithm reaches the leaf node(complete assignment), stops and return the map of navigated nodes
        #For us, we need to expand all of the leaf nodes and select the one with least f_score
        '''
        if len(n.K) == taskGraph.num_vertices -1:
            expand all the leafs
            select the least f_score
            return visited
        '''
        if x == goal_node:
            return visited
        closed_set.add(x)   #if 



        for y in graph.edges[x]:
            if y in closed_set:
                continue
            tentative_g_score = g_score[x] + graph.distances[(x, y)]
 
            flag = False
            if y not in open_set or tentative_g_score < g_score[y]:
                open_set.add(y)
                flag = True
 
            if flag:
                visited[y] = x
 
                g_score[y] = tentative_g_score
                h_score[y] = h(y, goal_node)
                f_score[y] = g_score[y] + h_score[y]
 
    return (astarGraph.completeAssignment, processorGraph)  #processorGraph is updated in case altering value here doesn't change the processor graph
 
 
 




if __name__ == '__main__':
    import math
    sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
    g = Graph()
    g.add_node((0, 0))
    g.add_node((1, 1))
    g.add_node((1, 0))
    g.add_node((0, 1))
    g.add_node((2, 2))
 
    g.add_edge((0, 0), (1, 1), 1.5)
    g.add_edge((0, 0), (0, 1), 1.2)
    g.add_edge((0, 0), (1, 0), 1)
    g.add_edge((1, 0), (2, 2), 2)
    g.add_edge((0, 1), (2, 2), 2)
    g.add_edge((1, 1), (2, 2), 1.5)
 
 
    g.distances[((0, 0), (1, 1))] = 2
    g.distances[((1, 1), (0, 0))] = 2
 
    g.distances[((0, 0), (1, 0))] = 1.3
    g.distances[((1, 0), (0, 0))] = 1.3