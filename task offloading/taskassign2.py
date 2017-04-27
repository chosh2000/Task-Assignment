'''
This is a Task assignment Algorithm



'''

import sys
import random
import math

class Vertex:
    def __init__(self, node):
        # for all the graphs
        self.id = node
        self.adjacent = {}
        self.distance = sys.maxint  # Set distance to infinity for all nodes
        self.visited = False        # Mark all nodes unvisited        
        self.previous = None        # Predecessor

        # for task graph
        self.task_origin = object   # processor that initiates the application(root task)


        # for processor graph
        self.processorType = ''     # desk, rpi, jetson
        self.TA = 0                 #turnaround time\
        
        # for treenode n
        self.current_processor = object
        self.current_task = object
        self.K = {}                 #partial assignment  e.g. K={t1:p5, t2:p6, ...}
        self.UT = {}                #set of unassigned task
        self.f_score = 0


    def add_neighbor(self, neighbor, weight):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_visited(self):
        self.visited = True

    def update_TA(self, turnaound):
        self.TA += turnaround

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    # def set_distance(self, dist):
    #     self.distance = dist

    # def get_distance(self):
    #     return self.distance

    # def set_previous(self, prev):
    #     self.previous = prev

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0
        self.completeAssignment = {}        # This is for tree graph

        # for task graph
        self.task_origin = object           # processor that initiates the application(root task)


    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    #def add_edge(self, frm, to, cost = 0):
    def add_edge(self, frm, to, cost):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous


def generate_processor_link_matrix(num_processor_node, prob):
    '''

    output:

       | p1 | p2 | p3 | p4 |
        --------------------
     p1  b11  b12  b13  b14

     p2  b21  b22  b23  b24

     p3  b31  b32  b33  b34

     p4  b41  b42  b43  b44


     ex) b12 = link between processor 1,2

    link capacities were based on 
        3G: 1.6 Mbps
        4G:  12 Mbps
        Lan: 25 Mbps
    probability = 0.08
    '''


    m = [[0 for row in range(0,num_processor_node)] for col in range(0,num_processor_node)]
    probability = prob
    g3 = 2
    g4 = 12
    lan = 25
    capacities = [g3, g4, lan]
    f = open('processor_link_matrix', 'w')

    # Create multi list of link capacities
    for i in range (0, num_processor_node):
        for j in range(i+1, num_processor_node):
            index = random.randint(0,2)
            if random.random() < probability:
                m[i][j] = capacities[index]
                m[j][i] = m[i][j]

    # Write this to a text file
    for i in range (0, num_processor_node):
        for j in range(0, num_processor_node):
            f.write(str(m[i][j]))
            f.write(" ")
        f.write('\n')


def generate_processor_type_matrix(num_processor_node):
    '''
        probabilities:
            desktop = 0.2
            jetson = 0.4
            rpi = 0.4
    '''
    # Create matrix of 0
    f = open('processor_type_matrix', 'w')
    m = [0 for elem in range(0,num_processor_node)]
    # Assign numbers
    # desk = 0, rpi = 1, jetson = 2
    for i in range (0, num_processor_node):
        ran = random.random()

        if ran < 0.2:       #desktop
            m[i] = 0
        elif ran < 0.6:     #rpi
            m[i] = 1
        else:
            m[i] = 2     #jetson
        f.write(str(m[i]))
        f.write(" ")


def matrix2graph():
    ''' Return a processor graph from a randomly generated square matrix
        input: square matrices (processor type + link)
        output: processor graph
    '''
    #Create a graph
    G = Graph()
    i = 0
    j = 0
    k = 0

    # Processor type matrix
    ptype =  open("processor_type_matrix", "r")

    # Add vertices, each with id = 0,1,2,3,4,...
    for line in ptype:
        for elem in line.split():
            G.add_vertex(i)
            if int(elem)==0:
                G.vert_dict[i].processorType = 'desk'
            if int(elem)==1:
                G.vert_dict[i].processorType = 'rpi'
            if int(elem)==2:
                G.vert_dict[i].processorType = 'jetson'
            i = i+1


    # Add edges to the graph
    linkfile = open("processor_link_matrix", "r")
    linelist = linkfile.readlines()
    for y in linelist:
        k = 0
        for x in y.split():
            if int(x) > 0:
                G.add_edge(j,k,int(x))      #j(from), k(to), x(weight)
            k+=1
        j+=1



    return G

    
def create_task_graph():

    #Create a graph
    G_task = Graph()


    #Setting original processor for the task graph
    random_processor_id = random.choice(G_proc.vert_dict.keys())  # gives id (int) 
    G_task.task_origin = G_proc.vert_dict[random_processor_id]
    # print G_task.task_origin

    # Create task vertices (SAMPLE)
    for i in range(0,7):
        G_task.add_vertex(i)

    # Create task link, in KB (node1, node2, link)
    G_task.add_edge(0, 1, 6)
    G_task.add_edge(0, 2, 4)
    G_task.add_edge(1, 3, 2)
    G_task.add_edge(1, 4, 3)
    G_task.add_edge(2, 5, 1)
    G_task.add_edge(2, 6, 2)
    # G_task.vert_dict[0].add_neighbor(G_task.vert_dict[1], 60)
    # G_task.vert_dict[0].add_neighbor(G_task.vert_dict[2], 40)
    # G_task.vert_dict[1].add_neighbor(G_task.vert_dict[3], 20)
    # G_task.vert_dict[1].add_neighbor(G_task.vert_dict[4], 30)
    # G_task.vert_dict[2].add_neighbor(G_task.vert_dict[5], 10)
    # G_task.vert_dict[2].add_neighbor(G_task.vert_dict[6], 20)

    return G_task


def getExecTime(task,processor,resol):
    '''
    ax^2 + bx + c
    Execution time is assumed to be in quadratic function with resolution as the independent variable x.

    a,b,c may vary for each processor and task. These coefficients are extrapolated from manually measured values. 

    ex) for 5 tasks and 3 different processors, need 15 equations
    '''

    if task.id == 0:
        if processor.processorType == "desk":
            a = 0
            b = 0
            c = 0.0001
        if processor.processorType == "rpi":
            a = 0
            b = 0
            c = 0.005
        if processor.processorType == "jetson":
            a = 0
            b = 0
            c = 0.001
    if task.id == 1:
        if processor.processorType == 'desk':
            a = 0
            b = 0
            c = 0.225
        if processor.processorType == 'rpi':
            a = 0
            b = 0
            c = 2.6
        if processor.processorType == 'jetson':
            a = 0
            b = 0
            c = 1.3
    if task.id == 2:
        if processor.processorType == 'desk':
            a = 0
            b = 0
            c = 0.055
        if processor.processorType == 'rpi':
            a = 0
            b = 0
            c = 0.55
        if processor.processorType == 'jetson':
            a = 0
            b = 0
            c = 0.25
    if task.id == 3:
        if processor.processorType == 'desk':
            a = 0
            b = 0
            c = 0.1
        if processor.processorType == 'rpi':
            a = 0
            b = 0
            c = 1
        if processor.processorType == 'jetson':
            a = 0
            b = 0
            c = 0.5
    if task.id == 4:
        if processor.processorType == 'desk':
            a = 0
            b = 0
            c = 0.2
        if processor.processorType == 'rpi':
            a = 0
            b = 0
            c = 2
        if processor.processorType == 'jetson':
            a = 0
            b = 0
            c = 1
    if task.id == 5:
        if processor.processorType == 'desk':
            a = 0
            b = 0
            c = 0.07
        if processor.processorType == 'rpi':
            a = 0
            b = 0
            c = 0.7
        if processor.processorType == 'jetson':
            a = 0
            b = 0
            c = 0.35
    if task.id == 6:
        if processor.processorType == 'desk':
            a = 0
            b = 0
            c = 0.12
        if processor.processorType == 'rpi':
            a = 0
            b = 0
            c = 1.2
        if processor.processorType == 'jetson':
            a = 0
            b = 0
            c = 0.6
    return (a*math.pow(resol,2) + b*resol + c)


def getCommTime(vt1, vt2, vp1, vp2, resol):

    return True


def calc_e(t, q):
    '''
    Calculate the execution cost of assigning task t to processor q
    task(t) must come before processor(q)
    R = {} this must be global for all functions to see
    '''
    resol = R[t.id]          # find the resolution for task t in the set of resolutions R={t1:200, t2:500, ...}
    # print("resol = %d"% resol)
    # print("t=")
    # print(t)
    # print("q=")
    # print(q.processorType)
    x = getExecTime(t,q,resol)
    return x

def calc_c(i,q,j,r):
    '''
    Calculate the communication cost of task (i,j) between processor (q,r)
    =0 when no connection between (i,j)
    =sys.maxint when no connection (q,r)
    '''

    if i in j.adjacent:     # task i adjacent to j
        if q in r.adjacent: # proc q adjacent to r
            return  (i.get_weight(j)*0.008/q.get_weight(r))     # 1 KB = 0.008 Megabit
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
    searchnode = 0                                #id of the search node
    final_set = {}                   # set of nodes already evaluated
    open_set = {}                     # set of tentative nodes to be evaluated  {0:search_vertex1, 1:search_vertex2, ...}

    #Create a search graph
    astarGraph = Graph()

    #Root task is assigned at the original processor
    astarGraph.add_vertex(searchnode)                                               #initial search node
    search_Vertex = astarGraph.vert_dict[searchnode]                                          
    search_Vertex.current_processor = taskGraph.task_origin    
    search_Vertex.current_task = taskGraph.vert_dict[0]
    search_Vertex.K[search_Vertex.current_task] = search_Vertex.current_processor                 #task t0 is assigned to original processor

    #update Unassigned Task
    search_Vertex.UT = taskGraph.vert_dict    # Fixing the Unassigned Task set
    for assignedtask in search_Vertex.K:                    
        if assignedtask.id in search_Vertex.UT:
            del search_Vertex.UT[assignedtask.id]

    #f_score for initial node
    search_Vertex.f_score = calc_f(search_Vertex)

    #add this search node to the open set
    open_set[searchnode] = search_Vertex



    ''' the expansion of original node begins :D '''
    #select the node with least f_score and remove it from the open list(nodes)
    while len(open_set)>0:
        n = None
        for node in open_set:
            if n is None:
                n = node
            elif open_set[node].f_score < open_set[n].f_score:
                n = node

        expandingnode = open_set[n]
        print("\n expanding node:")
        print(expandingnode)
        for ex in expandingnode.K:
            print("expandingnode.K:<%d,%d>"%(ex.id,expandingnode.K[ex].id))

        # print("open_set:")
        # for open_elem in open_set:
        #     print(open_elem)
        del open_set[n]

        #select the next task in the expanding node
        nexttask = expandingnode.UT[(expandingnode.current_task.id+1)]          #next task = unassigned task connected to task in expanding node 



        ''' find out where the nexttask is originating from '''
        for kk in nexttask.adjacent:
            if kk.id < nexttask.id:
                nexttask_originate_from = expandingnode.K[kk]   #nexttask_originate_from = processor


        ''' assign it to current processor'''
        searchnode += 1
        print("=============================")
        print("\nsearchnode: %d"%searchnode)
        astarGraph.add_vertex(searchnode)
        search_Vertex = astarGraph.vert_dict[searchnode]
        search_Vertex.current_processor = nexttask_originate_from
        search_Vertex.current_task = nexttask
        
        #copy K from expandingnode
        for p in expandingnode.K:
            search_Vertex.K[p] = expandingnode.K[p]
        #add new element to the K
        search_Vertex.K[search_Vertex.current_task] = search_Vertex.current_processor

        for check_task in search_Vertex.K:
            print("K:<%d,%d>"%(check_task.id, search_Vertex.K[check_task].id))


        #update Unassigned Task
        for uu in expandingnode.UT:
            search_Vertex.UT[uu] = expandingnode.UT[uu]
        del search_Vertex.UT[nexttask.id]

        for cc in search_Vertex.UT:
            print("unassigned tasks so far: %d"%cc)

        #f_score
        search_Vertex.f_score = calc_f(search_Vertex)

        #add this search node to the open set
        open_set[searchnode] = search_Vertex

        # if the search vertex is a leaf node, add it to the final set
        if len(expandingnode.UT) == 1:
            final_set[searchnode] = search_Vertex



        '''assign it to adjacent processors'''
        for adjacentProcessor in nexttask_originate_from.adjacent:       #get_connections gives the processor vertex
            searchnode += 1
            print("=============================")
            print("Searchnode: %d"%searchnode)
            print("Expandingnode's processor %d"%expandingnode.current_processor.id)
            astarGraph.add_vertex(searchnode)
            search_Vertex = astarGraph.vert_dict[searchnode]
            search_Vertex.current_processor = adjacentProcessor
            search_Vertex.current_task = nexttask

            #copying K from expanding node
            for p in expandingnode.K:
                search_Vertex.K[p] = expandingnode.K[p]

            #add new element to the K
            search_Vertex.K[search_Vertex.current_task] = search_Vertex.current_processor

            # search_Vertex.K = expandingnode.K
            for p in expandingnode.K:
                print("Expandingnode's assignment:<%d,%d>"%(p.id, expandingnode.K[p].id))



            for check_task in search_Vertex.K:
                print("K:<%d,%d>"%(check_task.id, search_Vertex.K[check_task].id))


            #update Unassigned Task
            for uu in expandingnode.UT:
                search_Vertex.UT[uu] = expandingnode.UT[uu]
            del search_Vertex.UT[nexttask.id]

            for cc in search_Vertex.UT:
                print("unassigned tasks so far: %d"%cc)

            #f_score
            search_Vertex.f_score = calc_f(search_Vertex)

            #add this search node to the open set
            open_set[searchnode] = search_Vertex
            # print("^^^^^^^^open set^^^^^^^^^")
            # for u in open_set:
            #     print u
            # if the search vertex is a leaf node, add it to the final set
            if len(expandingnode.UT) ==1:
                final_set[searchnode] = search_Vertex

        # if expanded node was the last expanding node, find the leaf node with least f score
        if len(expandingnode.UT) == 1:
            print("***********************************")
            final_solution = None
            for leafnode in final_set:
                if final_solution is None:
                    final_solution = leafnode
                elif final_set[leafnode].f_score < final_set[final_solution].f_score:
                    final_solution = leafnode
            break           #break out of the open set while loop



    '''Finally, get the complete assignment '''
    print("final assignment:")
    for final_task in final_set[final_solution].K:
        final_processor = final_set[final_solution].K[final_task]
        print("task id = %d, processor id = %d\n" % (final_task.id, final_set[final_solution].K[final_task].id))
        '''Update the turnaround time for this assignment '''
        final_processor.TA += calc_e(final_task,final_processor)   #add execution cost for each processor
        # print("final_processor.TA: %d"%final_processor.TA)

        for final_adjacent_task in final_task.adjacent:            #add communication cost for each processor
            if final_adjacent_task.id > final_task.id:
                if final_processor.id != final_set[final_solution].K[final_adjacent_task].id:   # if adjacent task is assigned to different processor
                    final_processor.TA += calc_c(final_task, final_processor, final_adjacent_task, final_set[final_solution].K[final_adjacent_task])
                    final_set[final_solution].K[final_adjacent_task].TA += calc_c(final_task, final_processor, final_adjacent_task, final_set[final_solution].K[final_adjacent_task])




    # return (astarGraph.completeAssignment, processorGraph)  #processorGraph is updated in case altering value here doesn't change the processor graph
 
 
def without_taskassign(gt, gp):
    orig_proc_id = gt.task_origin.id
    orig_proc = gp.vert_dict[orig_proc_id] 
    for tid in gt.vert_dict:
        orig_proc.TA += calc_e(gt.vert_dict[tid], orig_proc)


if __name__ == '__main__':

    ''' prestep) uncomment below to get processor matrices '''
    # Creating processor Graph matrix
    num_processor_node = 70
    processor_matrix = generate_processor_link_matrix(num_processor_node, 0.40)  # 40% probability
    generate_processor_type_matrix(num_processor_node)



    ''' step1) Create processor graph from'''
    #             - processor type matrix
    #             - processor link matrix
    G_proc = matrix2graph()
    G_proc2 = matrix2graph()


    ''' step2) set modular resolution    ex)R = {0:600, 1:540, ...}'''
    R = {}
    R[0] = 640
    R[1] = 640
    R[2] = 640
    R[3] = R[1]
    R[4] = R[1]
    R[5] = R[2]
    R[6] = R[2]

    ''' step3) Create task graph based on step 2 '''
    ''' THIS IS A SAMPLE, task link must depend on resolution'''
    for counter in range(0,20):
        G_task = create_task_graph()


        print 'Graph data:'
        for v in G_task:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                print '( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w))

        
        ''' step 4) Run A* '''
        # input: task graph + processor graph
        # output: Assignment K + least throughput(max TA) + Average latency(sum of max TA for each application / #application)

        astar(G_task, G_proc)
        without_taskassign(G_task, G_proc2)


    total_TA_astar = 0
    total_TA_without_astar = 0
    max_TA_astar = 0
    max_TA_without_astar = 0

    for procs in G_proc.vert_dict:
        print("processor id:%d, TA:%f"%(procs, G_proc.vert_dict[procs].TA))
        total_TA_astar += G_proc.vert_dict[procs].TA
        if G_proc.vert_dict[procs].TA > max_TA_astar:
            max_TA_astar = G_proc.vert_dict[procs].TA

    print("#############without task assignment##################")
    

    for procs in G_proc2.vert_dict:
        print("processor id:%d, TA:%f"%(procs, G_proc2.vert_dict[procs].TA))
        total_TA_without_astar += G_proc2.vert_dict[procs].TA
        if G_proc2.vert_dict[procs].TA > max_TA_without_astar:
            max_TA_without_astar = G_proc2.vert_dict[procs].TA


    print("total_TA_astar: %f"%total_TA_astar)
    print("max_TA_astar: %f"%max_TA_astar)
    print("average_TA_astar: %f"%(total_TA_astar/70))

    print("total_TA_without_astar: %f"%total_TA_without_astar)
    print("max_TA_without_astar: %f"%max_TA_without_astar)
    print("average_TA_without_Astar: %f"%(total_TA_without_astar/70))
    # dijkstra(g, g.get_vertex(1), g.get_vertex('e')) 

    # target = g.get_vertex('e')
    # path = [target.get_id()]
    # shortest(target, path)
    # print 'The shortest path : %s' %(path[::-1])