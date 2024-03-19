# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue
    
def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    row = maze.size.y
    col = maze.size.x
    start = maze.start
    current = maze.start
    waypoints = maze.waypoints
    path = []
    frontier = []
    frontier.append(start)
    visited = []
    visited.append((start,start))
    explored = set()
    explored.add(start)
    
    while current != waypoints[0]:
        frontier.pop(0)
        neighbors = maze.neighbors(current[0], current[1])
        for neighbor in neighbors:
            if(neighbor not in explored and maze.navigable(neighbor[0], neighbor[1])):
                visited.append((current,neighbor))
                frontier.append(neighbor)
                explored.add(neighbor)
        current = frontier[0]
    
    current = waypoints[0]
    path.append(current)
    while current != start:
        for visit in visited:
            if(visit[1] == current):
                path.append(visit[0])
                current = visit[0]
    path.reverse()

    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single

    frontier = queue.PriorityQueue()
    start = maze.start
    end = maze.waypoints[0]
    open_list = set([start])
    closed_list = set([])
    g = {}
    g[start] = 0
    parents = {}
    parents[start] = start
    frontier.put(tuple([g[start] + h(start, end), start]))

    while len(open_list) > 0:
        temp = frontier.get()
        current = temp[1]

        if current == end:
            reconst_path = []

            while parents[current] != current:
                reconst_path.append(current)
                current = parents[current]

            reconst_path.append(start)

            reconst_path.reverse()
            return reconst_path

            # for all neighbors of the current node do
        neighbors = maze.neighbors(current[0], current[1])
        for neighbor in neighbors:
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
            if neighbor not in open_list and neighbor not in closed_list and maze.navigable(neighbor[0], neighbor[1]):
                open_list.add(neighbor)
                parents[neighbor] = current
                g[neighbor] = g[current] + 1
                frontier.put(tuple([g[neighbor] + h(neighbor, end), neighbor]))

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
            else:
                if g[neighbor] > g[current] + 1:
                    g[neighbor] = g[current] + 1;
                    parents[neighbor] = current;

                    if neighbor in closed_list:
                        closed_list.remove(neighbor)
                        open_list.add(neighbor)
                        frontier.put(tuple([g[neighbor] + h(neighbor, end), neighbor]))

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
        open_list.remove(current)
        closed_list.add(current)

def h(n, end):
    return(abs(n[1] - end[1]) + abs(n[0] - end[0]))


# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
