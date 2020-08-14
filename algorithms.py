"""This module contains the Algorithms class which contains the
implementation of 4 graph traversal/pathfinding algorithms: Depth-first search,
Breath-first search, Dijkstra's algorithm, and A* pathfinding algorithm
"""
import random
import sys
import time
from collections import deque, namedtuple
from queue import PriorityQueue

import numpy as np
import pygame

Color = namedtuple("Color", ["r", "g", "b"])
WHITE = Color(240, 240, 240)
GREY = Color(150, 150, 150)
BLUE = Color(60, 210, 210)
GREEN = Color(90, 220, 40)
PURPLE = Color(235, 40, 235)
YELLOW = Color(235, 235, 0)
PEACH = Color(255, 218, 185)


def _timer(func):
    """Measure the time taken to execute func

    Args:
        func (def): function to have execution time measured
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, *kwargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Time elapsed for {func.__name__}: {time_elapsed}")

    return wrapper


class Algorithms:
    """Class containing various pathfinding algorithms
    """

    class Node:
        """Represents a node (box) of the maze, each node has its mode
        which determines color
        """

        color_code = {
            0: WHITE,
            1: BLUE,
            2: YELLOW,
            3: PURPLE,
            4: GREEN,
            5: GREY,
            6: PEACH,
            7: PEACH,
        }

        def __init__(self, row, col):
            self.row = row
            self.col = col

            # 0 for empty, 1 for wall, 2 for start, 3 for end, 4 for path
            # 5 for added to queue, 6 for visited
            self._mode = 0
            self.color = self.color_code[self._mode]

        @property
        def mode(self):
            """Getter of mode

            Returns:
                int: mode of this node, (0, 1...6)
            """
            return self._mode

        @mode.setter
        def mode(self, val):
            """Setter of mode

            Args:
                val (int): intended new mode for this node
            """
            # only white nodes can overwrite wall, start, or end nodes
            if self._mode in (0, 4, 5, 6, 7) or (self._mode in (1, 2, 3) and val == 0):
                self._mode = val

        def enforce_color(self):
            """Changes the color of the node according to the node's mode
            """
            self.color = self.color_code[self._mode]

        def get_pos(self):
            """Get the position of this node in tuple form

            Returns:
                tuple of int: [0] is row, [1] is col
            """
            return self.row, self.col

        def __repr__(self):
            return str(self.get_pos())

    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.nodes = np.zeros((num_rows, num_cols), dtype=Algorithms.Node)
        for i in range(num_rows):
            for j in range(num_cols):
                self.nodes[i, j] = Algorithms.Node(i, j)
        # 2D array to mark visited nodes
        self.visited = np.zeros((num_rows, num_cols), dtype=bool)
        # 2D array to mark each nodes previous node (parent)
        self.paths = np.zeros((num_rows, num_cols), dtype=Algorithms.Node)
        longest_dist = (self.num_rows * self.num_cols) * 10
        # 2D array to keep track of shortest path from start to each node
        self.shortest_dists = np.full((num_rows, num_cols), longest_dist, dtype=int)
        # can represent LIFO stack, FIFO queue, or Priority Queue
        self.queue = None
        self.queue2 = None
        # used to break ties in priority queue
        self.count = 0
        # used for bi-directional algorithms
        self.cur_node = None
        self.target_node = None

        # 1 for building walls, 2 for start position, 3 for end position
        self.mode = 1
        self.start = self.nodes[0, 0]
        self.end = self.nodes[-1, -1]
        self.start.mode = 2
        self.end.mode = 3

        self.solved = False

    def set_maze_state(self, opt):
        """Clears or resets maze

        Args:
            opt (str): "clear" to clear maze, "reset" to reset maze
        """
        longest_dist = (self.num_rows * self.num_cols) * 10

        # clearing all data structures used in pathfinding
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if opt == "clear" or (opt == "reset" and self.nodes[i, j].mode != 1):
                    self.nodes[i, j].mode = 0
                    self.visited[i, j] = False
                    self.paths[i, j] = 0
                    self.shortest_dists[i, j] = longest_dist

        self.count = 0
        self.start.mode = 2
        self.end.mode = 3
        self.solved = False

        self.cur_node = None
        self.target_node = None

    def follow_path(self, pointer, length):
        """Helper function for both print paths

        Args:
            pointer (Node): the current node to find parent of
            length (int): current path length

        Returns:
            (Node, int): parent of current node and updated path length
        """
        # while there is still a previous node, keep printing
        pointer_row, pointer_col = pointer.get_pos()
        pointer.mode = 4
        pointer = self.paths[pointer_row, pointer_col]

        self.update_gui()
        return pointer, length + 1

    def print_path(self):
        """Print the shortest path on screen
        """
        end_row, end_col = self.end.get_pos()
        pointer = self.paths[end_row, end_col]
        length = 2

        while pointer not in (0, self.start):
            pointer, length = self.follow_path(pointer, length)

        if pointer == self.start:
            print(f"Path found! Length of path is : {length}")
        else:
            print("Path not found...")

        self.solved = True

    def bi_print_path(self):
        """Print the shortest path on screen for bi-directional algorithms
        """
        if None in (self.cur_node, self.target_node):
            print("Path not found...")
        else:
            end_row0, end_col0 = self.cur_node.get_pos()
            self.cur_node.mode = 4
            end_row1, end_col1 = self.target_node.get_pos()
            self.target_node.mode = 4
            pointer0 = self.paths[end_row0, end_col0]
            pointer1 = self.paths[end_row1, end_col1]
            length = 4

            while pointer0 not in (0, self.start, self.end):
                pointer0, length = self.follow_path(pointer0, length)

            while pointer1 not in (0, self.start, self.end):
                pointer1, length = self.follow_path(pointer1, length)

            print(f"Path found! Length of path is : {length}")

        self.solved = True

    def get_neighbors(self, node, func, **kwargs):
        """Template of the get neighbors function for non-bi algorithms

        Args:
            node (Node): current node being evaluated
            func (def): unique function to be called within this template for
            each algorithm
        """
        row, col = node.get_pos()
        end_row, end_col, new_dist = None, None, None
        if kwargs.get("get_dist"):
            new_dist = self.shortest_dists[row, col] + 1
        if kwargs.get("end"):
            end_row, end_col = self.end.get_pos()

        if col != self.num_cols - 1:
            func(node, row, col + 1, new_dist, end_row, end_col)

        if row != 0:
            func(node, row - 1, col, new_dist, end_row, end_col)

        if col != 0:
            func(node, row, col - 1, new_dist, end_row, end_col)

        if row != self.num_rows - 1:
            func(node, row + 1, col, new_dist, end_row, end_col)

    def bi_get_neighbors(self, node, func, **kwargs):
        """Template of the get neighbors function for bi-directional algorithms

        Args:
            node (Node): current node being evaluated
            func (def): unique function to be called within this template for
            each algorithm
        """
        row, col = node.get_pos()
        end_row, end_col, new_dist = None, None, None
        if kwargs.get("get_dist"):
            new_dist = self.shortest_dists[row, col] + 1
        if kwargs.get("end"):
            end_row, end_col = self.end.get_pos()
        mode = kwargs.get("mode")
        end_row, end_col = self.end.get_pos() if mode == 6 else self.start.get_pos()

        if col != self.num_cols - 1:
            if func(node, row, col + 1, new_dist, end_row, end_col, mode):
                return True

        if row != 0:
            if func(node, row - 1, col, new_dist, end_row, end_col, mode):
                return True

        if col != 0:
            if func(node, row, col - 1, new_dist, end_row, end_col, mode):
                return True

        if row != self.num_rows - 1:
            if func(node, row + 1, col, new_dist, end_row, end_col, mode):
                return True

        return False

    @_timer
    def bfs(self):
        """Breadth-first-search algorithm
        """
        self.set_maze_state("reset")
        self.queue = deque()
        self.queue.append(self.start)
        row, col = self.start.get_pos()
        self.visited[row, col] = True

        while self.queue:
            for event in pygame.event.get():
                # while algorithm is running, need to make sure quit command is heeded
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # repeatedly get first element in queue and get neighbors
            cur_node = self.queue.popleft()
            if cur_node is self.end:
                # found end node
                break
            self.get_neighbors(cur_node, self.bfs_get_neighbors)
            cur_node.mode = 5

            self.update_gui()

        self.print_path()

    def bfs_get_neighbors(self, node, row, col, *_):
        """Helper function for breadth-first search to get all neighbors

        Args:
            node (Node): current node being evaluated
            row (int): row of neighbor node
            col (int): column of neighbor node
        """
        n_node = self.nodes[row, col]
        if n_node.mode != 1 and not self.visited[row, col]:
            # only add node if neighbor node is unvisited and isn't a wall
            n_node.mode = 6
            self.queue.append(n_node)
            self.visited[row, col] = True
            self.paths[row, col] = node

    @_timer
    def dfs(self):
        """Depth-first-search algorithm
        """
        self.set_maze_state("reset")
        self.queue = deque()
        self.queue.append(self.start)
        row, col = self.start.get_pos()
        self.visited[row, col] = True

        while self.queue:
            for event in pygame.event.get():
                # while algorithm is running, need to make sure quit command is heeded
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # repeatedly get last element in queue and get nieghbors
            cur_node = self.queue.pop()
            if cur_node is self.end:
                # found end node
                break
            self.get_neighbors(cur_node, self.dfs_get_neighbors)
            cur_node.mode = 5
            row, col = cur_node.get_pos()
            self.visited[row, col] = True

            self.update_gui()

        self.print_path()

    def dfs_get_neighbors(self, node, row, col, *_):
        """Helper function for depth-first search to get all neighbors

        Args:
            node (Node): current node being evaluated
            row (int): row of neighbor node
            col (int): column of neighbor node
        """
        n_node = self.nodes[row, col]
        if n_node.mode != 1 and not self.visited[row, col]:
            # only add node if node is unvisited and isn't a wall
            n_node.mode = 6
            self.queue.append(n_node)
            self.paths[row, col] = node

    @_timer
    def dijkstra(self):
        """Dijkstra's algorithm
        """
        self.set_maze_state("reset")
        row, col = self.start.get_pos()
        self.shortest_dists[row, col] = 0
        self.queue = PriorityQueue()
        self.queue.put((0, self.count, self.start))

        while not self.queue.empty():
            for event in pygame.event.get():
                # while algorithm is running, need to make sure quit command is heeded
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # repeatedly get element with shortest distance and get neighbors
            cur_node = self.queue.get()[2]
            if cur_node is self.end:
                break
            self.get_neighbors(cur_node, self.dij_get_neighbors, get_dist=True)
            cur_node.mode = 5
            row, col = cur_node.get_pos()
            self.visited[row, col] = True

            self.update_gui()

        self.print_path()

    def dij_get_neighbors(self, node, row, col, new_dist, *_):
        """Helper function for Dijstra's algorithm to get all neighbors

        Args:
            node (Node): current node being evaluated
            row (int): row of neighbor node
            col (int): column of neighbor node
            new_dist (int): potential new shortest distance to current node
        """
        n_node = self.nodes[row, col]
        dist_bool = new_dist < self.shortest_dists[row, col]
        if n_node.mode != 1 and not self.visited[row, col] and dist_bool:
            # only add node if node is unvisited, isn't a wall,
            # and has current shortest path longer than new dist
            n_node.mode = 6
            self.count += 1
            self.queue.put((new_dist, self.count, n_node))
            self.paths[row, col] = node
            self.shortest_dists[row, col] = new_dist

    @_timer
    def astar(self):
        """A* pathfinding algorithm
        """
        self.set_maze_state("reset")
        row, col = self.start.get_pos()
        self.shortest_dists[row, col] = 0
        self.queue = PriorityQueue()
        self.queue.put((0, self.count, self.start))

        while not self.queue.empty():
            for event in pygame.event.get():
                # while algorithm is running, need to make sure quit command is heeded
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # repeatedly get element with lowest f score and get neighbors
            cur_node = self.queue.get()[2]
            if cur_node is self.end:
                break
            self.get_neighbors(
                cur_node, self.astar_get_neighbors, get_dist=True, end=True
            )
            cur_node.mode = 5
            row, col = cur_node.get_pos()
            self.visited[row, col] = True

            self.update_gui()

        self.print_path()

    def astar_get_neighbors(self, node, row, col, new_dist, end_row, end_col, *_):
        """Helper function for A* algorithm to get all neighbors

        Args:
            node (Node): current node being evaluated
            row (int): row of neighbor node
            col (int): column of neighbor node
            new_dist (int): potential new shortest distance to current node
            end_row (int): row of end node
            end_col (int): column of end node
        """
        n_node = self.nodes[row, col]
        dist_bool = new_dist < self.shortest_dists[row, col]
        if n_node.mode != 1 and not self.visited[row, col] and dist_bool:
            # only add node if node is unvisited, isn't a wall
            # and has current shortest path longer than new dist
            n_node.mode = 6
            self.count += 1
            heuristic = self.manhattan_dist(row, col, end_row, end_col)
            self.queue.put((new_dist + heuristic, self.count, n_node))
            self.paths[row, col] = node
            self.shortest_dists[row, col] = new_dist

    @_timer
    def bi_astar(self):
        """Bi-directional A* pathfinding algorithm
        """
        self.set_maze_state("reset")
        start_row, start_col = self.start.get_pos()
        end_row, end_col = self.end.get_pos()
        self.shortest_dists[start_row, start_col] = 0
        self.shortest_dists[end_row, end_col] = 0

        self.queue = PriorityQueue()
        self.queue2 = PriorityQueue()
        self.queue.put((0, self.count, self.start))
        self.count += 1
        self.queue2.put((0, self.count, self.end))

        turn = 0
        while True:
            for event in pygame.event.get():
                # while algorithm is running, need to make sure quit command is heeded
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if turn % 2 == 0 and not self.help_bi_astar(self.queue, 6):
                break
            elif turn % 2 == 1 and not self.help_bi_astar(self.queue2, 7):
                break
            turn += 1

            self.update_gui()

        self.bi_print_path()

    def help_bi_astar(self, queue, mode):
        """Helper method for bi-directional A*

        Returns:
            bool: True if target node isn't found and queue is not empty, else False
        """
        o_mode = 7 if mode == 6 else 6
        if not queue.empty():
            # repeatedly get element with lowest f score and get neighbors
            cur_node = queue.get()[2]

            if cur_node.mode == o_mode or self.bi_get_neighbors(
                cur_node,
                self.bi_astar_get_neighbors,
                get_dist=True,
                end=True,
                mode=mode,
            ):
                return False
            cur_node.mode = 5

            row, col = cur_node.get_pos()
            self.visited[row, col] = True
            return True

        return False

    def bi_astar_get_neighbors(self, node, row, col, new_dist, end_row, end_col, mode):
        """Helper function for A* algorithm to get all neighbors

        Args:
            node (Node): current node being evaluated
            row (int): row of neighbor node
            col (int): column of neighbor node
            new_dist (int): potential new shortest distance to current node
            end_row (int): row of end node
            end_col (int): column of end node
            mode (int): 6 indicates start, 7 indicates end
        """
        if mode == 6:
            other_mode = 7
            queue = self.queue
        else:
            other_mode = 6
            queue = self.queue2

        n_node = self.nodes[row, col]
        dist_bool = new_dist < self.shortest_dists[row, col]
        if n_node.mode == other_mode:
            self.cur_node, self.target_node = node, n_node
            return True
        if n_node.mode != 1 and not self.visited[row, col] and dist_bool:
            # only add node if node is unvisited, isn't a wall
            # and has current shortest path longer than new dist
            n_node.mode = mode
            self.count += 1
            heuristic = self.manhattan_dist(row, col, end_row, end_col)
            queue.put((new_dist + heuristic, self.count, n_node))
            self.paths[row, col] = node
            self.shortest_dists[row, col] = new_dist

        return False

    @staticmethod
    def manhattan_dist(start_row, start_col, end_row, end_col):
        """Get manhattan distance between two nodes

        Args:
            row (int): row of start node
            col (int): column of start node
            end_row (int): row of end node
            end_col (int): column of end node

        Returns:
            int: manhanttan distance of start and end node
        """
        return abs(end_row - start_row) + abs(end_col - start_col)

    def generate_maze(self, mode):
        """Depth-first-search algorithm to generate maze
        """
        # walling up entire maze
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.nodes[row, col].mode = 1
                self.visited[row, col] = False
        self.queue = deque()
        self.queue.append(self.start)
        row, col = self.start.get_pos()
        self.visited[row, col] = True

        while self.queue:
            # repeatedly get last element in queue and get nieghbors
            cur_node = self.queue.pop()
            row, col = cur_node.get_pos()

            if self.maze_get_neighbors(cur_node, mode):
                cur_node.mode = 0

                if mode != "normal":
                    self.visited[row, col] = True

            if mode == "normal":
                self.visited[row, col] = True

        self.set_maze_state("reset")

    def maze_get_neighbors(self, node, mode):
        """Get all valid neighbors of a node for maze

        Args:
            node (Node): target node to get neighbors of
        """
        row, col = node.get_pos()
        directions = []
        count = 0

        if col != self.num_cols - 1:
            if self.visited[row, col + 1]:
                # if node has already been visited, do not add to queue
                count += 1
            else:
                directions.append(self.eval_right)
        if row != 0:
            if self.visited[row - 1, col]:
                # if node has already been visited, do not add to queue
                count += 1
            else:
                directions.append(self.eval_above)
        if col != 0:
            if self.visited[row, col - 1]:
                # if node has already been visited, do not add to queue
                count += 1
            else:
                directions.append(self.eval_left)
        if row != self.num_rows - 1:
            if self.visited[row + 1, col]:
                # if node has already been visited, do not add to queue
                count += 1
            else:
                directions.append(self.eval_below)

        if mode == "perfect":
            count_limit = 2
        elif mode == "sparse":
            count_limit = 3
        else:
            count_limit = random.choice([2, 3])

        if count >= count_limit:
            # if too many surrounding nodes have been visited
            # then do not unwall this node
            return False

        # add surrounding nodes to queue in random order
        for direction in random.sample(directions, len(directions)):
            direction(row, col)
        return True

    def eval_right(self, row, col):
        """Helper method to evaluate right node

        Args:
            row (int): row of current node
            col (int): column of current node
        """
        self.queue.append(self.nodes[row, col + 1])

    def eval_above(self, row, col):
        """Helper method to evaluate above node

        Args:
            row (int): row of current node
            col (int): column of current node
        """
        self.queue.append(self.nodes[row - 1, col])

    def eval_left(self, row, col):
        """Helper method to evaluate left node

        Args:
            row (int): row of current node
            col (int): column of current node
        """
        self.queue.append(self.nodes[row, col - 1])

    def eval_below(self, row, col):
        """Helper method to evaluate below node

        Args:
            row (int): row of current node
            col (int): column of current node
        """
        self.queue.append(self.nodes[row + 1, col])

    def update_gui(self):
        """Abstract method for children classes to implement.
        Intended to update the GUI
        """
        raise NotImplementedError
