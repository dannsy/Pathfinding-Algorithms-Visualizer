import random
import time
from collections import deque, namedtuple
from queue import PriorityQueue

# from math import sqrt

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
            5: PEACH,
            6: GREY,
        }

        def __init__(self, row, col, num_rows, num_cols):
            self.row = row
            self.col = col
            self.num_rows = num_rows
            self.num_cols = num_cols

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
        def mode(self, value):
            """Setter of mode

            Args:
                value (int): intended new mode for this node
            """
            if self._mode in (0, 4, 5, 6) or (self._mode in (1, 2, 3) and value == 0):
                self._mode = value

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
        for row in range(num_rows):
            for col in range(num_cols):
                self.nodes[row, col] = Algorithms.Node(row, col, num_rows, num_cols)
        # 2D array to mark visited nodes
        self.visited = np.zeros((num_rows, num_cols), dtype=bool)
        # 2D array to mark each nodes previous node
        self.paths = np.zeros((num_rows, num_cols), dtype=Algorithms.Node)
        longest_dist = (self.num_rows * self.num_cols) * 10
        # 2D array to keep track of shortest path from start to each node
        self.shortest_dists = np.full((num_rows, num_cols), longest_dist, dtype=int)
        self.count = 0

        # 1 for building walls, 2 for start position, 3 for end position
        self.mode = 1
        self.start = self.nodes[0, 0]
        self.end = self.nodes[-1, -1]
        self.start.mode = 2
        self.end.mode = 3

        self.solved = False
        self.queue = None

    def clear_maze(self):
        """Clears the entire maze so remove all walls and visited
        """
        longest_dist = (self.num_rows * self.num_cols) * 10

        # clearing all data structures used in pathfinding
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.nodes[row, col].mode = 0
                self.visited[row, col] = False
                self.paths[row, col] = 0
                self.shortest_dists[row, col] = longest_dist

        self.count = 0
        self.start.mode = 2
        self.end.mode = 3
        self.solved = False

    def reset_maze(self):
        """Resets maze so only start, end and walls remain
        """
        longest_dist = (self.num_rows * self.num_cols) * 10

        # resetting all data structures used in pathfinding, taking walls into account
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if self.nodes[row, col].mode != 1:
                    self.nodes[row, col].mode = 0
                    self.visited[row, col] = False
                    self.paths[row, col] = 0
                    self.shortest_dists[row, col] = longest_dist

        self.count = 0
        self.start.mode = 2
        self.end.mode = 3
        self.solved = False

    def print_path(self):
        """Print the shortest path to screen
        """
        end_row, end_col = self.end.get_pos()
        pointer = self.paths[end_row, end_col]
        length = 1

        while pointer != 0:
            length += 1
            # while there is still a previous node, keep printing
            pointer_row, pointer_col = pointer.get_pos()
            pointer.mode = 4
            pointer = self.paths[pointer_row, pointer_col]

            self.update_gui()

        self.solved = True
        print(f"Length of path is: {length}")

    @_timer
    def bfs(self):
        """Breadth-first-search algorithm
        """
        self.reset_maze()
        self.queue = deque()
        self.queue.append(self.start)
        row, col = self.start.get_pos()
        self.visited[row, col] = True

        while self.queue:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # repeatedly get first element in queue and get neighbors
            cur_node = self.queue.popleft()

            if cur_node is self.end:
                # found end node
                break

            self.bfs_get_neighbors(cur_node)
            cur_node.mode = 6

            self.update_gui()

        self.print_path()

    def bfs_get_neighbors(self, node):
        """Get all valid neighbors of a node

        Args:
            node (Node): target node to get neighbors of
        """
        row, col = node.get_pos()

        if col != self.num_cols - 1:
            right_node = self.nodes[row, col + 1]
            if right_node.mode != 1 and not self.visited[row, col + 1]:
                # only add node if node is unvisited and isn't a wall
                right_node.mode = 5
                self.queue.append(right_node)
                self.visited[row, col + 1] = True
                self.paths[row, col + 1] = node

        if row != 0:
            above_node = self.nodes[row - 1, col]
            if above_node.mode != 1 and not self.visited[row - 1, col]:
                # only add node if node is unvisited and isn't a wall
                above_node.mode = 5
                self.queue.append(above_node)
                self.visited[row - 1, col] = True
                self.paths[row - 1, col] = node

        if col != 0:
            left_node = self.nodes[row, col - 1]
            if left_node.mode != 1 and not self.visited[row, col - 1]:
                # only add node if node is unvisited and isn't a wall
                left_node.mode = 5
                self.queue.append(left_node)
                self.visited[row, col - 1] = True
                self.paths[row, col - 1] = node

        if row != self.num_rows - 1:
            below_node = self.nodes[row + 1, col]
            if below_node.mode != 1 and not self.visited[row + 1, col]:
                # only add node if node is unvisited and isn't a wall
                below_node.mode = 5
                self.queue.append(below_node)
                self.visited[row + 1, col] = True
                self.paths[row + 1, col] = node

    @_timer
    def dfs(self):
        """Depth-first-search algorithm
        """
        self.reset_maze()
        self.queue = deque()
        self.queue.append(self.start)
        row, col = self.start.get_pos()
        self.visited[row, col] = True

        while self.queue:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # repeatedly get last element in queue and get nieghbors
            cur_node = self.queue.pop()

            if cur_node is self.end:
                # found end node
                break

            self.dfs_get_neighbors(cur_node)
            cur_node.mode = 6

            row, col = cur_node.get_pos()
            self.visited[row, col] = True

            self.update_gui()

        self.print_path()

    def dfs_get_neighbors(self, node):
        """Get all valid neighbors of a node

        Args:
            node (Node): target node to get neighbors of
        """
        row, col = node.get_pos()

        if col != self.num_cols - 1:
            right_node = self.nodes[row, col + 1]
            if right_node.mode != 1 and not self.visited[row, col + 1]:
                # only add node if node is unvisited and isn't a wall
                right_node.mode = 5
                self.queue.append(right_node)
                self.paths[row, col + 1] = node

        if row != 0:
            above_node = self.nodes[row - 1, col]
            if above_node.mode != 1 and not self.visited[row - 1, col]:
                # only add node if node is unvisited and isn't a wall
                above_node.mode = 5
                self.queue.append(above_node)
                self.paths[row - 1, col] = node

        if col != 0:
            left_node = self.nodes[row, col - 1]
            if left_node.mode != 1 and not self.visited[row, col - 1]:
                # only add node if node is unvisited and isn't a wall
                left_node.mode = 5
                self.queue.append(left_node)
                self.paths[row, col - 1] = node

        if row != self.num_rows - 1:
            below_node = self.nodes[row + 1, col]
            if below_node.mode != 1 and not self.visited[row + 1, col]:
                # only add node if node is unvisited and isn't a wall
                below_node.mode = 5
                self.queue.append(below_node)
                self.paths[row + 1, col] = node

    @_timer
    def dijkstra(self):
        """Dijkstra's algorithm
        """
        self.reset_maze()
        row, col = self.start.get_pos()
        self.shortest_dists[row, col] = 0
        self.queue = PriorityQueue()
        self.queue.put((0, self.count, self.start))

        while not self.queue.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            # repeatedly get element with shortest distance and get neighbors
            cur_node = self.queue.get()[2]

            if cur_node is self.end:
                break

            self.dij_get_neighbors(cur_node)
            cur_node.mode = 6

            row, col = cur_node.get_pos()
            self.visited[row, col] = True

            self.update_gui()

        self.print_path()

    def dij_get_neighbors(self, node):
        """Get all valid neighbors of a node

        Args:
            node (Node): target node to get neighbors of
        """
        row, col = node.get_pos()
        new_dist = self.shortest_dists[row, col] + 1

        if col != self.num_cols - 1:
            right_node = self.nodes[row, col + 1]
            dist_bool = new_dist < self.shortest_dists[row, col + 1]
            if right_node.mode != 1 and not self.visited[row, col + 1] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                right_node.mode = 5
                self.count += 1
                self.queue.put((new_dist, self.count, right_node))
                self.paths[row, col + 1] = node
                self.shortest_dists[row, col + 1] = new_dist

        if row != 0:
            above_node = self.nodes[row - 1, col]
            dist_bool = new_dist < self.shortest_dists[row - 1, col]
            if above_node.mode != 1 and not self.visited[row - 1, col] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                above_node.mode = 5
                self.count += 1
                self.queue.put((new_dist, self.count, above_node))
                self.paths[row - 1, col] = node
                self.shortest_dists[row - 1, col] = new_dist

        if col != 0:
            left_node = self.nodes[row, col - 1]
            dist_bool = new_dist < self.shortest_dists[row, col - 1]
            if left_node.mode != 1 and not self.visited[row, col - 1] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                left_node.mode = 5
                self.count += 1
                self.queue.put((new_dist, self.count, left_node))
                self.paths[row, col - 1] = node
                self.shortest_dists[row, col - 1] = new_dist

        if row != self.num_rows - 1:
            below_node = self.nodes[row + 1, col]
            dist_bool = new_dist < self.shortest_dists[row + 1, col]
            if below_node.mode != 1 and not self.visited[row + 1, col] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                below_node.mode = 5
                self.count += 1
                self.queue.put((new_dist, self.count, below_node))
                self.paths[row + 1, col] = node
                self.shortest_dists[row + 1, col] = new_dist

    @_timer
    def astar(self):
        """A* pathfinding algorithm
        """
        self.reset_maze()
        row, col = self.start.get_pos()
        self.shortest_dists[row, col] = 0
        self.queue = PriorityQueue()
        self.queue.put((0, self.count, self.start))

        while not self.queue.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            # repeatedly get element with shortest distance and get neighbors
            cur_node = self.queue.get()[2]

            if cur_node is self.end:
                break

            self.astar_get_neighbors(cur_node)
            cur_node.mode = 6

            row, col = cur_node.get_pos()
            self.visited[row, col] = True

            self.update_gui()

        self.print_path()

    def astar_get_neighbors(self, node):
        """Get all valid neighbors of a node

        Args:
            node (Node): target node to get neighbors of
        """
        row, col = node.get_pos()
        new_dist = self.shortest_dists[row, col] + 1

        if col != self.num_cols - 1:
            right_node = self.nodes[row, col + 1]
            dist_bool = new_dist < self.shortest_dists[row, col + 1]
            if right_node.mode != 1 and not self.visited[row, col + 1] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                right_node.mode = 5
                self.count += 1
                heuristic = self.manhattan_dist(row, col + 1)
                self.queue.put((new_dist + heuristic, self.count, right_node))
                self.paths[row, col + 1] = node
                self.shortest_dists[row, col + 1] = new_dist

        if row != 0:
            above_node = self.nodes[row - 1, col]
            dist_bool = new_dist < self.shortest_dists[row - 1, col]
            if above_node.mode != 1 and not self.visited[row - 1, col] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                above_node.mode = 5
                self.count += 1
                heuristic = self.manhattan_dist(row - 1, col)
                self.queue.put((new_dist + heuristic, self.count, above_node))
                self.paths[row - 1, col] = node
                self.shortest_dists[row - 1, col] = new_dist

        if col != 0:
            left_node = self.nodes[row, col - 1]
            dist_bool = new_dist < self.shortest_dists[row, col - 1]
            if left_node.mode != 1 and not self.visited[row, col - 1] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                left_node.mode = 5
                self.count += 1
                heuristic = self.manhattan_dist(row, col - 1)
                self.queue.put((new_dist + heuristic, self.count, left_node))
                self.paths[row, col - 1] = node
                self.shortest_dists[row, col - 1] = new_dist

        if row != self.num_rows - 1:
            below_node = self.nodes[row + 1, col]
            dist_bool = new_dist < self.shortest_dists[row + 1, col]
            if below_node.mode != 1 and not self.visited[row + 1, col] and dist_bool:
                # only add node if node is unvisited and isn't a wall
                below_node.mode = 5
                self.count += 1
                heuristic = self.manhattan_dist(row + 1, col)
                self.queue.put((new_dist + heuristic, self.count, below_node))
                self.paths[row + 1, col] = node
                self.shortest_dists[row + 1, col] = new_dist

    def manhattan_dist(self, row, col, end_row=None, end_col=None):
        """Get manhattan distance between two nodes

        Args:
            row (int): row of start node
            col (int): column of start node
            end_row (int, optional): row of end node. Defaults to None.
            end_col (int, optional): column of end node. Defaults to None.

        Returns:
            int: manhanttan distance of start and end node
        """
        if end_row is None:
            end_row, end_col = self.end.get_pos()

        return abs(end_row - row) + abs(end_col - col)

    @_timer
    def generate_maze(self, mode):
        """Depth-first-search algorithm to generate maze
        """
        # clearing all data structures used in pathfinding
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

        self.reset_maze()

    def maze_get_neighbors(self, node, mode):
        """Get all valid neighbors of a node

        Args:
            node (Node): target node to get neighbors of
        """
        row, col = node.get_pos()
        directions = []
        count = 0

        if col != self.num_cols - 1:
            if self.visited[row, col + 1]:
                count += 1
            else:
                directions.append(self.eval_right)
        if row != 0:
            if self.visited[row - 1, col]:
                count += 1
            else:
                directions.append(self.eval_above)
        if col != 0:
            if self.visited[row, col - 1]:
                count += 1
            else:
                directions.append(self.eval_left)
        if row != self.num_rows - 1:
            if self.visited[row + 1, col]:
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
            directions = []

        if directions:
            for direction in random.sample(directions, len(directions)):
                direction(row, col)
            return True
        else:
            return False

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
        pass
