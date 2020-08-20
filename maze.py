"""This module contains a visualizer class for the 4 pathfinding algorithms
"""
import sys
from collections import namedtuple

import pygame

from algorithms import Algorithms

Color = namedtuple("Color", ["r", "g", "b"])
WHITE = Color(240, 240, 240)
BLACK = Color(0, 0, 0)
BLUE = Color(60, 210, 210)
PURPLE = Color(235, 40, 235)
YELLOW = Color(235, 235, 90)
LIGHT_GREY = Color(210, 210, 210)


class Maze(Algorithms):
    """Represents the maze in which the pathfinding algorithms will run.
    Walls can be erected in the maze to test the pathfinding algorithms
    """

    # constants for grid
    spacing = 20
    top_pad = 100

    # constants for buttons
    but_width = 100
    but_height = 40
    but_y = 15
    bfs_x = 100
    dfs_x = bfs_x + 130
    dij_x = bfs_x + 130 * 2
    astar_x = bfs_x + 130 * 3
    bi_astar_x = bfs_x + 130 * 4
    per_x = bfs_x + 130 * 5 + 100
    nor_x = bfs_x + int(130 * 6.5) + 100
    spa_x = bfs_x + 130 * 8 + 100

    # rectangle of buttons
    bfs_but = pygame.Rect(bfs_x, but_y, but_width, but_height)
    dfs_but = pygame.Rect(dfs_x, but_y, but_width, but_height)
    dij_but = pygame.Rect(dij_x, but_y, but_width, but_height)
    astar_but = pygame.Rect(astar_x, but_y, but_width, but_height)
    bi_astar_but = pygame.Rect(bi_astar_x, but_y, but_width, but_height)
    per_but = pygame.Rect(per_x, but_y, int(but_width * 1.5), but_height)
    nor_but = pygame.Rect(nor_x, but_y, int(but_width * 1.5), but_height)
    spa_but = pygame.Rect(spa_x, but_y, int(but_width * 1.5), but_height)

    def __init__(self, width, height):
        self.width = width
        self.height = height + self.top_pad
        super(Maze, self).__init__(height // self.spacing, width // self.spacing)
        self.but_list = [
            (self.bfs_but, "BFS", self.bfs, None),
            (self.dfs_but, "DFS", self.dfs, None),
            (self.dij_but, "Dijkstra", self.dijkstra, None),
            (self.astar_but, "A*", self.astar, None),
            (self.bi_astar_but, "Bi-A*", self.bi_astar, None),
            (self.per_but, "Perfect Maze", self.generate_maze, "perfect"),
            (self.nor_but, "Normal Maze", self.generate_maze, "normal"),
            (self.spa_but, "Sparse Maze", self.generate_maze, "sparse"),
        ]

        self.display = pygame.display.set_mode((self.width, self.height))
        self.running = True
        self.font = pygame.font.SysFont("calibri", 20, True)

        self.main()

    def click_pos(self, pos, delete=False):
        """Detect which node user clicked on

        Args:
            pos (tuple of int): pos[0] is the x position, pos[1] is the y position
            delete (bool, optional): determine if deleting wall. Defaults to False.
        """
        if self.solved:
            # remove coloring of previously visited nodes
            self.set_maze_state("reset")

        col = pos[0] // self.spacing
        row = (pos[1] - self.top_pad) // self.spacing
        if row >= 0:
            if delete and self.nodes[row, col].mode == 1:
                self.nodes[row, col].mode = 0
            elif not delete:
                if self.mode == 1:
                    # building walls
                    self.nodes[row, col].mode = 1
                elif self.mode == 2 and self.nodes[row, col].mode == 0:
                    # setting start position
                    self.start.mode = 0
                    self.start = self.nodes[row, col]
                    self.start.mode = 2
                elif self.mode == 3 and self.nodes[row, col].mode == 0:
                    # setting end position
                    self.end.mode = 0
                    self.end = self.nodes[row, col]
                    self.end.mode = 3

        for but, _, func, arg in self.but_list:
            # detecting button clicks
            if but.collidepoint(pos):
                if arg is None:
                    func()
                else:
                    func(arg)

    def draw_grid(self):
        """Drawing the grid lines of the maze
        """
        for i in range(self.num_rows):
            # drawing horizontal lines
            start_pos = (0, self.top_pad + i * self.spacing)
            end_pos = (self.width, self.top_pad + i * self.spacing)
            pygame.draw.line(self.display, BLUE, start_pos, end_pos)
        for i in range(self.num_cols):
            # drawing vertical lines
            start_pos = (i * self.spacing, self.top_pad)
            end_pos = (i * self.spacing, self.height)
            pygame.draw.line(self.display, BLUE, start_pos, end_pos)

    def draw_nodes(self):
        """Drawing the nodes (boxes) of the maze
        """
        s_row, s_col = self.start.get_pos()
        # used to calculate gradual color change of visited nodes
        max_dist = self.num_rows + self.num_cols

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                node = self.nodes[row, col]
                node.enforce_color()

                if node.mode == 5:
                    # change visited nodes color according to distance from start node
                    dist = self.manhattan_dist(row, col, s_row, s_col) / max_dist
                    color = Color(250, 220 - 180 * dist, 40 + 180 * dist)
                else:
                    # all other types of node have a constant color
                    color = node.color

                if node.mode not in (2, 3):
                    pygame.draw.rect(
                        self.display,
                        color,
                        pygame.Rect(
                            col * self.spacing,
                            self.top_pad + row * self.spacing,
                            self.spacing,
                            self.spacing,
                        ),
                    )
                else:
                    # if node is start or end node, draw circle instead of square
                    pygame.draw.circle(
                        self.display,
                        color,
                        (
                            col * self.spacing + self.spacing // 2,
                            self.top_pad + row * self.spacing + self.spacing // 2,
                        ),
                        self.spacing // 2,
                    )

    def draw_buts(self):
        """Drawing all buttons and their text
        """
        for but, content, *_ in self.but_list:
            pygame.draw.rect(self.display, LIGHT_GREY, but)
            text = self.font.render(content, True, BLACK)
            text_x = but.centerx - text.get_width() // 2
            text_y = but.centery - text.get_height() // 2
            self.display.blit(text, (text_x, text_y))

    def draw_mode(self):
        """Drawing the current mode on top left hand corner
        """
        if self.mode == 1:
            text = self.font.render("WALL", True, BLUE)
        elif self.mode == 2:
            text = self.font.render("START", True, YELLOW)
        else:
            text = self.font.render("END", True, PURPLE)
        self.display.blit(text, (15, 25))

    def update_gui(self):
        """Updates the view of the maze
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()

        self.display.fill(WHITE)

        self.draw_nodes()
        self.draw_grid()
        self.draw_buts()
        self.draw_mode()

        pygame.display.update()

    def main(self):
        """Main running loop of the GUI, takes keyboard and mouse inputs
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                if pygame.mouse.get_pressed()[0]:
                    # if user has left clicked on mouse
                    self.click_pos(pygame.mouse.get_pos())
                if pygame.mouse.get_pressed()[2]:
                    # if user has right clicked on mouse
                    self.click_pos(pygame.mouse.get_pos(), delete=True)
                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_m]:
                        self.mode = (self.mode) % 3 + 1
                    if keys[pygame.K_LCTRL] and keys[pygame.K_r]:
                        self.set_maze_state("reset")
                    if keys[pygame.K_LCTRL] and keys[pygame.K_c]:
                        self.set_maze_state("clear")

            self.update_gui()


if __name__ == "__main__":
    pygame.init()
    maze = Maze(1400, 800)
