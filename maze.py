### implement dijkstra, A*, greedy BFS, BFS, DFS
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

    spacing = 20
    top_pad = 100

    but_width = 100
    but_height = 40
    but_y = 15
    bfs_x = 100
    dfs_x = bfs_x + 130
    dij_x = bfs_x + 130 * 2
    astar_x = bfs_x + 130 * 3
    per_x = bfs_x + 130 * 5
    nor_x = bfs_x + int(130 * 6.5)
    spa_x = bfs_x + 130 * 8

    bfs_but = pygame.Rect(bfs_x, but_y, but_width, but_height)
    dfs_but = pygame.Rect(dfs_x, but_y, but_width, but_height)
    dij_but = pygame.Rect(dij_x, but_y, but_width, but_height)
    astar_but = pygame.Rect(astar_x, but_y, but_width, but_height)
    per_but = pygame.Rect(per_x, but_y, int(but_width * 1.5), but_height)
    nor_but = pygame.Rect(nor_x, but_y, int(but_width * 1.5), but_height)
    spa_but = pygame.Rect(spa_x, but_y, int(but_width * 1.5), but_height)

    def __init__(self, width, height):
        self.width = width
        self.height = height + self.top_pad
        super(Maze, self).__init__(height // self.spacing, width // self.spacing)

        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("calibri", 20, True)

        self.main()

    def click_pos(self, pos, delete=False):
        """Detect which node user clicked on

        Args:
            pos (tuple of int): pos[0] is the x position, pos[1] is the y position
        """
        if self.solved:
            self.reset_maze()

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
        elif self.bfs_but.collidepoint(pos):
            self.bfs()
        elif self.dfs_but.collidepoint(pos):
            self.dfs()
        elif self.dij_but.collidepoint(pos):
            self.dijkstra()
        elif self.astar_but.collidepoint(pos):
            self.astar()
        elif self.per_but.collidepoint(pos):
            self.generate_maze("perfect")
        elif self.nor_but.collidepoint(pos):
            self.generate_maze("normal")
        elif self.spa_but.collidepoint(pos):
            self.generate_maze("sparse")

    def draw_grid(self):
        """Drawing the grid lines of the maze
        """
        for i in range(self.num_rows):
            pygame.draw.line(
                self.display,
                BLUE,
                (0, self.top_pad + i * self.spacing),
                (self.width, self.top_pad + i * self.spacing),
                1,
            )
        for i in range(self.num_cols):
            pygame.draw.line(
                self.display,
                BLUE,
                (i * self.spacing, self.top_pad),
                (i * self.spacing, self.height),
            )

    def draw_nodes(self):
        """Drawing the nodes (boxes) of the maze
        """
        s_row, s_col = self.start.get_pos()
        max_dist = self.num_rows + self.num_cols

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                node = self.nodes[row, col]
                node.enforce_color()

                if node.mode == 6:
                    dist = self.manhattan_dist(row, col, s_row, s_col) / max_dist
                    color = Color(250, 220 - 180 * dist, 40 + 180 * dist)
                else:
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
        pygame.draw.rect(self.display, LIGHT_GREY, self.bfs_but)
        pygame.draw.rect(self.display, LIGHT_GREY, self.dfs_but)
        pygame.draw.rect(self.display, LIGHT_GREY, self.dij_but)
        pygame.draw.rect(self.display, LIGHT_GREY, self.astar_but)
        pygame.draw.rect(self.display, LIGHT_GREY, self.per_but)
        pygame.draw.rect(self.display, LIGHT_GREY, self.nor_but)
        pygame.draw.rect(self.display, LIGHT_GREY, self.spa_but)

        text = self.font.render("DFS", True, BLACK)
        text_x = self.dfs_x + (self.but_width - text.get_width()) // 2
        text_y = self.but_y + (self.but_height - text.get_height()) // 2
        self.display.blit(text, (text_x, text_y))
        text = self.font.render("BFS", True, BLACK)
        text_x = self.bfs_x + (self.but_width - text.get_width()) // 2
        text_y = self.but_y + (self.but_height - text.get_height()) // 2
        self.display.blit(text, (text_x, text_y))
        text = self.font.render("Dijkstra", True, BLACK)
        text_x = self.dij_x + (self.but_width - text.get_width()) // 2
        text_y = self.but_y + (self.but_height - text.get_height()) // 2
        self.display.blit(text, (text_x, text_y))
        text = self.font.render("A*", True, BLACK)
        text_x = self.astar_x + (self.but_width - text.get_width()) // 2
        text_y = self.but_y + (self.but_height - text.get_height()) // 2
        self.display.blit(text, (text_x, text_y))
        text = self.font.render("Perfect Maze", True, BLACK)
        text_x = self.per_x + (int(self.but_width * 1.5) - text.get_width()) // 2
        text_y = self.but_y + (self.but_height - text.get_height()) // 2
        self.display.blit(text, (text_x, text_y))
        text = self.font.render("Normal Maze", True, BLACK)
        text_x = self.nor_x + (int(self.but_width * 1.5) - text.get_width()) // 2
        text_y = self.but_y + (self.but_height - text.get_height()) // 2
        self.display.blit(text, (text_x, text_y))
        text = self.font.render("Sparse Maze", True, BLACK)
        text_x = self.spa_x + (int(self.but_width * 1.5) - text.get_width()) // 2
        text_y = self.but_y + (self.but_height - text.get_height()) // 2
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
                        self.reset_maze()
                    if keys[pygame.K_LCTRL] and keys[pygame.K_c]:
                        self.clear_maze()

            # self.clock.tick(60)
            self.update_gui()


if __name__ == "__main__":
    pygame.init()
    maze = Maze(1400, 800)
