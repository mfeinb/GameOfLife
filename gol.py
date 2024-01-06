import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class GoLBoardManager:
    def __init__(self, size=50):
        """
        Initialize the Game of Life board.

        Args:
        size (int): The size of the game board (size x size).
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def reset_board(self):
        """ Reset the board to all dead cells. """
        self.board.fill(0)
    
    def randomize_board(self):
        """ Randomize the state of the board. """
        self.board = np.random.choice([0, 1], (self.size, self.size))

    def update_board(self):
        """ 
        Update the board by applying the Game of Life rules to each cell.
        """
        new_board = self.board.copy()
        for i in range(self.size):
            for j in range(self.size):
                live_neighbors = self._count_live_neighbors(i, j)
                if self.board[i, j] == 1:  # Alive cell
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_board[i, j] = 0  # Cell dies
                else:  # Dead cell
                    if live_neighbors == 3:
                        new_board[i, j] = 1  # Cell becomes alive
        self.board = new_board

    def _count_live_neighbors(self, x, y):
        """
        Count the number of live neighbors around a given cell.

        Args:
        x (int): X-coordinate of the cell.
        y (int): Y-coordinate of the cell.

        Returns:
        int: The number of live neighbors.
        """
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                ni, nj = x + i, y + j
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    count += self.board[ni, nj]
        return count

    def display_board(self):
        """ Display the board using matplotlib. """
        plt.imshow(self.board, cmap='binary')
        plt.title("Game of Life Board")
        plt.show()

    def set_cell_state(self, x, y, state):
        """
        Set the state of a specific cell.

        Args:
        x (int): X-coordinate of the cell.
        y (int): Y-coordinate of the cell.
        state (int): The new state of the cell (1 for alive, 0 for dead).
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            self.board[x, y] = state
    
    def run_iterations(self, num_iterations):
        """
        Run a specified number of iterations, storing each state.

        Args:
        num_iterations (int): The number of iterations to run.

        Returns:
        List[np.ndarray]: A list of board states.
        """
        states = [self.board.copy()]
        for _ in range(num_iterations):
            self.update_board()
            states.append(self.board.copy())
        return states


class InteractiveGoLViewer:
    def __init__(self, states):
        self.states = states
        self.index = 0
        self.fig, self.ax = plt.subplots()
        self.img = plt.imshow(self.states[0], cmap='binary')
        plt.title("Game of Life")

        # Add buttons for navigation
        self.btn_prev = Button(plt.axes([0.1, 0.05, 0.1, 0.075]), 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next = Button(plt.axes([0.8, 0.05, 0.1, 0.075]), 'Next')
        self.btn_next.on_clicked(self.next_frame)

    def prev_frame(self, event):
        if self.index > 0:
            self.index -= 1
            self.update()

    def next_frame(self, event):
        if self.index < len(self.states) - 1:
            self.index += 1
            self.update()

    def update(self):
        self.img.set_data(self.states[self.index])
        self.ax.set_title(f"Frame {self.index}")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # Example Usage
    board_manager = GoLBoardManager(size=50)
    board_manager.set_cell_state(25, 25, 1)  # Example to set a specific cell
    board_manager.randomize_board()
    #board_manager.display_board()            # Display initial state
    #board_manager.update_board()             # Apply one iteration of GoL rules
    #board_manager.display_board()            # Display updated state

    # states = board_manager.run_iterations(50)
    # viewer = InteractiveGoLViewer(states)
    # plt.show()
