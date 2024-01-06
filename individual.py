import numpy as np
from gol import GoLBoardManager, InteractiveGoLViewer
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, board_manager, structure=None, individual_size=5):
        """
        Initialize an individual for the Game of Life.

        Args:
        board_manager (GoLBoardManager): The board manager to place the individual.
        structure (np.ndarray, optional): A specific structure for the individual. 
                                          If None, a random structure is created.
        individual_size (int): The size of the individual structure.
        """
        self.board_manager = board_manager
        self.individual_size = individual_size
        self.structure = structure if structure is not None else np.random.choice([0, 1], (individual_size, individual_size))
        self.place_individual()

    def place_individual(self):
        """ Place the individual in the middle of the board. """
        start_x = (self.board_manager.size - self.individual_size) // 2
        start_y = (self.board_manager.size - self.individual_size) // 2
        for i in range(self.individual_size):
            for j in range(self.individual_size):
                self.board_manager.set_cell_state(start_x + i, start_y + j, self.structure[i, j])

    def calculate_fitness(self, max_generations=10000):
        """
        Calculate the fitness of the individual based on the evolution of the entire board.

        Args:
        max_generations (int): The maximum number of generations to simulate.

        Returns:
        int: The fitness score of the individual.
        """
        self.fitness = 0
        seen_patterns = set()
        self.board_manager.reset_board()
        self.place_individual()
        for generation in range(max_generations):
            self.board_manager.update_board()
            current_pattern = self.board_manager.board

            # Check if all cells are dead
            if np.all(current_pattern == 0):
                return 0  # Fitness is zero if the individual dies

            # Convert current pattern to a hashable type to track seen patterns
            pattern_hash = hash(current_pattern.tobytes())
            if pattern_hash in seen_patterns:
                self.fitness = generation
                return generation  # Return the generation count if the pattern repeats
            seen_patterns.add(pattern_hash)

        return 0  # Return the maximum if it never stabilized or repeated

    def show(self):
        """
        Display the starting structure of the individual.
        """
        start_x = (self.board_manager.size - self.individual_size) // 2
        start_y = (self.board_manager.size - self.individual_size) // 2
        temp_board = np.zeros((self.board_manager.size, self.board_manager.size), dtype=int)
        for i in range(self.individual_size):
            for j in range(self.individual_size):
                temp_board[start_x + i, start_y + j] = self.structure[i, j]
        
        plt.imshow(temp_board, cmap='binary')
        plt.title("Starting Individual Structure")
        plt.show()

    def simulate(self, max_generations=1000):
        """
        Simulate the individual's life for a specified number of generations and display each state.

        Args:
        max_generations (int): The maximum number of generations to simulate.
        """
        self.board_manager.reset_board()
        self.place_individual()

        states = [self.board_manager.board.copy()]
        for _ in range(max_generations):
            self.board_manager.update_board()
            states.append(self.board_manager.board.copy())

        viewer = InteractiveGoLViewer(states)
        plt.show()




if __name__ == "__main__":

    # Example usage
    board_manager = GoLBoardManager(size=50)
    individual = Individual(board_manager)
    fit = individual.calculate_fitness()  # Implement the fitness calculation logic
    print("calculated fitness:", )

    individual.simulate()
