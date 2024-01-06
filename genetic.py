import numpy as np
import random
from individual import Individual
from gol import GoLBoardManager

class GeneticAlgorithm:
    def __init__(self, board_manager, population_size=10, individual_size=5, max_generations=1000, fitness_max_gen=1000):
        """
        Initialize the Genetic Algorithm for Conway's Game of Life.

        Args:
        population_size (int): The number of individuals in the population.
        board_manager (GoLBoardManager): The board manager for running the Game of Life.
        individual_size (int): The size of each individual (pattern size).
        max_generations (int): The maximum number of generations to simulate for each individual.
        """
        self.board_manager = board_manager
        self.population_size = population_size
        self.individual_size = individual_size
        self.max_generations = max_generations
        self.fitness_max_gen = fitness_max_gen
        self.population = [Individual(self.board_manager, individual_size=self.individual_size) for _ in range(population_size)]

    def evaluate_fitness(self):
        """ Evaluate the fitness of each individual in the population. """
        for individual in self.population:
            individual.calculate_fitness(self.fitness_max_gen)

    def select(self, tournament_size=3):
        """ 
        Select individuals using tournament selection.

        Args:
        tournament_size (int): The number of individuals in each tournament.

        Returns:
        List[Individual]: Selected individuals for the next generation.
        """
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected


    def crossover(self, parent1, parent2, rate = 0.1):
        """
        Perform crossover between two parents to produce offspring.

        Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

        Returns:
        Individual: Offspring resulting from the crossover.
        """
        # Choose a crossover point, which is a slice index in the individual's structure
        crossover_point = random.randint(1, self.individual_size - 1)

        # Create new structure for the offspring
        new_structure = np.zeros((self.individual_size, self.individual_size), dtype=int)

        # First half from parent1, second half from parent2
        new_structure[:crossover_point, :] = parent1.structure[:crossover_point, :]
        new_structure[crossover_point:, :] = parent2.structure[crossover_point:, :]

        # Create a new individual with the combined structure
        offspring = Individual(self.board_manager, structure=new_structure, individual_size=self.individual_size)
        return offspring


    def mutate(self, individual, mutation_rate=0.1):
        """
        Mutate an individual by randomly flipping some cells.

        Args:
        individual (Individual): The individual to mutate.
        mutation_rate (float): The probability of each cell being mutated.

        Returns:
        Individual: The mutated individual.
        """
        new_structure = individual.structure.copy()
        for i in range(individual.individual_size):
            for j in range(individual.individual_size):
                if random.random() < mutation_rate:
                    # Flip the cell state (0 -> 1 or 1 -> 0)
                    new_structure[i, j] = 1 if new_structure[i, j] == 0 else 0

        # Create a new individual with the mutated structure
        mutated_individual = Individual(self.board_manager, structure=new_structure, individual_size=individual.individual_size)
        return mutated_individual
    
    def __str__(self):
        return str([ind.fitness for ind in self.population])

    def run(self):
        """ Run the genetic algorithm for a specified number of generations. """
        for generation in range(self.max_generations):
            self.evaluate_fitness()
            # Sort the population based on fitness (descending order)
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Reporting
            best_fitness = self.population[0].fitness
            print(f"Generation {generation + 1}/{self.max_generations} - Best Fitness: {best_fitness}")
            print("[*]", [ind.fitness for ind in self.population])
 
            # Elitism: Carry over a portion of the best individuals
            elite_count = int(self.population_size * 0.3)
            new_population = self.population[:elite_count]

            # Fill the rest of the next generation
            selected = self.select()
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(selected, 2, replace=False)
                offspring1 = self.crossover(parent1, parent2)
                offspring2 = self.crossover(parent2, parent1)
                new_population.extend([self.mutate(offspring1), self.mutate(offspring2)])

            self.population = new_population
    

if __name__ == "__main__":
    board_manager = GoLBoardManager(size=50)
    ga = GeneticAlgorithm(population_size=10, board_manager=board_manager, max_generations=20)
    ga.run()

    p1, p2, p3 = ga.population[1:4]
    p1.show()
    p2.show()
    p3.show()
