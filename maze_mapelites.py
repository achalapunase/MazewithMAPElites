import numpy as np
import matplotlib.pyplot as plt
from random import randrange

class Maze:
    # creating a 2D maze with walls.
    def __init__(self, size=1000):
        self.size = size
        self.grid = np.zeros((size, size))

        #  Horizontal walls
        for i in range(50, size-200, 200):
            self.grid[i:i+5, 50:150] = 1  
            self.grid[i+160:i+165, 350:500] = 1  
            self.grid[i+260:i+265, 750:860] = 1                   

        #  Vertical walls
        for j in range(60, size-250, 250):
            self.grid[50:210, j:j+5] = 1  
            self.grid[450:510, j+100:j+105 ] = 1  
            self.grid[780:860, j+250:j+255] = 1
            
        # Clear the space aroundstart position
        self.start_pos = np.array([size // 2, size // 2], dtype=float)
        self.grid[int(self.start_pos[0])-10:int(self.start_pos[0])+10,
                  int(self.start_pos[1])-10:int(self.start_pos[1])+10] = 0


    def collision_occurs(self, x, y):
        # Check if new position collides with wall, return true if collision and false if not
        x, y = int(np.clip(x, 0, self.size-1)), int(np.clip(y, 0, self.size-1))
        return self.grid[x, y] == 1

    def visualize(self, ax=None):
        # Visualize the maze
        if ax is None: 
            fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.grid, cmap='gray', origin='lower')
        ax.plot(self.start_pos[1], self.start_pos[0], 'g*', markersize=15, label='Start')
        ax.set_title('Maze')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        return ax


class NNController:
    # nn w 1 hidden layer to predict the next best movement for the agent. 
    
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5  # gives as many vectors as "input_dim" of dimension "hidden_dim" , total 32 weights
        self.b1 = np.random.randn(hidden_dim) * 0.1 ## gives 1 of dimension "hidden_dim"
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5 #gives 8*2=16 weights
        self.b2 = np.random.randn(output_dim) * 0.1 #2 biases
    
    def forward(self, inputs):
        #Forward pass through network

        # Hidden layer with ReLU
        h = np.dot(inputs, self.W1) + self.b1
        h = np.maximum(0, h)  # ReLU
        
        # Output layer with bounds : tanh (so the agent doesn't get the instruction to move by (134,3242))
        output = np.dot(h, self.W2) + self.b2
        output = np.tanh(output)  # Output in [-1, 1]
        return output
    
    def sep_weights_biases(self, weights):
        # cretae separate weights and biases vectors from 1D array
        idx = 0
        
        size_W1 = self.input_dim * self.hidden_dim
        self.W1 = weights[idx:idx+size_W1].reshape((self.input_dim, self.hidden_dim))
        idx += size_W1
        
        size_b1 = self.hidden_dim
        self.b1 = weights[idx:idx+size_b1]
        idx += size_b1
        
        size_W2 = self.hidden_dim * self.output_dim
        self.W2 = weights[idx:idx+size_W2].reshape((self.hidden_dim, self.output_dim))
        idx += size_W2
        
        size_b2 = self.output_dim
        self.b2 = weights[idx:idx+size_b2]


def simulate_agent(controller, maze, max_steps=350):
    #how the agent moves. start at the center and use NNcontroller to decide where to go at each iteration. if the chosen cell is blocked due to a wall, the agent slides along the wall. 
    #Returns: 1. final_position: [x, y] where agent ended up , 2. fitness

    pos = maze.start_pos.copy()
    start_pos = pos.copy()
    total_distance = 0.0
    visited_cells = set()
    visited_cells.add((int(pos[0]), int(pos[1])))
    max_velocity = randrange(1, 5)
    alpha = 0.1

    for step in range(max_steps):
        # Network input: normalize position
        x_norm = pos[0] / maze.size
        y_norm = pos[1] / maze.size
        t_norm = step / max_steps
        
        inputs = np.array([x_norm, y_norm, np.sin(2*np.pi*t_norm), np.cos(2*np.pi*t_norm)])
        
        # Get action from network
        action = controller.forward(inputs)
        dx, dy = action * max_velocity
        
        #if np.sqrt(dx**2 + dy**2) < 0.005:
         #   break

        # new recommended position
        new_x = pos[0] + dx
        new_y = pos[1] + dy
        
        # move agent to newx and newy. if the recommended position makes the end up on a blocked cell, slide along walls
        if not maze.collision_occurs(new_x, new_y):
            pos[0] = new_x
            pos[1] = new_y
        else:
            if not maze.collision_occurs(new_x, pos[1]):
                pos[0] = new_x
            elif not maze.collision_occurs(pos[0], new_y):
                pos[1] = new_y

        
        total_distance += np.sqrt(dx**2 + dy**2)
        visited_cells.add((int(pos[0]), int(pos[1])))
    

    #Fitness-1: distance traveled + extra points for exploring many cells
    exploration_bonus = len(visited_cells) * 0.5
    #fitness = total_distance + exploration_bonus
   # fitness-2: displacement from start - total distance traveled + exploration bonus
    displacement = np.sqrt((pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2) 
    fitness = 0.8*displacement + exploration_bonus - 0.1*total_distance 
    return pos, fitness


class MapElites:
    #MAP-Elites algorithm for quality-diversity optimization
    
    def __init__(self, feature_ranges, grid_size=32):
        self.feature_ranges = feature_ranges
        self.grid_size = grid_size
        self.feature_dims = len(feature_ranges)  # we have 2 features in our case 
        
        # Dict Archive: {grid indices: (solution, fitness)}
        self.archive = {}
        
        # Tracking stats for all iterations
        self.iteration_stats = []
    
    def get_grid_index(self, features):
        # Convert features to grid indices 
        grid_index = []
        for i, feature in enumerate(features):
            min_val, max_val = self.feature_ranges[i]
            
            feature = np.clip(feature, min_val, max_val)    # clip out of bound x,y positions to the min-max range
            
            idx = int((feature - min_val) / (max_val - min_val) * (self.grid_size - 1))
            grid_index.append(idx)   
        return tuple(grid_index)   
    
    def get_random_elite(self):
        # Get a random elite from archive
        if not self.archive:
            return None
        idx = np.random.randint(len(self.archive))
        return list(self.archive.values())[idx][0] #0th index was our arrayt of weights
    
    def mutate_solution(self, weights, mutation_std=0.2):
        # Gaussian mutation
        return weights + np.random.normal(0, mutation_std, weights.shape)

    def add_update_archive(self, solution_weights, fitness, features):
        # Add solution to archive if it's better than existing elite in that cell
        # Returns True if added/updated, False if not
        grid_idx = self.get_grid_index(features)
        
        # Add if cell is empty or new solution is better
        if grid_idx not in self.archive or fitness > self.archive[grid_idx][1]:
            self.archive[grid_idx] = (solution_weights.copy(), fitness)
            return True
        return False # if no cells filled or updated
    
    def run(self, maze, controller_dim=88, iterations=150, batch_size=150, 
            mutation_std=0.2, init_mutation_std=0.8):
        
        #to run the algo-

        print(f"Starting MAP-Elites for {iterations} iterations, batch_size: {batch_size}")
        print(f"Grid: {self.grid_size}^{self.feature_dims} = {self.grid_size**self.feature_dims} cells")
        print()
        
        for iteration in range(iterations):
            for _ in range(batch_size):
                # Random choice: create new or mutate from archive
                if np.random.rand() < 0.4 or not self.archive:
                    # Random initialization
                    weights = np.random.randn(controller_dim) * init_mutation_std
                else:
                    # Mutate elite
                    elite_weights = self.get_random_elite()
                    weights = self.mutate_solution(elite_weights, mutation_std)
                
                # Create controller and evaluate
                controller = NNController(input_dim=4, hidden_dim=8, output_dim=2)
                controller.sep_weights_biases(weights)
                
                final_pos, fitness = simulate_agent(controller, maze)
                
                # Features are the final position (normalized)
                features = (
                    final_pos[0] / maze.size,
                    final_pos[1] / maze.size
                )
                
                # Add to archive
                self.add_update_archive(weights, fitness, features)
            
            # Statistics
            if self.archive:
                fitnesses = [f for _, f in self.archive.values()]
                self.iteration_stats.append({
                    'iteration': iteration,
                    'archive_size': len(self.archive),
                    'best_fitness': max(fitnesses),
                    'mean_fitness': np.mean(fitnesses),
                    'coverage': 100 * len(self.archive) / (self.grid_size ** self.feature_dims)
                })
                
                if (iteration + 1) % 10 == 0:
                    stats = self.iteration_stats[-1]
                    print(f"Iteration {iteration + 1}: "
                          f"Archive size={stats['archive_size']}, "
                          f"Best fitness ={stats['best_fitness']:.2f}, "
                          f"Coverage={stats['coverage']:.1f}%")
        
        print(f"\n Final: {len(self.archive)} solutions in archive")
        return self.archive
    
    def visualize(self, maze):
        #Visualize the maze and archive 
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Maze Environment 
        ax = axes[0, 0]
        ax.imshow(maze.grid, cmap='gray', origin='lower')
        ax.plot(maze.start_pos[1], maze.start_pos[0], 'g*', markersize=15, label='Start')
        ax.set_title('Environment: Maze layout')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()


        # Plot 2: Archive grid (fitness heatmap)
        ax = axes[0, 1]
        grid_fitness = np.full((self.grid_size, self.grid_size), np.nan)
        
        for (x_idx, y_idx), (_, fitness) in self.archive.items():
            grid_fitness[y_idx, x_idx] = fitness
        
        im1 = ax.imshow(grid_fitness, cmap='viridis', origin='lower', aspect='auto')
        ax.set_title('Archive: Fitness Heatmap')
        ax.set_xlabel('Behavior: Final X Position')
        ax.set_ylabel('Behavior: Final Y Position')
        plt.colorbar(im1, ax=ax, label='Fitness (Displacement - Penalty)')
        
        
        # Plot 3: Statistics over time
        ax = axes[1, 0]
        iterations = [s['iteration'] for s in self.iteration_stats]
        archive_sizes = [s['archive_size'] for s in self.iteration_stats]
        
        ax.plot(iterations, archive_sizes, 'o-', label='Archive Size', linewidth=2, markersize=4)
        ax.fill_between(iterations, archive_sizes, alpha=0.3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Archive Size')
        ax.set_title('Archive Growth Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Best fitness over time
        ax = axes[1, 1]
        best_fitnesses = [s['best_fitness'] for s in self.iteration_stats]
        ax.plot(iterations, best_fitnesses, 's-', color='orange', label='Best Fitness', linewidth=2, markersize=4)
        ax.fill_between(iterations, best_fitnesses, alpha=0.3, color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Best Fitness Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze(self):
        #Print analysis of results
        if not self.archive:
            print("Archive is empty!")
            return
        
        fitnesses = [f for _, f in self.archive.values()]
        
        
        print("\nMAP-ELITES RESULTS ANALYSIS")

        print(f"\nBehavior Space Coverage:")
        print(f"  Behavior dimensions: {self.feature_dims}")
        print(f"  Grid resolution: {self.grid_size}x{self.grid_size}")
        print(f"  Total possible cells: {self.grid_size**self.feature_dims}")

        print(f"\nArchive Statistics:")
        print(f"  Total elites found: {len(self.archive)}")
        print(f"  Maximum fitness: {max(fitnesses):.4f}")
        print(f"  Grid coverage: {100*len(self.archive)/(self.grid_size**2):.1f}%")
        



print("MAP-ELITES: MAZE NAVIGATION")

# Create maze
print("\nCreating maze...")
maze = Maze(size=1000)

# Initialize MAP-Elites
print("Initializing MAP-Elites...")
map_elites = MapElites(
    feature_ranges=[(0, 1), (0, 1)],  # X and Y coordinates
    grid_size=32  # 16x16 grid
)

# Run algorithm
print("Running MAP-Elites...\n")
archive = map_elites.run(
    maze=maze,
    controller_dim=88,  #  4x8 + 8 + 8x2 + 2 = 88 weights
    iterations=150,
    batch_size=150,
    mutation_std=0.2,
    init_mutation_std=0.8
)

# Analyze and visualize
map_elites.analyze()
map_elites.visualize(maze)

breakpoint()