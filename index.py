import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import List, Tuple
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("whitegrid")

@dataclass
class Organism:
    """Represents an individual organism with genetic traits"""
    traits: np.ndarray  # Array of trait values
    fitness: float = 0.0
    age: int = 0

class Population:
    """Manages a population of organisms"""
    def __init__(self, size: int, num_traits: int, trait_range: Tuple[float, float] = (0, 10)):
        self.size = size
        self.num_traits = num_traits
        self.trait_range = trait_range
        self.organisms: List[Organism] = []
        self.initialize_population()
        
    def initialize_population(self):
        """Create initial random population"""
        self.organisms = []
        for _ in range(self.size):
            traits = np.random.uniform(self.trait_range[0], self.trait_range[1], self.num_traits)
            self.organisms.append(Organism(traits=traits))
    
    def calculate_fitness(self, optimal_traits: np.ndarray, selection_strength: float):
        """Calculate fitness based on distance from optimal traits"""
        for org in self.organisms:
            distance = np.sqrt(np.sum((org.traits - optimal_traits) ** 2))
            org.fitness = np.exp(-selection_strength * distance)
    
    def get_trait_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and std of each trait"""
        trait_matrix = np.array([org.traits for org in self.organisms])
        return np.mean(trait_matrix, axis=0), np.std(trait_matrix, axis=0)

class EvolutionSimulator:
    """Main simulator for evolutionary processes"""
    def __init__(self, population_size: int = 200, num_traits: int = 3):
        self.population = Population(population_size, num_traits)
        self.generation = 0
        self.history = {
            'generations': [],
            'mean_fitness': [],
            'trait_means': [[] for _ in range(num_traits)],
            'trait_stds': [[] for _ in range(num_traits)],
            'allele_frequencies': [],
            'population_size': []
        }
        
        # Simulation parameters
        self.optimal_traits = np.array([7.0, 5.0, 8.0])
        self.selection_strength = 0.3
        self.mutation_rate = 0.1
        self.mutation_std = 0.5
        self.migration_rate = 0.0
        self.genetic_drift_strength = 0.0
        
    def simulate_generation(self):
        """Simulate one generation of evolution"""
        # Calculate fitness
        self.population.calculate_fitness(self.optimal_traits, self.selection_strength)
        
        # Natural Selection: Reproduce based on fitness
        new_organisms = self.selection()
        
        # Mutation
        self.mutate(new_organisms)
        
        # Genetic Drift
        if self.genetic_drift_strength > 0:
            new_organisms = self.apply_drift(new_organisms)
        
        # Gene Flow (Migration)
        if self.migration_rate > 0:
            new_organisms = self.apply_gene_flow(new_organisms)
        
        self.population.organisms = new_organisms
        # Recompute fitness for the new population after reproduction/mutation/drift/migration
        self.population.calculate_fitness(self.optimal_traits, self.selection_strength)
        self.generation += 1
        
        # Record statistics
        self.record_statistics()
    
    def selection(self) -> List[Organism]:
        """Natural selection through fitness-proportionate reproduction"""
        fitnesses = np.array([org.fitness for org in self.population.organisms])
        
        # Normalize fitness
        if fitnesses.sum() > 0:
            probabilities = fitnesses / fitnesses.sum()
        else:
            probabilities = np.ones(len(fitnesses)) / len(fitnesses)
        
        # Select parents
        new_organisms = []
        for _ in range(self.population.size):
            parent_idx = np.random.choice(len(self.population.organisms), p=probabilities)
            parent = self.population.organisms[parent_idx]
            offspring = Organism(traits=parent.traits.copy())
            new_organisms.append(offspring)
        
        return new_organisms
    
    def mutate(self, organisms: List[Organism]):
        """Apply mutations to organisms"""
        for org in organisms:
            if np.random.random() < self.mutation_rate:
                mutation = np.random.normal(0, self.mutation_std, len(org.traits))
                org.traits += mutation
                # Keep traits within bounds
                org.traits = np.clip(org.traits, self.population.trait_range[0], 
                                    self.population.trait_range[1])
    
    def apply_drift(self, organisms: List[Organism]) -> List[Organism]:
        """Apply genetic drift (random sampling)"""
        drift_size = int(len(organisms) * (1 - self.genetic_drift_strength))
        drift_size = max(10, drift_size)  # Minimum population
        
        if drift_size < len(organisms):
            selected_indices = np.random.choice(len(organisms), drift_size, replace=False)
            organisms = [organisms[i] for i in selected_indices]
            
            # Repopulate to original size
            while len(organisms) < self.population.size:
                parent = organisms[np.random.randint(len(organisms))]
                organisms.append(Organism(traits=parent.traits.copy()))
        
        return organisms
    
    def apply_gene_flow(self, organisms: List[Organism]) -> List[Organism]:
        """Apply gene flow (migration from external population)"""
        num_migrants = int(len(organisms) * self.migration_rate)
        
        for _ in range(num_migrants):
            # Replace random organism with migrant
            idx = np.random.randint(len(organisms))
            migrant_traits = np.random.uniform(self.population.trait_range[0], 
                                              self.population.trait_range[1], 
                                              self.population.num_traits)
            organisms[idx] = Organism(traits=migrant_traits)
        
        return organisms
    
    def record_statistics(self):
        """Record population statistics"""
        self.history['generations'].append(self.generation)
        
        # Mean fitness
        fitnesses = [org.fitness for org in self.population.organisms]
        self.history['mean_fitness'].append(np.mean(fitnesses))
        
        # Trait statistics
        means, stds = self.population.get_trait_stats()
        for i in range(self.population.num_traits):
            self.history['trait_means'][i].append(means[i])
            self.history['trait_stds'][i].append(stds[i])
        
        # Population size
        self.history['population_size'].append(len(self.population.organisms))
    
    def reset(self):
        """Reset simulation"""
        self.population.initialize_population()
        self.generation = 0
        self.history = {
            'generations': [],
            'mean_fitness': [],
            'trait_means': [[] for _ in range(self.population.num_traits)],
            'trait_stds': [[] for _ in range(self.population.num_traits)],
            'allele_frequencies': [],
            'population_size': []
        }

class SimulatorGUI:
    """GUI for the evolution simulator"""
    def __init__(self, root):
        self.root = root
        self.root.title("Evolution Simulator - Natural Selection, Drift & Gene Flow")
        # --- MODIFIED: Changed geometry for vertical layout ---
        self.root.geometry("1000x1200") 
        
        self.simulator = EvolutionSimulator()
        self.running = False
        self.tick_ms = 100  # Simulation speed (milliseconds per step)
        self.bottom_right_mode = tk.StringVar(value="Population Size")
        
        self.setup_ui()
        
        # Initialize the plots with the starting population
        self.simulator.population.calculate_fitness(self.simulator.optimal_traits, 
                                                    self.simulator.selection_strength)
        self.simulator.record_statistics()
        self.update_plots()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Control Panel with larger fonts and better spacing
        control_frame = ttk.Frame(self.root, padding="20")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.columnconfigure(1, weight=1)
        
        # Configure style for larger fonts
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 20))
        style.configure('TButton', font=('Arial', 20), padding=10)
        style.configure('TEntry', font=('Arial', 20))
        style.configure('TCombobox', font=('Arial', 20))

        row_padding = 8
        
        # Parameters with sliders AND entry boxes
        # Selection Strength
        ttk.Label(control_frame, text="Selection Strength:").grid(
            row=0, column=0, sticky=tk.W, pady=row_padding)
        self.selection_scale = ttk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                         command=self.update_selection, length=500)
        self.selection_scale.set(0.3)
        self.selection_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.selection_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.selection_entry.insert(0, "0.30")
        self.selection_entry.grid(row=0, column=2, padx=5, pady=row_padding)
        self.selection_entry.bind('<Return>', lambda e: self.update_from_entry('selection'))
        
        # Mutation Rate
        ttk.Label(control_frame, text="Mutation Rate:").grid(
            row=1, column=0, sticky=tk.W, pady=row_padding)
        self.mutation_scale = ttk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                        command=self.update_mutation, length=300)
        self.mutation_scale.set(0.1)
        self.mutation_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.mutation_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.mutation_entry.insert(0, "0.10")
        self.mutation_entry.grid(row=1, column=2, padx=5, pady=row_padding)
        self.mutation_entry.bind('<Return>', lambda e: self.update_from_entry('mutation'))
        
        # Genetic Drift
        ttk.Label(control_frame, text="Genetic Drift:").grid(
            row=2, column=0, sticky=tk.W, pady=row_padding)
        self.drift_scale = ttk.Scale(control_frame, from_=0, to=0.5, orient=tk.HORIZONTAL,
                                     command=self.update_drift, length=300)
        self.drift_scale.set(0)
        self.drift_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.drift_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.drift_entry.insert(0, "0.00")
        self.drift_entry.grid(row=2, column=2, padx=5, pady=row_padding)
        self.drift_entry.bind('<Return>', lambda e: self.update_from_entry('drift'))
        
        # Migration Rate
        ttk.Label(control_frame, text="Migration Rate:").grid(
            row=3, column=0, sticky=tk.W, pady=row_padding)
        self.migration_scale = ttk.Scale(control_frame, from_=0, to=0.3, orient=tk.HORIZONTAL,
                                         command=self.update_migration, length=300)
        self.migration_scale.set(0)
        self.migration_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.migration_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.migration_entry.insert(0, "0.00")
        self.migration_entry.grid(row=3, column=2, padx=5, pady=row_padding)
        self.migration_entry.bind('<Return>', lambda e: self.update_from_entry('migration'))
        
        # Optimal trait controls
        ttk.Label(control_frame, text="Optimal Trait 1:").grid(
            row=4, column=0, sticky=tk.W, pady=row_padding)
        self.trait1_scale = ttk.Scale(control_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                      command=self.update_trait1, length=300)
        self.trait1_scale.set(7.0)
        self.trait1_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.trait1_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.trait1_entry.insert(0, "7.0")
        self.trait1_entry.grid(row=4, column=2, padx=5, pady=row_padding)
        self.trait1_entry.bind('<Return>', lambda e: self.update_from_entry('trait1'))
        
        # Optimal Trait 2
        ttk.Label(control_frame, text="Optimal Trait 2:").grid(
            row=5, column=0, sticky=tk.W, pady=row_padding)
        self.trait2_scale = ttk.Scale(control_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                      command=self.update_trait2, length=300)
        init_trait2 = float(self.simulator.optimal_traits[1]) if len(self.simulator.optimal_traits) > 1 else 5.0
        self.trait2_scale.set(init_trait2)
        self.trait2_scale.grid(row=5, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.trait2_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.trait2_entry.insert(0, f"{init_trait2:.1f}")
        self.trait2_entry.grid(row=5, column=2, padx=5, pady=row_padding)
        self.trait2_entry.bind('<Return>', lambda e: self.update_from_entry('trait2'))
        
        # Optimal Trait 3
        ttk.Label(control_frame, text="Optimal Trait 3:").grid(
            row=6, column=0, sticky=tk.W, pady=row_padding)
        self.trait3_scale = ttk.Scale(control_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                      command=self.update_trait3, length=300)
        init_trait3 = float(self.simulator.optimal_traits[2]) if len(self.simulator.optimal_traits) > 2 else 8.0
        self.trait3_scale.set(init_trait3)
        self.trait3_scale.grid(row=6, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.trait3_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.trait3_entry.insert(0, f"{init_trait3:.1f}")
        self.trait3_entry.grid(row=6, column=2, padx=5, pady=row_padding)
        self.trait3_entry.bind('<Return>', lambda e: self.update_from_entry('trait3'))
        
        # Simulation speed control
        ttk.Label(control_frame, text="Speed (ms/step):").grid(
            row=7, column=0, sticky=tk.W, pady=row_padding)
        self.speed_scale = ttk.Scale(control_frame, from_=10, to=2000, orient=tk.HORIZONTAL,
                                     command=self.update_speed, length=300)
        self.speed_scale.set(self.tick_ms)
        self.speed_scale.grid(row=7, column=1, sticky=(tk.W, tk.E), padx=10, pady=row_padding)
        self.speed_entry = ttk.Entry(control_frame, width=8, font=('Arial', 20))
        self.speed_entry.insert(0, f"{self.tick_ms}")
        self.speed_entry.grid(row=7, column=2, padx=5, pady=row_padding)
        self.speed_entry.bind('<Return>', lambda e: self.update_from_entry('speed'))
        
        # Bottom-right view selector
        ttk.Label(control_frame, text="Bottom-right view:").grid(
            row=8, column=0, sticky=tk.W, pady=row_padding)
        self.view_selector = ttk.Combobox(control_frame, width=18, state='readonly',
                                          values=["Population Size", "Trait Space 2D", "Trait Space 3D"],
                                          textvariable=self.bottom_right_mode, font=('Arial', 20))
        self.view_selector.grid(row=8, column=1, sticky=tk.W, padx=10, pady=row_padding)
        self.view_selector.bind('<<ComboboxSelected>>', lambda e: self.update_plots())
        self.view_selector.option_add('*TCombobox*Listbox.Font', ('Arial', 20))

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Simulation", 
                                       command=self.toggle_simulation)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Step", command=self.step).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=10)
        
        self.gen_label = ttk.Label(control_frame, text="Generation: 0", font=('Arial', 20, 'bold'))
        self.gen_label.grid(row=10, column=0, columnspan=3, pady=15)
        
        # --- MODIFIED: Plots now go in a frame below the controls ---
        self.figure, self.axes = plt.subplots(2, 3, figsize=(14, 8))
        self.figure.tight_layout(pad=3.0)
        
        canvas_frame = ttk.Frame(self.root)
        # --- MODIFIED: Changed grid to row=1, column=0 ---
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- MODIFIED: Layout weights for vertical stacking ---
        # Column 0 (the only column) expands
        self.root.columnconfigure(0, weight=1) 
        # Row 0 (controls) does not expand
        self.root.rowconfigure(0, weight=0)
        # Row 1 (canvas) expands
        self.root.rowconfigure(1, weight=1)
        
    def update_selection(self, val):
        self.simulator.selection_strength = float(val)
        # Guard during initialization before entry widgets exist
        if hasattr(self, 'selection_entry'):
            self.selection_entry.delete(0, tk.END)
            self.selection_entry.insert(0, f"{float(val):.2f}")
    
    def update_mutation(self, val):
        self.simulator.mutation_rate = float(val)
        if hasattr(self, 'mutation_entry'):
            self.mutation_entry.delete(0, tk.END)
            self.mutation_entry.insert(0, f"{float(val):.2f}")
    
    def update_drift(self, val):
        self.simulator.genetic_drift_strength = float(val)
        if hasattr(self, 'drift_entry'):
            self.drift_entry.delete(0, tk.END)
            self.drift_entry.insert(0, f"{float(val):.2f}")
    
    def update_migration(self, val):
        self.simulator.migration_rate = float(val)
        if hasattr(self, 'migration_entry'):
            self.migration_entry.delete(0, tk.END)
            self.migration_entry.insert(0, f"{float(val):.2f}")
    
    def update_trait1(self, val):
        self.simulator.optimal_traits[0] = float(val)
        if hasattr(self, 'trait1_entry'):
            self.trait1_entry.delete(0, tk.END)
            self.trait1_entry.insert(0, f"{float(val):.1f}")
    
    def update_trait2(self, val):
        if len(self.simulator.optimal_traits) > 1:
            self.simulator.optimal_traits[1] = float(val)
        if hasattr(self, 'trait2_entry'):
            self.trait2_entry.delete(0, tk.END)
            self.trait2_entry.insert(0, f"{float(val):.1f}")
    
    def update_trait3(self, val):
        if len(self.simulator.optimal_traits) > 2:
            self.simulator.optimal_traits[2] = float(val)
        if hasattr(self, 'trait3_entry'):
            self.trait3_entry.delete(0, tk.END)
            self.trait3_entry.insert(0, f"{float(val):.1f}")

    def update_speed(self, val):
        try:
            self.tick_ms = int(float(val))
        except ValueError:
            return
        if hasattr(self, 'speed_entry'):
            self.speed_entry.delete(0, tk.END)
            self.speed_entry.insert(0, f"{self.tick_ms}")
    
    def update_from_entry(self, param_name):
        """Update parameter from entry box"""
        try:
            if param_name == 'selection':
                val = float(self.selection_entry.get())
                val = max(0.0, min(1.0, val))  # Clamp to valid range
                self.selection_scale.set(val)
                self.simulator.selection_strength = val
            elif param_name == 'mutation':
                val = float(self.mutation_entry.get())
                val = max(0.0, min(1.0, val))
                self.mutation_scale.set(val)
                self.simulator.mutation_rate = val
            elif param_name == 'drift':
                val = float(self.drift_entry.get())
                val = max(0.0, min(0.5, val))
                self.drift_scale.set(val)
                self.simulator.genetic_drift_strength = val
            elif param_name == 'migration':
                val = float(self.migration_entry.get())
                val = max(0.0, min(0.3, val))
                self.migration_scale.set(val)
                self.simulator.migration_rate = val
            elif param_name == 'trait1':
                val = float(self.trait1_entry.get())
                val = max(0.0, min(10.0, val))
                self.trait1_scale.set(val)
                self.simulator.optimal_traits[0] = val
            elif param_name == 'trait2':
                val = float(self.trait2_entry.get())
                val = max(0.0, min(10.0, val))
                self.trait2_scale.set(val)
                if len(self.simulator.optimal_traits) > 1:
                    self.simulator.optimal_traits[1] = val
            elif param_name == 'trait3':
                val = float(self.trait3_entry.get())
                val = max(0.0, min(10.0, val))
                self.trait3_scale.set(val)
                if len(self.simulator.optimal_traits) > 2:
                    self.simulator.optimal_traits[2] = val
            elif param_name == 'speed':
                val = int(float(self.speed_entry.get()))
                val = max(10, min(2000, val))
                self.speed_scale.set(val)
                self.tick_ms = val
        except ValueError:
            pass  # Invalid input, ignore
    
    def toggle_simulation(self):
        self.running = not self.running
        if self.running:
            self.start_button.config(text="Pause Simulation")
            self.run_simulation()
        else:
            self.start_button.config(text="Start Simulation")
    
    def step(self):
        self.simulator.simulate_generation()
        self.update_plots()
        self.gen_label.config(text=f"Generation: {self.simulator.generation}")
    
    def run_simulation(self):
        if self.running:
            self.step()
            self.root.after(self.tick_ms, self.run_simulation)
    
    def reset(self):
        self.running = False
        self.start_button.config(text="Start Simulation")
        self.simulator.reset()
        # Recompute initial fitness and seed history so plots aren't empty/flat
        self.simulator.population.calculate_fitness(self.simulator.optimal_traits,
                                                    self.simulator.selection_strength)
        self.simulator.record_statistics()
        self.update_plots()
        self.gen_label.config(text="Generation: 0")
    
    def update_plots(self):
        """Update all plots"""
        for ax in self.axes.flat:
            ax.clear()
        
        history = self.simulator.history
        
        if len(history['generations']) > 0:
            # Plot 1: Mean Fitness over time
            self.axes[0, 0].plot(history['generations'], history['mean_fitness'], 
                                'b-', linewidth=2)
            self.axes[0, 0].set_xlabel('Generation')
            self.axes[0, 0].set_ylabel('Mean Fitness')
            self.axes[0, 0].set_title('Natural Selection: Fitness Evolution')
            self.axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Trait means over time
            colors = ['red', 'green', 'blue']
            for i in range(self.simulator.population.num_traits):
                self.axes[0, 1].plot(history['generations'], history['trait_means'][i],
                                    color=colors[i], label=f'Trait {i+1}', linewidth=2)
                self.axes[0, 1].axhline(y=self.simulator.optimal_traits[i], 
                                       color=colors[i], linestyle='--', alpha=0.5)
            self.axes[0, 1].set_xlabel('Generation')
            self.axes[0, 1].set_ylabel('Mean Trait Value')
            self.axes[0, 1].set_title('Trait Evolution (dashed = optimal)')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Trait variance (Genetic Drift indicator)
            for i in range(self.simulator.population.num_traits):
                self.axes[0, 2].plot(history['generations'], history['trait_stds'][i],
                                    color=colors[i], label=f'Trait {i+1}', linewidth=2)
            self.axes[0, 2].set_xlabel('Generation')
            self.axes[0, 2].set_ylabel('Trait Standard Deviation')
            self.axes[0, 2].set_title('Genetic Drift: Trait Variance')
            self.axes[0, 2].legend()
            self.axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Current trait distribution
            trait_matrix = np.array([org.traits for org in self.simulator.population.organisms])
            for i in range(min(3, self.simulator.population.num_traits)):
                self.axes[1, 0].hist(trait_matrix[:, i], bins=20, alpha=0.5, 
                                    color=colors[i], label=f'Trait {i+1}')
            self.axes[1, 0].set_xlabel('Trait Value')
            self.axes[1, 0].set_ylabel('Frequency')
            self.axes[1, 0].set_title('Current Trait Distribution')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Fitness distribution
            fitnesses = [org.fitness for org in self.simulator.population.organisms]
            self.axes[1, 1].hist(fitnesses, bins=30, color='purple', alpha=0.7, edgecolor='black')
            self.axes[1, 1].set_xlabel('Fitness')
            self.axes[1, 1].set_ylabel('Number of Organisms')
            self.axes[1, 1].set_title('Current Fitness Distribution')
            self.axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Bottom-right view toggle
            mode = self.bottom_right_mode.get() if hasattr(self, 'bottom_right_mode') else 'Population Size'
            # Ensure correct axis type for the selected mode
            def ensure_axis(is3d: bool):
                ax = self.axes[1, 2]
                ax_is_3d = getattr(ax, 'name', '') == '3d'
                if is3d and not ax_is_3d:
                    # replace with 3D axis
                    self.figure.delaxes(ax)
                    self.axes[1, 2] = self.figure.add_subplot(2, 3, 6, projection='3d')
                elif not is3d and ax_is_3d:
                    # replace with 2D axis
                    self.figure.delaxes(ax)
                    self.axes[1, 2] = self.figure.add_subplot(2, 3, 6)

            if mode == 'Population Size':
                ensure_axis(False)
                self.axes[1, 2].plot(history['generations'], history['population_size'],
                                     'g-', linewidth=2)
                self.axes[1, 2].set_xlabel('Generation')
                self.axes[1, 2].set_ylabel('Population Size')
                self.axes[1, 2].set_title('Population Size (Gene Flow Effect)')
                self.axes[1, 2].grid(True, alpha=0.3)
            elif mode == 'Trait Space 2D':
                ensure_axis(False)
                trait_matrix = np.array([org.traits for org in self.simulator.population.organisms])
                fitnesses = [org.fitness for org in self.simulator.population.organisms]
                self.axes[1, 2].scatter(trait_matrix[:, 0], trait_matrix[:, 1],
                                        c=fitnesses, cmap='viridis', s=30, edgecolor='none')
                self.axes[1, 2].set_xlabel('Trait 1')
                self.axes[1, 2].set_ylabel('Trait 2')
                self.axes[1, 2].set_title('Trait Space (colored by fitness)')
                self.axes[1, 2].grid(True, alpha=0.2)
            elif mode == 'Trait Space 3D':
                ensure_axis(True)
                trait_matrix = np.array([org.traits for org in self.simulator.population.organisms])
                fitnesses = [org.fitness for org in self.simulator.population.organisms]
                ax3d = self.axes[1, 2]
                ax3d.scatter(trait_matrix[:, 0], trait_matrix[:, 1], trait_matrix[:, 2],
                             c=fitnesses, cmap='viridis', s=15, depthshade=True)
                ax3d.set_xlabel('Trait 1')
                ax3d.set_ylabel('Trait 2')
                ax3d.set_zlabel('Trait 3')
                ax3d.set_title('3D Trait Space (colored by fitness)')
        
        self.figure.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()