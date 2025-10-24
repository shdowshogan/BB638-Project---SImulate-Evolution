"""
Test Preset 1 over extended evolution to see if fitness reaches 0.85+
"""
import numpy as np

class SimpleOrganism:
    def __init__(self, traits):
        self.traits = np.array(traits, dtype=float)
        
class SimpleSimulator:
    def __init__(self):
        self.optimal = np.array([7.0, 5.0, 8.0])
        self.selection_strength = 0.6
        self.mutation_rate = 0.05
        self.mutation_std = 0.5
        
    def calculate_fitness(self, traits):
        distance = np.sqrt(np.sum((traits - self.optimal) ** 2))
        return np.exp(-self.selection_strength * distance)
    
    def select_parents(self, population):
        """Fitness-weighted selection"""
        fitnesses = [self.calculate_fitness(org.traits) for org in population]
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            # Random selection if all zero fitness
            return np.random.choice(population, size=len(population), replace=True)
        
        probabilities = [f / total_fitness for f in fitnesses]
        return np.random.choice(population, size=len(population), replace=True, p=probabilities)
    
    def mutate(self, traits):
        """Apply mutations"""
        new_traits = traits.copy()
        for i in range(len(new_traits)):
            if np.random.random() < self.mutation_rate:
                new_traits[i] += np.random.normal(0, self.mutation_std)
        return new_traits
    
    def evolve_generation(self, population):
        """Run one generation of evolution"""
        # Select parents
        parents = self.select_parents(population)
        
        # Create offspring with mutation
        new_population = []
        for parent in parents:
            child_traits = self.mutate(parent.traits)
            new_population.append(SimpleOrganism(child_traits))
        
        return new_population

# Run simulation
print("Testing Preset 1: Pure Selection")
print("=" * 60)
print()

sim = SimpleSimulator()

# Initialize population
population = [SimpleOrganism([5.0, 5.0, 5.0]) for _ in range(200)]

# Track fitness over time
generations = [0, 25, 50, 75, 100, 150]
results = []

for gen in range(151):
    if gen in generations:
        # Calculate statistics
        all_traits = [org.traits for org in population]
        mean_traits = np.mean(all_traits, axis=0)
        fitness_values = [sim.calculate_fitness(org.traits) for org in population]
        mean_fitness = np.mean(fitness_values)
        variance = np.mean([np.var([org.traits[i] for org in population]) for i in range(3)])
        
        distance_to_optimal = np.sqrt(np.sum((mean_traits - sim.optimal) ** 2))
        
        results.append({
            'gen': gen,
            'fitness': mean_fitness,
            'variance': variance,
            'distance': distance_to_optimal,
            'traits': mean_traits.copy()
        })
        
        print(f"Generation {gen:3d}:")
        print(f"  Mean Fitness: {mean_fitness:.3f}")
        print(f"  Mean Traits: [{mean_traits[0]:.2f}, {mean_traits[1]:.2f}, {mean_traits[2]:.2f}]")
        print(f"  Distance to Optimal: {distance_to_optimal:.2f}")
        print(f"  Variance: {variance:.3f}")
        print()
    
    # Evolve to next generation
    population = sim.evolve_generation(population)

print("=" * 60)
print("FINAL ASSESSMENT:")
print("=" * 60)
final = results[-1]
print(f"Final Fitness: {final['fitness']:.3f}")
print(f"Final Traits: [{final['traits'][0]:.2f}, {final['traits'][1]:.2f}, {final['traits'][2]:.2f}]")
print(f"Optimal Traits: [7.00, 5.00, 8.00]")
print(f"Distance: {final['distance']:.2f}")
print()

if final['fitness'] >= 0.85:
    print("✅ SUCCESS: Fitness reached target (0.85+)")
elif final['fitness'] >= 0.70:
    print("⚠️  BORDERLINE: Fitness is good but below target")
    print("   Suggestion: Increase selection_strength to 0.7-0.8")
else:
    print("❌ ISSUE: Fitness too low")
    print("   Need stronger selection or longer evolution time")
print()

# Check if variance collapsed as expected
if final['variance'] < 1.0:
    print("✅ Variance properly collapsed (< 1.0)")
else:
    print("⚠️  Variance still high - may need lower mutation rate")
