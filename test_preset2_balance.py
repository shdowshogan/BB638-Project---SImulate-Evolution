"""
Test Preset 2 to verify mutation-selection balance produces LOW fitness and HIGH variance
"""
import numpy as np

class SimpleOrganism:
    def __init__(self, traits):
        self.traits = np.array(traits, dtype=float)
        
class SimpleSimulator:
    def __init__(self):
        self.optimal = np.array([7.0, 5.0, 8.0])
        self.selection_strength = 0.2  # WEAK
        self.mutation_rate = 0.7       # VERY HIGH
        self.mutation_std = 1.5        # LARGE EFFECTS
        
    def calculate_fitness(self, traits):
        distance = np.sqrt(np.sum((traits - self.optimal) ** 2))
        return np.exp(-self.selection_strength * distance)
    
    def select_parents(self, population):
        """Fitness-weighted selection"""
        fitnesses = [self.calculate_fitness(org.traits) for org in population]
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            return np.random.choice(population, size=len(population), replace=True)
        
        probabilities = [f / total_fitness for f in fitnesses]
        return np.random.choice(population, size=len(population), replace=True, p=probabilities)
    
    def mutate(self, traits):
        """Apply mutations - 70% rate with large effects"""
        new_traits = traits.copy()
        for i in range(len(new_traits)):
            if np.random.random() < self.mutation_rate:
                new_traits[i] += np.random.normal(0, self.mutation_std)
        return new_traits
    
    def evolve_generation(self, population):
        """Run one generation"""
        parents = self.select_parents(population)
        new_population = []
        for parent in parents:
            child_traits = self.mutate(parent.traits)
            new_population.append(SimpleOrganism(child_traits))
        return new_population

# Run simulation
print("Testing Preset 2: Mutation-Selection Balance")
print("=" * 60)
print("Parameters:")
print("  Selection: 0.2 (WEAK)")
print("  Mutation Rate: 0.7 (70% of organisms mutate)")
print("  Mutation Std: 1.5 (LARGE effect sizes)")
print()

sim = SimpleSimulator()
population = [SimpleOrganism([5.0, 5.0, 5.0]) for _ in range(200)]

generations = [0, 10, 25, 50, 75, 100]
results = []

for gen in range(101):
    if gen in generations:
        all_traits = [org.traits for org in population]
        mean_traits = np.mean(all_traits, axis=0)
        fitness_values = [sim.calculate_fitness(org.traits) for org in population]
        mean_fitness = np.mean(fitness_values)
        
        # Calculate variance per trait, then average
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
    
    population = sim.evolve_generation(population)

print("=" * 60)
print("FINAL ASSESSMENT:")
print("=" * 60)
final = results[-1]
print(f"Final Fitness: {final['fitness']:.3f}")
print(f"Final Variance: {final['variance']:.3f}")
print(f"Final Traits: [{final['traits'][0]:.2f}, {final['traits'][1]:.2f}, {final['traits'][2]:.2f}]")
print()

# Check if showing mutation-selection balance
if final['fitness'] < 0.55:
    print("✅ Fitness LOW as expected (< 0.55)")
elif final['fitness'] < 0.65:
    print("⚠️  Fitness slightly high but acceptable")
else:
    print("❌ Fitness too high - selection overpowering mutation")
    print("   Need: stronger mutation OR weaker selection")
print()

if final['variance'] >= 2.0:
    print("✅ Variance HIGH as expected (≥ 2.0)")
elif final['variance'] >= 1.5:
    print("⚠️  Variance moderately high")
else:
    print("❌ Variance too low - mutations not strong enough")
    print("   Need: higher mutation_rate OR larger mutation_std")
print()

# Check equilibrium
if len(results) >= 3:
    recent_fitness = [r['fitness'] for r in results[-3:]]
    fitness_change = max(recent_fitness) - min(recent_fitness)
    if fitness_change < 0.05:
        print("✅ Fitness stabilized (equilibrium reached)")
    else:
        print("⚠️  Fitness still changing - may need more generations")
