import numpy as np
import sys
sys.path.insert(0, '.')
from index import EvolutionSimulator

print("=" * 60)
print("TESTING PRESET 3 - GENETIC DRIFT BOTTLENECKS")
print("=" * 60)

# Create simulator and apply Preset 3 settings
sim = EvolutionSimulator(population_size=200)
sim.selection_strength = 0.25
sim.mutation_rate = 0.15
sim.genetic_drift_strength = 0.6  # 60% kill rate
sim.base_growth_rate = 1.5
sim.base_death_rate = 0.1
sim.fitness_survival_bonus = 0.5

print(f"Starting population: {len(sim.population.organisms)}")
print(f"Drift strength: {sim.genetic_drift_strength} (60% kill rate)")
print(f"Bottleneck chance: 30% per generation")
print()

# Run simulation for 50 generations
pop_sizes = []
bottleneck_gens = []

for gen in range(50):
    initial_size = len(sim.population.organisms)
    sim.simulate_generation()
    final_size = len(sim.population.organisms)
    pop_sizes.append(final_size)
    
    # Detect bottleneck (significant population drop)
    if final_size < initial_size * 0.7:
        bottleneck_gens.append(gen)
        print(f"Gen {gen:3d}: 💀 BOTTLENECK! {initial_size} → {final_size} ({final_size/initial_size*100:.0f}%)")
    elif gen % 10 == 0:
        print(f"Gen {gen:3d}: Pop = {final_size}, Fitness = {sim.history['mean_fitness'][-1]:.3f}")

print()
print("=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Total generations: 50")
print(f"Bottleneck events: {len(bottleneck_gens)}")
print(f"Bottleneck frequency: {len(bottleneck_gens)/50*100:.0f}% of generations")
print(f"Min population: {min(pop_sizes)}")
print(f"Max population: {max(pop_sizes)}")
print(f"Mean population: {np.mean(pop_sizes):.0f}")
print(f"Population variance: {np.std(pop_sizes):.0f}")
print()

if len(bottleneck_gens) == 0:
    print("❌ ERROR: NO BOTTLENECKS OCCURRED!")
    print("   The drift mechanism is not working.")
elif len(bottleneck_gens) < 5:
    print("⚠️  WARNING: Too few bottlenecks (expected ~15 in 50 gens)")
elif np.std(pop_sizes) < 100:
    print("⚠️  WARNING: Population too stable (low variance)")
else:
    print("✅ SUCCESS: Dramatic sawtooth pattern confirmed!")
    print(f"   Population crashes and recovers {len(bottleneck_gens)} times")
