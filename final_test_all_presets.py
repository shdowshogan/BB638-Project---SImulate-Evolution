"""
FINAL COMPREHENSIVE TEST: Run all 4 presets and verify expected behaviors
This gives you a quick check before your presentation
"""
import numpy as np

class SimpleOrganism:
    def __init__(self, traits):
        self.traits = np.array(traits, dtype=float)

def calculate_fitness(traits, optimal, selection):
    distance = np.sqrt(np.sum((traits - optimal) ** 2))
    return np.exp(-selection * distance)

def select_parents(population, optimal, selection):
    fitnesses = [calculate_fitness(org.traits, optimal, selection) for org in population]
    total = sum(fitnesses)
    if total == 0:
        return np.random.choice(population, size=len(population), replace=True)
    probs = [f/total for f in fitnesses]
    return np.random.choice(population, size=len(population), replace=True, p=probs)

def mutate(traits, rate, std):
    new_traits = traits.copy()
    for i in range(len(new_traits)):
        if np.random.random() < rate:
            new_traits[i] += np.random.normal(0, std)
    return new_traits

def apply_drift(population, strength):
    """Apply bottleneck - kills fraction of population"""
    if np.random.random() < (strength * 0.67):
        # Bottleneck event!
        kill_fraction = strength
        survivors_count = max(50, int(len(population) * (1 - kill_fraction)))
        survivors = np.random.choice(population, size=survivors_count, replace=False)
        return list(survivors), True
    return population, False

def add_migrants(population, rate):
    """Replace some population with random immigrants"""
    num_migrants = int(len(population) * rate)
    if num_migrants > 0:
        # Remove random individuals
        survivors = list(np.random.choice(population, size=len(population)-num_migrants, replace=False))
        # Add random migrants
        for _ in range(num_migrants):
            random_traits = np.random.uniform(0, 10, size=3)
            survivors.append(SimpleOrganism(random_traits))
        return survivors
    return population

def run_preset(name, params, generations=100):
    """Run a preset simulation"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    optimal = np.array([7.0, 5.0, 8.0])
    population = [SimpleOrganism([5.0, 5.0, 5.0]) for _ in range(200)]
    
    bottleneck_count = 0
    
    # Run evolution
    for gen in range(generations + 1):
        # Calculate stats every 25 gens
        if gen % 25 == 0 or gen == generations:
            all_traits = [org.traits for org in population]
            mean_traits = np.mean(all_traits, axis=0)
            fitnesses = [calculate_fitness(org.traits, optimal, params['selection']) 
                        for org in population]
            mean_fitness = np.mean(fitnesses)
            variance = np.mean([np.var([org.traits[i] for org in population]) 
                              for i in range(3)])
            
            if gen == 0:
                print(f"Gen {gen:3d}: Fitness={mean_fitness:.3f}, Var={variance:.2f}, Pop={len(population)}")
            elif gen == generations:
                print(f"Gen {gen:3d}: Fitness={mean_fitness:.3f}, Var={variance:.2f}, Pop={len(population)}")
                print()
                return {
                    'fitness': mean_fitness,
                    'variance': variance,
                    'population': len(population),
                    'bottlenecks': bottleneck_count
                }
        
        # Selection
        parents = select_parents(population, optimal, params['selection'])
        
        # Reproduction with mutation
        new_population = []
        for parent in parents:
            child_traits = mutate(parent.traits, params['mutation_rate'], params['mutation_std'])
            new_population.append(SimpleOrganism(child_traits))
        
        # Drift
        if params.get('drift', 0) > 0:
            new_population, had_bottleneck = apply_drift(new_population, params['drift'])
            if had_bottleneck:
                bottleneck_count += 1
        
        # Migration
        if params.get('migration', 0) > 0:
            new_population = add_migrants(new_population, params['migration'])
        
        population = new_population

print("="*70)
print("EVOLUTION SIMULATOR - FINAL COMPREHENSIVE TEST")
print("="*70)
print("Testing all 4 presets to verify expected behaviors")
print("="*70)

# PRESET 1: Pure Selection
results_1 = run_preset(
    "PRESET 1: PURE SELECTION",
    {
        'selection': 0.6,
        'mutation_rate': 0.05,
        'mutation_std': 0.5
    },
    generations=100
)

# PRESET 2: Mutation-Selection Balance
results_2 = run_preset(
    "PRESET 2: MUTATION-SELECTION BALANCE",
    {
        'selection': 0.2,
        'mutation_rate': 0.7,
        'mutation_std': 1.5
    },
    generations=100
)

# PRESET 3: Genetic Drift
results_3 = run_preset(
    "PRESET 3: GENETIC DRIFT",
    {
        'selection': 0.25,
        'mutation_rate': 0.15,
        'mutation_std': 0.5,
        'drift': 0.6
    },
    generations=50  # Shorter - just need to see bottlenecks
)

# PRESET 4: Gene Flow
results_4 = run_preset(
    "PRESET 4: GENE FLOW",
    {
        'selection': 0.6,
        'mutation_rate': 0.1,
        'mutation_std': 0.5,
        'migration': 0.2
    },
    generations=100
)

# SUMMARY
print("="*70)
print("SUMMARY: VALIDATION CHECKLIST")
print("="*70)
print()

# Preset 1 checks
print("PRESET 1: PURE SELECTION")
if results_1['fitness'] >= 0.75:
    print("  ✅ High fitness achieved (≥0.75)")
else:
    print(f"  ❌ Fitness too low: {results_1['fitness']:.3f}")

if results_1['variance'] < 1.0:
    print("  ✅ Low variance (genetic variation collapsed)")
else:
    print(f"  ⚠️  Variance still high: {results_1['variance']:.2f}")
print()

# Preset 2 checks
print("PRESET 2: MUTATION-SELECTION BALANCE")
if 0.40 <= results_2['fitness'] <= 0.60:
    print(f"  ✅ Fitness in target range: {results_2['fitness']:.3f}")
else:
    print(f"  ⚠️  Fitness outside target: {results_2['fitness']:.3f} (want 0.40-0.60)")

if results_2['variance'] >= 2.0:
    print(f"  ✅ High variance maintained: {results_2['variance']:.2f}")
else:
    print(f"  ❌ Variance too low: {results_2['variance']:.2f} (want ≥2.0)")
print()

# Preset 3 checks
print("PRESET 3: GENETIC DRIFT")
if results_3['bottlenecks'] >= 10:
    print(f"  ✅ Sufficient bottlenecks: {results_3['bottlenecks']} in 50 gens")
else:
    print(f"  ⚠️  Few bottlenecks: {results_3['bottlenecks']} (want 10-25 per 50 gens)")

if results_3['variance'] >= 1.0:
    print(f"  ✅ Variance shows fluctuation: {results_3['variance']:.2f}")
else:
    print(f"  ⚠️  Variance low: {results_3['variance']:.2f}")
print()

# Preset 4 checks
print("PRESET 4: GENE FLOW")
if 0.45 <= results_4['fitness'] <= 0.70:
    print(f"  ✅ Fitness stalled by migration: {results_4['fitness']:.3f}")
else:
    print(f"  ⚠️  Fitness outside expected: {results_4['fitness']:.3f} (want 0.45-0.70)")

if results_4['variance'] >= 1.5:
    print(f"  ✅ Variance elevated by migrants: {results_4['variance']:.2f}")
else:
    print(f"  ⚠️  Variance low: {results_4['variance']:.2f}")
print()

# Overall assessment
print("="*70)
all_good = (
    results_1['fitness'] >= 0.75 and results_1['variance'] < 1.0 and
    0.40 <= results_2['fitness'] <= 0.60 and results_2['variance'] >= 2.0 and
    results_3['bottlenecks'] >= 10 and
    0.45 <= results_4['fitness'] <= 0.70
)

if all_good:
    print("✅✅✅ ALL PRESETS VALIDATED - READY FOR PRESENTATION! ✅✅✅")
else:
    print("⚠️  SOME PRESETS SHOW UNEXPECTED BEHAVIOR")
    print("    This may be due to random variation in single runs.")
    print("    Check full simulation with GUI for complete verification.")

print("="*70)
