import numpy as np

# Optimal traits from code
optimal = np.array([7.0, 5.0, 8.0])
starting = np.array([5.0, 5.0, 5.0])
distance = np.sqrt(np.sum((starting - optimal) ** 2))

print("=" * 60)
print("PRESET ANALYSIS - Population Dynamics Check")
print("=" * 60)
print(f"Optimal traits: {optimal}")
print(f"Starting traits: {starting}")
print(f"Euclidean distance: {distance:.2f}")
print()

# Preset parameters: (name, selection, mutation, drift, migration)
presets = [
    ("Preset 1: Pure Selection", 0.6, 0.05, 0.0, 0.0),
    ("Preset 2: Mutation-Selection", 0.5, 0.3, 0.0, 0.0),
    ("Preset 3: Genetic Drift", 0.4, 0.1, 0.3, 0.0),
    ("Preset 4: Gene Flow", 0.6, 0.1, 0.0, 0.2)
]

# Population dynamics parameters (UPDATED FIXED VALUES)
pop_size = 200
carrying_capacity = 1000
base_growth_rate = 1.0  # FIXED: Further increased
base_death_rate = 0.12  # FIXED: Further reduced
fitness_survival_bonus = 0.8  # FIXED: Further increased

for name, sel_strength, mut_rate, drift, migration in presets:
    print(f"\n{name}")
    print("-" * 60)
    print(f"Selection: {sel_strength}, Mutation: {mut_rate}, Drift: {drift}, Migration: {migration}")
    
    # Calculate starting fitness
    fitness = np.exp(-sel_strength * distance)
    
    # Calculate births
    crowding_factor = max(0, 1 - (pop_size / carrying_capacity))
    birth_rate = base_growth_rate * fitness * crowding_factor
    births = int(pop_size * birth_rate)
    
    # Calculate deaths
    death_rate = base_death_rate * (1 - fitness * fitness_survival_bonus)
    deaths = int(pop_size * death_rate)
    
    # Net change
    net_change = births - deaths
    new_pop = pop_size + net_change
    
    print(f"Starting Fitness: {fitness:.4f}")
    print(f"Crowding Factor: {crowding_factor:.2f}")
    print(f"Birth Rate: {birth_rate:.4f} → Births: {births}")
    print(f"Death Rate: {death_rate:.4f} → Deaths: {deaths}")
    print(f"Net Change: {net_change:+d}")
    print(f"New Population: {new_pop}")
    
    if net_change < -20:
        print(f"⚠️  WARNING: RAPID DECLINE! Population will likely go extinct!")
    elif net_change < 0:
        print(f"⚠️  WARNING: Population declining")
    elif net_change < 10:
        print(f"✓  Population stable/slow growth")
    else:
        print(f"✓✓ Population growing well")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("If populations crash, consider:")
print("1. Reduce base_death_rate (currently 0.2)")
print("2. Increase fitness_survival_bonus (currently 0.5)")
print("3. Start population closer to optimal traits")
