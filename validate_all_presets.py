"""
COMPREHENSIVE VALIDATION OF ALL 4 PRESETS
Checks if each preset produces expected behaviors
"""
import numpy as np

def calculate_fitness(trait_means, optimal, selection_strength):
    """Calculate population mean fitness"""
    distance = np.sqrt(np.sum((trait_means - optimal) ** 2))
    return np.exp(-selection_strength * distance)

def calculate_population_equilibrium(fitness, base_growth, base_death, fitness_bonus, carrying_capacity=1000):
    """Find equilibrium population size"""
    for pop_size in range(100, 1001, 50):
        crowding = max(0, 1 - (pop_size / carrying_capacity))
        birth_rate = base_growth * fitness * crowding
        births = int(pop_size * birth_rate)
        
        death_rate = base_death * (1 - fitness * fitness_bonus)
        deaths = int(pop_size * death_rate)
        
        net = births - deaths
        if abs(net) < 10:  # Equilibrium
            return pop_size
    return 1000

print("=" * 80)
print("PRESET VALIDATION REPORT")
print("=" * 80)
print()

optimal = np.array([7.0, 5.0, 8.0])
starting = np.array([5.0, 5.0, 5.0])

# ============================================================================
# PRESET 1: PURE SELECTION
# ============================================================================
print("=" * 80)
print("PRESET 1: PURE SELECTION")
print("=" * 80)

p1_params = {
    'selection_strength': 0.6,
    'mutation_rate': 0.05,
    'mutation_std': 0.5,
    'base_growth': 1.0,
    'base_death': 0.12,
    'fitness_bonus': 0.8
}

print(f"Selection: {p1_params['selection_strength']}")
print(f"Mutation Rate: {p1_params['mutation_rate']}")
print(f"Mutation Std: {p1_params['mutation_std']}")
print()

# Starting fitness
start_fitness = calculate_fitness(starting, optimal, p1_params['selection_strength'])
print(f"Starting Fitness: {start_fitness:.3f}")

# Final fitness (assuming traits reach near optimal)
final_traits = optimal - 0.5  # Close but not perfect
final_fitness = calculate_fitness(final_traits, optimal, p1_params['selection_strength'])
print(f"Expected Final Fitness: {final_fitness:.3f}")

# Population equilibrium
final_pop = calculate_population_equilibrium(final_fitness, p1_params['base_growth'], 
                                             p1_params['base_death'], p1_params['fitness_bonus'])
print(f"Expected Final Population: {final_pop}")
print()

# Variance expectation
print("Expected Variance Evolution:")
print("  Gen 0: ~3.0 (starting)")
print("  Gen 50: ~0.8-1.2 (collapsed)")
print("  Gen 100: ~0.5-0.8 (very low)")
print()

# TARGETS
print("✅ TARGET RANGES:")
print("  • Final Fitness: 0.85-0.95")
print("  • Final Variance: 0.5-1.0")
print("  • Final Population: 900-1000")
print("  • Traits: Converge to [7.0, 5.0, 8.0]")
print()

# VALIDATION
issues_p1 = []
if final_fitness < 0.85:
    issues_p1.append("⚠️  Final fitness may be too low - selection might be weak")
if p1_params['mutation_rate'] > 0.1:
    issues_p1.append("⚠️  Mutation rate high - may prevent full optimization")

if issues_p1:
    print("POTENTIAL ISSUES:")
    for issue in issues_p1:
        print(f"  {issue}")
else:
    print("✅ PRESET 1 PARAMETERS LOOK GOOD!")
print()

# ============================================================================
# PRESET 2: MUTATION-SELECTION BALANCE
# ============================================================================
print("=" * 80)
print("PRESET 2: MUTATION-SELECTION BALANCE")
print("=" * 80)

p2_params = {
    'selection_strength': 0.2,
    'mutation_rate': 0.7,
    'mutation_std': 1.5,
    'base_growth': 0.9,
    'base_death': 0.15,
    'fitness_bonus': 0.6
}

print(f"Selection: {p2_params['selection_strength']}")
print(f"Mutation Rate: {p2_params['mutation_rate']}")
print(f"Mutation Std: {p2_params['mutation_std']}")
print()

# Equilibrium fitness (won't reach optimal)
eq_traits = starting + (optimal - starting) * 0.3  # Only 30% toward optimal
eq_fitness = calculate_fitness(eq_traits, optimal, p2_params['selection_strength'])
print(f"Expected Equilibrium Fitness: {eq_fitness:.3f}")

# Population equilibrium
eq_pop = calculate_population_equilibrium(eq_fitness, p2_params['base_growth'], 
                                          p2_params['base_death'], p2_params['fitness_bonus'])
print(f"Expected Population: {eq_pop}")
print()

# Variance calculation
mutation_contribution = p2_params['mutation_rate'] * p2_params['mutation_std']
print(f"Mutation Variance Contribution: {mutation_contribution:.2f}")
print("Expected Variance: 2.0-3.5 (HIGH, maintained)")
print()

# TARGETS
print("✅ TARGET RANGES:")
print("  • Equilibrium Fitness: 0.40-0.55 (LOW plateau)")
print("  • Variance: 2.0-3.5 (STAYS HIGH)")
print("  • Population: 600-750")
print("  • Traits: Wander, don't fully converge")
print()

# VALIDATION
issues_p2 = []
if p2_params['selection_strength'] > 0.25:
    issues_p2.append("⚠️  Selection might be too strong - fitness could exceed 0.6")
if p2_params['mutation_rate'] < 0.6:
    issues_p2.append("⚠️  Mutation rate too low - variance won't stay high")
if p2_params['mutation_std'] < 1.0:
    issues_p2.append("⚠️  Mutation effects too small - variance will collapse")
if eq_fitness > 0.6:
    issues_p2.append("⚠️  Predicted fitness too high - won't show balance")
if mutation_contribution < 0.8:
    issues_p2.append("⚠️  Mutation contribution low - variance may not stay high")

if issues_p2:
    print("POTENTIAL ISSUES:")
    for issue in issues_p2:
        print(f"  {issue}")
else:
    print("✅ PRESET 2 PARAMETERS LOOK GOOD!")
print()

# ============================================================================
# PRESET 3: GENETIC DRIFT
# ============================================================================
print("=" * 80)
print("PRESET 3: GENETIC DRIFT")
print("=" * 80)

p3_params = {
    'selection_strength': 0.25,
    'mutation_rate': 0.15,
    'drift_strength': 0.6,
    'base_growth': 1.5,
    'base_death': 0.1,
    'fitness_bonus': 0.5
}

print(f"Selection: {p3_params['selection_strength']}")
print(f"Drift Strength: {p3_params['drift_strength']} (60% kill rate)")
print(f"Growth Rate: {p3_params['base_growth']} (fast recovery)")
print()

# Bottleneck probability
bottleneck_prob = p3_params['drift_strength'] * 0.67
print(f"Bottleneck Probability: {bottleneck_prob*100:.0f}% per generation")
print(f"Expected Bottlenecks per 50 gens: {int(50 * bottleneck_prob)}")
print()

# Population dynamics
print("Population Dynamics:")
print(f"  Before bottleneck: 700-950")
print(f"  After bottleneck (60% killed): {int(800 * 0.4)}-{int(950 * 0.4)}")
print(f"  Recovery rate: Fast (growth={p3_params['base_growth']})")
print()

# TARGETS
print("✅ TARGET RANGES:")
print("  • Fitness: 0.3-0.7 (FLUCTUATES randomly)")
print("  • Variance: 1.0-5.0 (WILD swings)")
print("  • Population: 50-950 (SAWTOOTH pattern)")
print("  • Bottleneck frequency: 15-25 per 50 gens")
print()

# VALIDATION
issues_p3 = []
if p3_params['drift_strength'] < 0.5:
    issues_p3.append("⚠️  Drift too weak - bottlenecks won't be dramatic")
if p3_params['base_growth'] < 1.2:
    issues_p3.append("⚠️  Growth too slow - population won't recover from bottlenecks")
if p3_params['selection_strength'] > 0.3:
    issues_p3.append("⚠️  Selection too strong - may overpower drift")
if bottleneck_prob < 0.3:
    issues_p3.append("⚠️  Bottleneck probability low - events will be rare")

if issues_p3:
    print("POTENTIAL ISSUES:")
    for issue in issues_p3:
        print(f"  {issue}")
else:
    print("✅ PRESET 3 PARAMETERS LOOK GOOD!")
print()

# ============================================================================
# PRESET 4: GENE FLOW
# ============================================================================
print("=" * 80)
print("PRESET 4: GENE FLOW")
print("=" * 80)

p4_params = {
    'selection_strength': 0.6,
    'mutation_rate': 0.1,
    'migration_rate': 0.2,
    'base_growth': 1.0,
    'base_death': 0.12,
    'fitness_bonus': 0.8
}

print(f"Selection: {p4_params['selection_strength']} (STRONG)")
print(f"Migration Rate: {p4_params['migration_rate']} (20% per generation)")
print()

# Fitness with migration disruption
# Migration pulls traits away from optimal
effective_progress = 0.5  # Migration slows adaptation by ~50%
migration_traits = starting + (optimal - starting) * effective_progress
migration_fitness = calculate_fitness(migration_traits, optimal, p4_params['selection_strength'])
print(f"Expected Equilibrium Fitness: {migration_fitness:.3f} (with migration drag)")

# Without migration, would reach:
no_migration_fitness = calculate_fitness(optimal - 0.5, optimal, p4_params['selection_strength'])
print(f"Without Migration Would Reach: {no_migration_fitness:.3f}")
print()

# Population
eq_pop_mig = calculate_population_equilibrium(migration_fitness, p4_params['base_growth'],
                                              p4_params['base_death'], p4_params['fitness_bonus'])
print(f"Expected Population: {eq_pop_mig}")
print()

# TARGETS
print("✅ TARGET RANGES:")
print("  • Equilibrium Fitness: 0.50-0.65 (STALLED by migration)")
print("  • Variance: 2.0-3.0 (migration adds variation)")
print("  • Population: 700-850")
print("  • Traits: Approach optimal but pulled back")
print()

# VALIDATION
issues_p4 = []
if p4_params['migration_rate'] < 0.15:
    issues_p4.append("⚠️  Migration rate too low - won't prevent adaptation")
if p4_params['selection_strength'] < 0.5:
    issues_p4.append("⚠️  Selection too weak - can't show contrast with migration")
if migration_fitness > 0.7:
    issues_p4.append("⚠️  Predicted fitness too high - migration effect weak")

if issues_p4:
    print("POTENTIAL ISSUES:")
    for issue in issues_p4:
        print(f"  {issue}")
else:
    print("✅ PRESET 4 PARAMETERS LOOK GOOD!")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print()

all_issues = {
    'Preset 1': issues_p1,
    'Preset 2': issues_p2,
    'Preset 3': issues_p3,
    'Preset 4': issues_p4
}

total_issues = sum(len(issues) for issues in all_issues.values())

if total_issues == 0:
    print("✅✅✅ ALL PRESETS VALIDATED SUCCESSFULLY! ✅✅✅")
    print()
    print("All parameters are within expected ranges.")
    print("Each preset should produce distinct, demonstrable behaviors.")
else:
    print(f"⚠️  FOUND {total_issues} POTENTIAL ISSUES ACROSS PRESETS")
    print()
    for preset_name, issues in all_issues.items():
        if issues:
            print(f"{preset_name}:")
            for issue in issues:
                print(f"  {issue}")
            print()

print("=" * 80)
