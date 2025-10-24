"""
Test Preset 2: Mutation-Selection Balance
Verify that variance stays HIGH and fitness stays LOW
"""
import numpy as np

# Simulate mutation-selection balance dynamics
print("=" * 70)
print("TESTING PRESET 2: MUTATION-SELECTION BALANCE (UPDATED)")
print("=" * 70)

# NEW Parameters
selection_strength = 0.2
mutation_rate = 0.7
mutation_std = 1.5

print(f"Selection strength: {selection_strength}")
print(f"Mutation rate: {mutation_rate} (70% of organisms mutate!)")
print(f"Mutation std dev: {mutation_std} (large trait changes)")
print()

# Simulate trait evolution over generations
optimal = np.array([7.0, 5.0, 8.0])
population_size = 500
trait_means = np.array([5.0, 5.0, 5.0])  # Starting traits
trait_vars = np.array([3.0, 3.0, 3.0])   # Starting variance

print(f"{'Gen':<6} {'Fitness':<10} {'Distance':<12} {'Variance':<12} {'Notes':<30}")
print("-" * 70)

for gen in range(100):
    # Calculate mean distance and fitness
    distance = np.sqrt(np.sum((trait_means - optimal) ** 2))
    fitness = np.exp(-selection_strength * distance)
    
    # Selection pulls traits toward optimal
    selection_pull = (optimal - trait_means) * selection_strength * 0.3
    
    # Mutation pushes traits randomly (increases variance)
    mutation_push = np.random.normal(0, mutation_std * mutation_rate, 3)
    
    # Update traits
    trait_means += selection_pull + mutation_push
    
    # Variance dynamics
    # Selection reduces variance
    trait_vars *= (1 - selection_strength * 0.1)
    # Mutation increases variance
    trait_vars += mutation_rate * mutation_std * 0.5
    
    mean_var = np.mean(trait_vars)
    
    if gen % 10 == 0:
        status = ""
        if mean_var > 2.0:
            status = "✅ High variance (good!)"
        elif mean_var > 1.5:
            status = "⚠️  Moderate variance"
        else:
            status = "❌ Variance too low"
            
        print(f"{gen:<6} {fitness:.4f}    {distance:>6.2f}      {mean_var:>6.2f}      {status}")

print()
print("=" * 70)
print("EXPECTED RESULTS:")
print("=" * 70)
print(f"Final fitness: {fitness:.3f}")
print(f"Final variance: {np.mean(trait_vars):.2f}")
print()

if fitness < 0.6 and np.mean(trait_vars) > 2.0:
    print("✅ SUCCESS! Parameters produce mutation-selection balance:")
    print("   • Fitness plateaus LOW (< 0.6)")
    print("   • Variance stays HIGH (> 2.0)")
elif fitness < 0.6:
    print("⚠️  FITNESS is good but VARIANCE too low")
    print("   Need even STRONGER mutations")
elif np.mean(trait_vars) > 2.0:
    print("⚠️  VARIANCE is good but FITNESS too high")
    print("   Need WEAKER selection or STRONGER mutations")
else:
    print("❌ FAILED - Neither fitness nor variance in target range")
    print("   Parameters need adjustment")

print()
print("TARGET FOR PRESET 2:")
print("  • Fitness: 0.4-0.6 (LOW plateau)")
print("  • Variance: 2.0-3.0 (HIGH maintained)")
print("=" * 70)
