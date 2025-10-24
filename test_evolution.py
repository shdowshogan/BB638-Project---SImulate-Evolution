import numpy as np

print("SIMULATING PRESET 1 OVER 10 GENERATIONS")
print("=" * 60)

optimal = np.array([7.0, 5.0, 8.0])
trait_mean = np.array([5.0, 5.0, 5.0])
pop_size = 200
carrying = 1000
growth = 1.0
death = 0.12
bonus = 0.8
selection = 0.6

for gen in range(10):
    # Calculate fitness
    distance = np.sqrt(np.sum((trait_mean - optimal) ** 2))
    fitness = np.exp(-selection * distance)
    
    # Population dynamics
    crowding = max(0, 1 - (pop_size / carrying))
    births = int(pop_size * growth * fitness * crowding)
    deaths_calc = int(pop_size * death * (1 - fitness * bonus))
    net = births - deaths_calc
    new_pop = max(10, pop_size + net)
    
    print(f"Gen {gen}: Pop={pop_size:3d}, Fitness={fitness:.3f}, "
          f"Distance={distance:.2f}, Births={births:2d}, Deaths={deaths_calc:2d}, Net={net:+3d}")
    
    # Simulate evolution toward optimal
    # Traits move 10% toward optimal each generation (simplified)
    trait_mean = trait_mean + 0.1 * (optimal - trait_mean)
    pop_size = new_pop
    
    if pop_size < 50:
        print("⚠️  POPULATION CRITICALLY LOW!")
        break

print("\nFINAL RESULT:", "SURVIVED ✓" if pop_size >= 50 else "EXTINCT ✗")
