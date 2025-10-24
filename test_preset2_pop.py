"""
Test Preset 2 Population Dynamics
Check if population stabilizes at 500-700 instead of 900+
"""
import numpy as np

print("=" * 70)
print("PRESET 2: POPULATION DYNAMICS TEST")
print("=" * 70)

# Updated parameters
base_growth = 0.9
base_death = 0.15
fitness_bonus = 0.6
carrying_capacity = 1000

# Assume equilibrium fitness around 0.45 (mutation-selection balance)
fitness_levels = [0.3, 0.4, 0.5, 0.6]

print(f"Growth rate: {base_growth}")
print(f"Death rate: {base_death}")
print(f"Fitness bonus: {fitness_bonus}")
print()

print(f"{'Fitness':<10} {'Pop Size':<12} {'Births':<10} {'Deaths':<10} {'Net':<10} {'Stable?':<10}")
print("-" * 70)

for fitness in fitness_levels:
    # Test different population sizes to find equilibrium
    for pop_size in [400, 500, 600, 700, 800, 900]:
        crowding = max(0, 1 - (pop_size / carrying_capacity))
        birth_rate = base_growth * fitness * crowding
        births = int(pop_size * birth_rate)
        
        death_rate = base_death * (1 - fitness * fitness_bonus)
        deaths = int(pop_size * death_rate)
        
        net = births - deaths
        
        # Equilibrium is when net change ~ 0
        if abs(net) < 10:
            stable = "✅ STABLE"
            print(f"{fitness:<10.2f} {pop_size:<12} {births:<10} {deaths:<10} {net:+<10} {stable}")
            break

print()
print("=" * 70)
print("EXPECTED EQUILIBRIUM:")
print("  At fitness 0.4-0.5 → Population should stabilize at 600-750")
print("  At fitness 0.3 → Population lower (~500)")
print("  At fitness 0.6 → Population higher (~800)")
print()
print("With mutation-selection balance (fitness ~0.45), expect pop ~650-700")
print("=" * 70)
