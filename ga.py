import random
import numpy as np
import math

# === Parameters ===
POP_SIZE = 50
GENES = 8
GENERATIONS = 30
ELITE_COUNT = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
INITIAL_MUTATION_STD = 0.5  # High initial value
MUTATION_DECAY = 0.95       # Decay factor per generation
INITIAL_SBX_ETA = 2
SBX_ETA_GROWTH = 1.05       # SBX eta increases each generation

EMERGENCE_IMPORTANCE=0.5
# === Fixed input individuals for blend ===
INPUT_A = [0.9, 0.9, 0.0, 0.8, 0.2, 0.9, 0.7, 0.7]
INPUT_B = [0.8, 0.2, 1.0, 0.3, 0.9, 0.4, 0.6, 0.3]
# Random Parent A
# INPUT_A = [0.57, 0.84, 0.13, 0.95, 0.23, 0.47, 0.61, 0.30]

# # Random Parent B
# INPUT_B = [0.10, 0.72, 0.38, 0.66, 0.89, 0.05, 0.77, 0.48]




def coherence_interval(candidate):
    total, norm = 0.0, 0.0
    for i in range(GENES):
        for j in range(i+1, GENES):
            low  = min(INPUT_A[i], INPUT_B[i], INPUT_A[j], INPUT_B[j])
            high = max(INPUT_A[i], INPUT_B[i], INPUT_A[j], INPUT_B[j])
            # distance outside [low, high]
            dist_i = max(0, low - candidate[i], candidate[i] - high)
            dist_j = max(0, low - candidate[j], candidate[j] - high)
            penalty = dist_i + dist_j
            weight  = 1.0  # or any scheme
            total += weight * (1 - penalty)
            norm  += weight
    return total / norm
# === Fitness Function ===

def shannon_entropy(p: float) -> float:
    """H(p) = –p log p – (1–p) log(1–p), with 0·log0→0."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p*math.log(p) - (1-p)*math.log(1-p)

def joint_entropy_pairwise(p_i: float, p_j: float) -> float:
    """
    Approximate H(X_i,X_j) by assuming P(1,1)=min(p_i,p_j),
    then P(1,0)=p_i–P11, P(0,1)=p_j–P11, P(0,0)=rest.
    """
    p11 = min(p_i, p_j)
    p10 = p_i - p11
    p01 = p_j - p11
    p00 = 1 - p_i - p_j + p11
    H = 0.0
    for p in (p00, p01, p10, p11):
        if p > 0:
            H -= p * math.log(p)
    return H

def coherence_entropy(candidate: list[float]) -> float:
    """
    Entropy-based coherence ≈ [∑ H(p_i) – H_joint] / ∑ H(p_i),
    where H_joint ≈ average of all pairwise joint entropies.
    """
    n = len(candidate)
    # 1) individual entropies
    Hs = [shannon_entropy(p) for p in candidate]
    sum_H = sum(Hs)
    if sum_H == 0:
        return 0.0

    # 2) approximate joint entropy via averaging pairwise H
    pair_H = []
    for i in range(n):
        for j in range(i+1, n):
            pair_H.append(joint_entropy_pairwise(candidate[i], candidate[j]))
    # average pairwise H
    H_joint_approx = sum(pair_H) * 2 / (n * (n-1))

    # 3) total correlation ≈ sum_H – H_joint_approx
    TC = sum_H - H_joint_approx

    # 4) normalized coherence in [0,1]
    return TC / sum_H

def fitness(candidate):
    INPUT_A = [0.9, 0.9, 0.0, 0.8, 0.2, 0.9, 0.7, 0.7]
    INPUT_B = [0.8, 0.2, 1.0, 0.3, 0.9, 0.4, 0.6, 0.3]
    # Random Parent A
#     INPUT_A = [0.57, 0.84, 0.13, 0.95, 0.23, 0.47, 0.61, 0.30]

# # Random Parent B
#     INPUT_B = [0.10, 0.72, 0.38, 0.66, 0.89, 0.05, 0.77, 0.48]
    emergence = [c - max(a, b) for c, a, b in zip(candidate, INPUT_A, INPUT_B)]
    emergence = [max(0, e) for e in emergence]  # clamp negative emergence to 0
    contributions = [min(a, b) * e for a, b, e in zip(INPUT_A, INPUT_B, emergence)]
    total = sum(contributions)
    norm_total=min(total / GENES, 1.0)
    # coh=coherence_entropy(candidate)
    coh=coherence_interval(candidate)
    
    fit=(EMERGENCE_IMPORTANCE*norm_total )+ ((1-EMERGENCE_IMPORTANCE) * coh)

    return fit



# === Initialization ===
def initialize_population():
    return [np.random.uniform(0, 1, GENES).tolist() for _ in range(POP_SIZE)]

# === Selection: Roulette-Wheel via Stochastic Acceptance ===
def roulette_stochastic_acceptance(population, fitnesses):
    w_max = max(fitnesses)
    while True:
        i = random.randint(0, len(population) - 1)
        if random.random() < fitnesses[i] / w_max:
            return population[i]

def roulette_wheel(population, fitnesses):
    total_f = sum(fitnesses)
    r = random.uniform(0, total_f)
    cum = 0.0
    for individual, fit in zip(population, fitnesses):
        cum += fit
        if cum >= r:
            return individual


#try to implement this in metta using non - detreministic methods if not change the selection mechanisms to tournament



# === Crossover: Simulated Binary Crossover (Adaptive Eta) ===
def sbx_crossover(p1, p2, eta):
    if random.random() > CROSSOVER_RATE:
        return p1[:], p2[:]

    child1, child2 = [], []
    for x1, x2 in zip(p1, p2):
        if random.random() <= 0.5:
            if abs(x1 - x2) > 1e-14:
                x1, x2 = min(x1, x2), max(x1, x2)
                rand = random.random()
                beta = 1.0 + (2.0 * (x1) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
                c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
                child1.append(min(max(c1, 0.0), 1.0))
                child2.append(min(max(c2, 0.0), 1.0))
            else:
                child1.append(x1)
                child2.append(x2)
        else:
            child1.append(x1)
            child2.append(x2)
    return child1, child2



#only works if all the genes have the same valid range of values cause it just mixes and matches, might not be favorable in this case 
def single_point_crossover(p1, p2):
    # With probability 1 – CROSSOVER_RATE we skip crossover entirely
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    
    # Choose a crossover point somewhere between 1 and len–1
    point = random.randint(1, len(p1) - 1)
    
    # Build children by swapping tails
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

# === Mutation ===
def mutate(individual, mutation_std):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] += random.gauss(0, mutation_std)
            individual[i] = min(max(individual[i], 0), 1)  # clip to [0, 1]
    return individual

# === Main GA Loop ===
# def genetic_algorithm():
#     population = initialize_population()
#     mutation_std = INITIAL_MUTATION_STD
#     sbx_eta = INITIAL_SBX_ETA

#     for gen in range(GENERATIONS):
#         fitnesses = [fitness(ind) for ind in population]
#         elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:ELITE_COUNT]
#         new_population = [ind for ind, _ in elites]

#         while len(new_population) < POP_SIZE:
#             parent1 = roulette_wheel(population, fitnesses)
#             parent2 = roulette_wheel(population, fitnesses)
#             child1, child2 = single_point_crossover(parent1, parent2)
#             new_population.extend([mutate(child1, mutation_std), mutate(child2, mutation_std)])

#         population = new_population[:POP_SIZE]  # Ensure population size remains constant
#         best_fitness = max(fitnesses)
#         print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f} | Mutation STD = {mutation_std:.4f} | SBX_ETA = {sbx_eta:.2f}")

#         mutation_std *= MUTATION_DECAY  # decay mutation over time
#         sbx_eta *= SBX_ETA_GROWTH       # increase SBX eta over time (more conservative)

#     # Final result
#     final_fitnesses = [fitness(ind) for ind in population]
#     best = max(zip(population, final_fitnesses), key=lambda x: x[1])
#     print("\nBest Individual:", best[0])
#     print("Best Fitness:", best[1])


def genetic_algorithm():
    population = initialize_population()
    mutation_std = INITIAL_MUTATION_STD
    sbx_eta = INITIAL_SBX_ETA

    for gen in range(GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]
        elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:ELITE_COUNT]
        new_population = [ind for ind, _ in elites]

        while len(new_population) < POP_SIZE:
            parent1 = roulette_stochastic_acceptance(population, fitnesses)
            parent2 = roulette_stochastic_acceptance(population, fitnesses)
            child1, child2 = sbx_crossover(parent1, parent2, eta=sbx_eta)
            new_population.extend([mutate(child1, mutation_std), mutate(child2, mutation_std)])

        population = new_population[:POP_SIZE]  # Ensure population size remains constant
        best_fitness = max(fitnesses)
        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f} | Mutation STD = {mutation_std:.4f} | SBX_ETA = {sbx_eta:.2f}")

        mutation_std *= MUTATION_DECAY  # decay mutation over time
        sbx_eta *= SBX_ETA_GROWTH       # increase SBX eta over time (more conservative)

    # Final result
    final_fitnesses = [fitness(ind) for ind in population]
    best = max(zip(population, final_fitnesses), key=lambda x: x[1])
    print("\nBest Individual:", best[0])
    print("Best Fitness:", best[1])

if __name__ == "__main__":
    genetic_algorithm()



#inital final candidate result when only emergence is considered: [1.0, 1.0, 0.9718061708943715, 1.0, 1.0, 1.0, 1.0, 1.0]

#a final candidate result when emergence and coherence are conisered with equal weight: [0.6706819125867991, 0.8051632611881101, 1, 0.7138530102011784, 0.594894535363899, 0.81549557983781, 0.8910415494970053, 0.7768066005701759]

 