"""A generic genetic algorithm over fixed-length integer chromosomes (fitness is maximised)."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Chromosome = list[int]
Fitness = Callable[[Sequence[int]], float]


@dataclass(frozen=True)
class GAResult:
    best: tuple[int, ...]
    fitness: float
    generations: int


@dataclass(frozen=True)
class GeneticAlgorithm:
    """Tournament selection, uniform crossover, random-reset mutation and elitism.

    Elitism copies the best ``elite`` individuals unchanged into the next generation, so the
    best fitness can never get worse from one generation to the next.
    """

    population_size: int
    chromosome_length: int
    alleles: int
    mutation_rate: float
    tournament_size: int = 5
    elite: int = 2

    def __post_init__(self) -> None:
        if self.elite >= self.population_size:
            raise ValueError("elite must be smaller than population_size")

    def run(self, fitness: Fitness, rng: random.Random, *, target: float = float("inf"),
            max_generations: int = 1_000) -> GAResult:
        population = [[rng.randrange(self.alleles) for _ in range(self.chromosome_length)]
                      for _ in range(self.population_size)]
        scores = [fitness(c) for c in population]
        generation = 0
        while generation < max_generations and max(scores) < target:
            population = self._next_generation(population, scores, rng)
            scores = [fitness(c) for c in population]
            generation += 1
        best = max(range(len(scores)), key=scores.__getitem__)
        return GAResult(tuple(population[best]), scores[best], generation)

    def _next_generation(self, population: list[Chromosome], scores: list[float],
                         rng: random.Random) -> list[Chromosome]:
        ranked = sorted(range(len(population)), key=scores.__getitem__, reverse=True)
        nxt = [population[i][:] for i in ranked[: self.elite]]
        while len(nxt) < self.population_size:
            a = self._tournament(population, scores, rng)
            b = self._tournament(population, scores, rng)
            child = [x if rng.random() < 0.5 else y for x, y in zip(a, b, strict=True)]
            nxt.append([rng.randrange(self.alleles) if rng.random() < self.mutation_rate else g for g in child])
        return nxt

    def _tournament(self, population: list[Chromosome], scores: list[float], rng: random.Random) -> Chromosome:
        contenders = [rng.randrange(len(population)) for _ in range(self.tournament_size)]
        return population[max(contenders, key=scores.__getitem__)]


def match_target(target: Sequence[int]) -> Fitness:
    """Fitness = number of genes equal to the target sequence."""
    target = tuple(target)
    return lambda chromosome: sum(g == t for g, t in zip(chromosome, target, strict=True))


def decode(bits: Sequence[int], lo: float, hi: float) -> float:
    """Binary chromosome (most significant bit first) -> real number in [lo, hi]."""
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return lo + (hi - lo) * value / ((1 << len(bits)) - 1)
