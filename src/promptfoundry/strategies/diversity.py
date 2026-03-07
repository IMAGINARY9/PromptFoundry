"""Diversity controls for evolutionary prompt optimization.

This module provides mechanisms to maintain population diversity,
prevent premature convergence, and detect/suppress duplicate prompts.

MVP 3 Feature: Diversity preservation and lineage reporting.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from promptfoundry.core.population import Individual, Population


@dataclass
class DiversityMetrics:
    """Metrics describing population diversity.

    Attributes:
        unique_ratio: Fraction of unique prompt texts (0.0-1.0).
        avg_distance: Average pairwise distance between prompts.
        cluster_count: Number of distinct prompt clusters.
        entropy: Shannon entropy of prompt token distribution.
        duplicate_count: Number of duplicate prompts detected.
    """

    unique_ratio: float = 1.0
    avg_distance: float = 0.0
    cluster_count: int = 1
    entropy: float = 0.0
    duplicate_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary."""
        return {
            "unique_ratio": self.unique_ratio,
            "avg_distance": self.avg_distance,
            "cluster_count": self.cluster_count,
            "entropy": self.entropy,
            "duplicate_count": self.duplicate_count,
        }


@dataclass
class LineageNode:
    """A node in the prompt lineage tree.

    Attributes:
        prompt_id: Unique identifier for the prompt.
        prompt_text: The prompt text content.
        fitness: Fitness score (if evaluated).
        generation: Generation number.
        parent_ids: IDs of parent prompts.
        mutation_operator: Name of mutation operator used.
        children: List of child nodes.
    """

    prompt_id: str
    prompt_text: str
    fitness: float | None = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_operator: str | None = None
    children: list[LineageNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.prompt_id,
            "text_preview": self.prompt_text[:100] + ("..." if len(self.prompt_text) > 100 else ""),
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_operator": self.mutation_operator,
            "children_count": len(self.children),
        }


class DiversityController:
    """Controls population diversity during evolution.

    Provides:
    - Duplicate detection and suppression
    - Diversity measurement
    - Lineage tracking
    - Crowding/niching mechanisms
    """

    def __init__(
        self,
        min_unique_ratio: float = 0.7,
        similarity_threshold: float = 0.85,
        max_cluster_dominance: float = 0.5,
    ) -> None:
        """Initialize the diversity controller.

        Args:
            min_unique_ratio: Minimum fraction of unique prompts required.
            similarity_threshold: Threshold for considering prompts similar (0-1).
            max_cluster_dominance: Maximum fraction of population in one cluster.
        """
        self.min_unique_ratio = min_unique_ratio
        self.similarity_threshold = similarity_threshold
        self.max_cluster_dominance = max_cluster_dominance

        self._seen_texts: set[str] = set()
        self._text_hashes: dict[str, str] = {}  # hash -> prompt_id
        self._lineage: dict[str, LineageNode] = {}  # prompt_id -> node

    def reset(self) -> None:
        """Reset all tracking state."""
        self._seen_texts.clear()
        self._text_hashes.clear()
        self._lineage.clear()

    # =========================================================================
    # Duplicate Detection
    # =========================================================================

    def is_duplicate(self, text: str) -> bool:
        """Check if a prompt text is a duplicate.

        Args:
            text: The prompt text to check.

        Returns:
            True if the text has been seen before.
        """
        normalized = self._normalize_text(text)
        return normalized in self._seen_texts

    def register_prompt(
        self,
        prompt_id: str,
        text: str,
        fitness: float | None = None,
        generation: int = 0,
        parent_ids: list[str] | None = None,
        mutation_operator: str | None = None,
    ) -> bool:
        """Register a prompt and check for duplicates.

        Args:
            prompt_id: Unique identifier.
            text: Prompt text.
            fitness: Optional fitness score.
            generation: Generation number.
            parent_ids: Parent prompt IDs.
            mutation_operator: Name of mutation used.

        Returns:
            True if the prompt is new, False if duplicate.
        """
        normalized = self._normalize_text(text)
        text_hash = self._hash_text(normalized)

        is_new = normalized not in self._seen_texts
        self._seen_texts.add(normalized)
        self._text_hashes[text_hash] = prompt_id

        # Track lineage
        node = LineageNode(
            prompt_id=prompt_id,
            prompt_text=text,
            fitness=fitness,
            generation=generation,
            parent_ids=parent_ids or [],
            mutation_operator=mutation_operator,
        )
        self._lineage[prompt_id] = node

        # Link to parents
        for parent_id in node.parent_ids:
            if parent_id in self._lineage:
                self._lineage[parent_id].children.append(node)

        return is_new

    def get_duplicates_in_population(self, population: Population) -> list[tuple[str, str]]:
        """Find duplicate pairs within a population.

        Args:
            population: Population to analyze.

        Returns:
            List of (prompt_id_1, prompt_id_2) duplicate pairs.
        """
        duplicates = []
        seen: dict[str, str] = {}  # normalized_text -> first prompt_id

        for individual in population.individuals:
            normalized = self._normalize_text(individual.prompt.text)
            if normalized in seen:
                duplicates.append((seen[normalized], individual.id))
            else:
                seen[normalized] = individual.id

        return duplicates

    # =========================================================================
    # Diversity Measurement
    # =========================================================================

    def measure_diversity(self, population: Population) -> DiversityMetrics:
        """Measure diversity metrics for a population.

        Args:
            population: Population to analyze.

        Returns:
            Diversity metrics.
        """
        texts = [ind.prompt.text for ind in population.individuals]
        normalized = [self._normalize_text(t) for t in texts]

        # Unique ratio
        unique_count = len(set(normalized))
        total_count = len(normalized)
        unique_ratio = unique_count / total_count if total_count > 0 else 1.0

        # Duplicate count
        duplicate_count = total_count - unique_count

        # Token entropy
        entropy = self._calculate_token_entropy(texts)

        # Average pairwise distance (sample-based for efficiency)
        avg_distance = self._calculate_avg_distance(texts)

        # Cluster count (simple heuristic based on prefix similarity)
        cluster_count = self._estimate_cluster_count(normalized)

        return DiversityMetrics(
            unique_ratio=unique_ratio,
            avg_distance=avg_distance,
            cluster_count=cluster_count,
            entropy=entropy,
            duplicate_count=duplicate_count,
        )

    def needs_diversity_injection(self, metrics: DiversityMetrics) -> bool:
        """Check if population needs diversity injection.

        Args:
            metrics: Current diversity metrics.

        Returns:
            True if diversity is below threshold.
        """
        return metrics.unique_ratio < self.min_unique_ratio

    # =========================================================================
    # Lineage Tracking
    # =========================================================================

    def get_lineage(self, prompt_id: str) -> LineageNode | None:
        """Get lineage node for a prompt.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            Lineage node or None if not found.
        """
        return self._lineage.get(prompt_id)

    def get_ancestry(self, prompt_id: str, max_depth: int = 10) -> list[LineageNode]:
        """Get ancestor chain for a prompt.

        Args:
            prompt_id: Prompt identifier.
            max_depth: Maximum depth to traverse.

        Returns:
            List of ancestor nodes, oldest first.
        """
        ancestors: list[LineageNode] = []
        current = self._lineage.get(prompt_id)

        depth = 0
        while current and depth < max_depth:
            ancestors.append(current)
            if current.parent_ids:
                parent_id = current.parent_ids[0]  # Follow first parent
                current = self._lineage.get(parent_id)
            else:
                break
            depth += 1

        return list(reversed(ancestors))

    def get_improvement_lineage(
        self,
        population: Population,
        fitness_scores: list[float],
    ) -> list[dict[str, Any]]:
        """Get lineage of improving prompts.

        Args:
            population: Evaluated population.
            fitness_scores: Fitness scores.

        Returns:
            List of improvement records with lineage info.
        """
        improvements = []

        for individual, score in zip(population.individuals, fitness_scores, strict=True):
            node = self._lineage.get(individual.id)
            if not node:
                continue

            # Check if this improved over parent
            for parent_id in node.parent_ids:
                parent_node = self._lineage.get(parent_id)
                if parent_node and parent_node.fitness is not None:
                    if score > parent_node.fitness:
                        improvements.append(
                            {
                                "prompt_id": individual.id,
                                "parent_id": parent_id,
                                "improvement": score - parent_node.fitness,
                                "mutation_operator": node.mutation_operator,
                                "generation": node.generation,
                            }
                        )

        return improvements

    def generate_lineage_report(self, best_individual: Individual) -> dict[str, Any]:
        """Generate a lineage report for the best individual.

        Args:
            best_individual: The best individual found.

        Returns:
            Lineage report dictionary.
        """
        ancestry = self.get_ancestry(best_individual.id)

        return {
            "best_prompt_id": best_individual.id,
            "best_fitness": best_individual.fitness,
            "generation": best_individual.generation,
            "ancestry_length": len(ancestry),
            "ancestry": [node.to_dict() for node in ancestry],
            "mutations_applied": [
                node.mutation_operator
                for node in ancestry
                if node.mutation_operator
            ],
        }

    # =========================================================================
    # Crowding / Niching
    # =========================================================================

    def apply_crowding_penalty(
        self,
        population: Population,
        fitness_scores: list[float],
        penalty_factor: float = 0.1,
    ) -> list[float]:
        """Apply fitness penalty to similar individuals (crowding).

        Reduces fitness of individuals that are too similar to others,
        encouraging diversity.

        Args:
            population: Population to penalize.
            fitness_scores: Original fitness scores.
            penalty_factor: Penalty strength (0-1).

        Returns:
            Adjusted fitness scores.
        """
        adjusted = list(fitness_scores)
        texts = [ind.prompt.text for ind in population.individuals]

        for i, text_i in enumerate(texts):
            similar_count = 0
            for j, text_j in enumerate(texts):
                if i != j:
                    sim = self._calculate_similarity(text_i, text_j)
                    if sim >= self.similarity_threshold:
                        similar_count += 1

            if similar_count > 0:
                penalty = penalty_factor * similar_count / len(texts)
                adjusted[i] = max(0.0, adjusted[i] * (1 - penalty))

        return adjusted

    def select_diverse_subset(
        self,
        population: Population,
        n: int,
        fitness_scores: list[float] | None = None,
    ) -> list[Individual]:
        """Select a diverse subset of individuals.

        Uses a greedy algorithm to maximize diversity while
        optionally considering fitness.

        Args:
            population: Source population.
            n: Number of individuals to select.
            fitness_scores: Optional fitness scores for weighted selection.

        Returns:
            Selected diverse subset.
        """
        if n >= len(population):
            return list(population.individuals)

        selected: list[Individual] = []
        remaining = list(enumerate(population.individuals))

        # Start with best individual if fitness available
        if fitness_scores:
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            selected.append(remaining[best_idx][1])
            del remaining[best_idx]
        elif remaining:
            selected.append(remaining[0][1])
            del remaining[0]

        # Greedily add most different individuals
        while len(selected) < n and remaining:
            best_candidate = None
            best_min_distance = -1.0

            for idx, individual in remaining:
                min_distance = min(
                    self._calculate_similarity(individual.prompt.text, sel.prompt.text)
                    for sel in selected
                )
                # We want LOW similarity (high distance)
                distance = 1 - min_distance

                if distance > best_min_distance:
                    best_min_distance = distance
                    best_candidate = (idx, individual)

            if best_candidate:
                selected.append(best_candidate[1])
                remaining = [(i, ind) for i, ind in remaining if i != best_candidate[0]]

        return selected

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, collapse whitespace, strip
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0-1)."""
        # Use Jaccard similarity on word n-grams
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_avg_distance(
        self,
        texts: Sequence[str],
        sample_size: int = 50,
    ) -> float:
        """Calculate average pairwise distance."""
        import random

        if len(texts) < 2:
            return 0.0

        # Sample pairs for efficiency
        pairs: list[tuple[int, int]] = []
        n = len(texts)
        max_pairs = n * (n - 1) // 2

        if max_pairs <= sample_size:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            while len(pairs) < sample_size:
                i, j = random.sample(range(n), 2)
                if (i, j) not in pairs and (j, i) not in pairs:
                    pairs.append((i, j))

        total_distance = sum(
            1 - self._calculate_similarity(texts[i], texts[j]) for i, j in pairs
        )

        return total_distance / len(pairs) if pairs else 0.0

    def _calculate_token_entropy(self, texts: Sequence[str]) -> float:
        """Calculate Shannon entropy of token distribution."""
        import math

        all_tokens: list[str] = []
        for text in texts:
            all_tokens.extend(self._normalize_text(text).split())

        if not all_tokens:
            return 0.0

        counts = Counter(all_tokens)
        total = len(all_tokens)

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _estimate_cluster_count(self, normalized_texts: Sequence[str]) -> int:
        """Estimate number of distinct clusters."""
        # Simple heuristic: count distinct instruction prefixes
        prefixes: set[str] = set()

        for text in normalized_texts:
            words = text.split()[:5]  # First 5 words
            prefix = " ".join(words)
            prefixes.add(prefix)

        return len(prefixes)
