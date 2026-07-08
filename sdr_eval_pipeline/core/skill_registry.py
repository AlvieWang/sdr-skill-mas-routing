"""
SDR Evaluation Pipeline - Skill Registry

The Skill Registry is the central abstraction layer of the SDR framework.
It manages:
- Skill storage and retrieval (inspired by SkillRouter's two-stage retrieve-and-rerank)
- Skill evolution tracking (inspired by SkillOrchestra's split/merge)
- Skill quality gating (inspired by SkillOpt's Selection Gate)

Key design decisions:
- Uses simple TF-IDF + cosine similarity for retrieval (can be replaced with bi-encoder)
- Tracks Beta-Bernoulli posteriors per model per skill
- Supports skill splitting (when variance is high) and merging (when indistinguishable)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np

from .types import Skill, SkillType, EvaluationConfig


class SkillRegistry:
    """
    Central registry for all skills in the SDR system.
    
    Inspired by:
    - SkillRouter: Two-stage retrieve-and-rerank for skill identification
    - SkillOrchestra: Beta-Bernoulli capability modeling + skill evolution
    - SkillOpt: Selection Gate for quality control + text-space optimization
    - Arbor: when_to_apply trigger conditions for on-demand loading
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.skills: dict[str, Skill] = {}
        self._tfidf_index: dict[str, dict[str, float]] = {}  # skill_name -> {term: tfidf}
        self._idf_cache: dict[str, float] = {}
        self._total_documents = 0
        self._evolution_log: list[dict] = []
    
    def register_skill(self, skill: Skill) -> None:
        """Register a new skill in the registry."""
        self.skills[skill.name] = skill
        self._index_skill(skill)
        self._total_documents += 1
        self._update_idf()
    
    def retrieve_top_k(
        self,
        query: str,
        k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Two-stage retrieval: Stage 1 (bi-encoder-like TF-IDF retrieval)
        
        Returns list of (skill_name, similarity_score) sorted by score descending.
        
        Note: In production, this should be replaced with a fine-tuned bi-encoder
        (e.g., 0.6B model as in SkillRouter). Here we use TF-IDF for demonstration.
        """
        query_terms = self._tokenize(query)
        query_vec = self._compute_tfidf_vector(query_terms, is_query=True)
        
        scores = []
        for skill_name, skill_vec in self._tfidf_index.items():
            sim = self._cosine_similarity(query_vec, skill_vec)
            scores.append((skill_name, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def rerank(
        self,
        candidates: list[tuple[str, float]],
        step_context
    ) -> list[tuple[str, float]]:
        """
        Two-stage retrieval: Stage 2 (cross-encoder-like reranking)
        
        Applies additional signals:
        - when_to_apply condition matching (from Arbor)
        - Budget constraints
        - Previous step failure penalty
        """
        reranked = []
        for skill_name, base_score in candidates:
            skill = self.skills.get(skill_name)
            if skill is None:
                continue
            
            # Boost score if when_to_apply condition matches
            condition_boost = self._check_when_to_apply(skill, step_context)
            
            # Penalize if budget is low and skill is expensive
            budget_penalty = 0.0
            if step_context.budget_remaining < 0.3:
                avg_cost = np.mean([
                    skill.cost_profile.get(m, {}).get("avg_tokens", 1000)
                    for m in self.config.models
                ])
                if avg_cost > 2000:
                    budget_penalty = 0.1
            
            # Boost if previous step failed and this skill handles failures
            failure_boost = 0.0
            if step_context.previous_step_failed:
                if "debug" in skill.name.lower() or "verify" in skill.name.lower():
                    failure_boost = 0.15
            
            final_score = base_score + condition_boost - budget_penalty + failure_boost
            reranked.append((skill_name, final_score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def update_skill_posterior(
        self,
        skill_name: str,
        model_id: str,
        success: bool
    ) -> None:
        """Update Beta-Bernoulli posterior after observing an outcome."""
        skill = self.skills.get(skill_name)
        if skill is None:
            return
        skill.update_posterior(model_id, success)
        skill.last_modified_step += 1
    
    def check_skill_evolution(self, skill_name: str, current_step: int) -> Optional[str]:
        """
        Check if a skill needs evolution (split or merge).
        
        Returns: "split", "merge", or None
        """
        skill = self.skills.get(skill_name)
        if skill is None:
            return None
        
        # Check for split: high variance across models
        probs = [skill.success_prob(m) for m in self.config.models]
        if len(probs) > 1:
            variance = np.var(probs)
            if variance > self.config.split_variance_threshold:
                self._evolution_log.append({
                    "step": current_step,
                    "skill": skill_name,
                    "action": "split",
                    "reason": f"high_variance={variance:.3f}",
                    "probs": probs,
                })
                skill.split_count += 1
                return "split"
        
        # Check for merge: indistinguishable from another skill
        for other_name, other_skill in self.skills.items():
            if other_name == skill_name:
                continue
            other_probs = [other_skill.success_prob(m) for m in self.config.models]
            if len(other_probs) == len(probs):
                diff = np.mean(np.abs(np.array(probs) - np.array(other_probs)))
                if diff < self.config.merge_indistinguishable_threshold:
                    self._evolution_log.append({
                        "step": current_step,
                        "skill": skill_name,
                        "action": "merge",
                        "reason": f"indistinguishable_from_{other_name}_diff={diff:.3f}",
                    })
                    skill.merge_count += 1
                    return "merge"
        
        return None
    
    def get_skill_coverage(self) -> float:
        """Fraction of skill types that have at least one registered skill."""
        covered = set()
        for skill in self.skills.values():
            covered.add(skill.skill_type)
        return len(covered) / len(SkillType)
    
    def get_evolution_summary(self) -> dict:
        """Summary of skill evolution events."""
        actions = defaultdict(int)
        for event in self._evolution_log:
            actions[event["action"]] += 1
        return {
            "total_events": len(self._evolution_log),
            "splits": actions["split"],
            "merges": actions["merge"],
            "events": self._evolution_log[-10:],  # Last 10 events
        }
    
    # ============================================================
    # Private methods
    # ============================================================
    
    def _index_skill(self, skill: Skill) -> None:
        """Build TF-IDF index for a skill's description + when_to_apply."""
        text = f"{skill.description} {skill.when_to_apply}"
        terms = self._tokenize(text)
        self._tfidf_index[skill.name] = self._compute_tfidf_vector(terms)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization (lowercase + split on non-alphanumeric)."""
        import re
        tokens = re.findall(r'[a-z_]+', text.lower())
        return tokens
    
    def _compute_tfidf_vector(
        self,
        terms: list[str],
        is_query: bool = False
    ) -> dict[str, float]:
        """Compute TF-IDF vector from term list."""
        tf = defaultdict(int)
        for term in terms:
            tf[term] += 1
        
        total = len(terms) if terms else 1
        vec = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = self._idf_cache.get(term, math.log(max(1, self._total_documents)))
            vec[term] = tf_val * idf_val
        
        return vec
    
    def _update_idf(self) -> None:
        """Update IDF cache after adding a document."""
        doc_freq = defaultdict(int)
        for skill_vec in self._tfidf_index.values():
            for term in skill_vec:
                doc_freq[term] += 1
        
        for term, df in doc_freq.items():
            self._idf_cache[term] = math.log(max(1, self._total_documents / df))
    
    @staticmethod
    def _cosine_similarity(v1: dict, v2: dict) -> float:
        """Cosine similarity between two sparse vectors."""
        if not v1 or not v2:
            return 0.0
        common = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[t] * v2[t] for t in common)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def _check_when_to_apply(self, skill: Skill, step_context) -> float:
        """Check if the step context matches the skill's when_to_apply condition."""
        condition = skill.when_to_apply.lower()
        query = step_context.query.lower()
        
        # Simple keyword matching for demonstration
        keywords = [w for w in condition.split() if len(w) > 3]
        matches = sum(1 for kw in keywords if kw in query)
        if keywords:
            return min(0.2, matches / len(keywords) * 0.2)
        return 0.0
