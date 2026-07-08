"""
SDR Evaluation Pipeline - Metric A: Skill-Level Routing Accuracy

Implements routing accuracy metrics at the skill level, inspired by SkillRouter.
Measures how accurately the router identifies the required skills and selects
the correct model for each skill.

Metrics:
- Skill Hit@1: Is the top-1 predicted skill correct?
- Skill MRR@10: Reciprocal rank of the first correct skill in top-10
- Skill Recall@K: Fraction of ground-truth skills in top-K
- Skill FC@10: Does top-10 contain all ground-truth skills?
- Skill Conditioned Routing Accuracy: Given correct skill, is model correct?
"""

from __future__ import annotations

from typing import List

from core.types import (
    Trajectory, MetricResult, RoutingDecision, StepContext,
    EvaluationConfig
)


class RoutingAccuracyMetrics:
    """Category A: Skill-level routing accuracy metrics."""
    
    CATEGORY = "A"
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(self, trajectories: List[Trajectory]) -> List[MetricResult]:
        results = []
        
        # Collect all (decision, step_context) pairs
        all_decisions = []
        all_contexts = []
        for traj in trajectories:
            for decision, ctx in zip(traj.decisions, traj.steps):
                all_decisions.append(decision)
                all_contexts.append(ctx)
        
        # A1: Skill Hit@1
        hit1 = self._skill_hit_at_1(all_decisions, all_contexts)
        results.append(MetricResult(
            name="Skill Hit@1",
            value=hit1,
            category=self.CATEGORY,
            description="Top-1 predicted skill matches ground truth",
            baseline_value=0.70,  # Rubar's coarse condition matching
            detail={"source": "SkillRouter", "n_samples": len(all_decisions)},
        ))
        
        # A2: Skill MRR@10
        mrr = self._skill_mrr_at_10(all_decisions, all_contexts)
        results.append(MetricResult(
            name="Skill MRR@10",
            value=mrr,
            category=self.CATEGORY,
            description="Reciprocal rank of first correct skill in top-10",
            baseline_value=None,
            detail={"source": "SkillRouter"},
        ))
        
        # A3: Skill Recall@K (K=10)
        recall = self._skill_recall_at_k(all_decisions, all_contexts, k=10)
        results.append(MetricResult(
            name="Skill Recall@10",
            value=recall,
            category=self.CATEGORY,
            description="Fraction of ground-truth skills recovered in top-10",
            baseline_value=None,
            detail={"source": "SkillRouter"},
        ))
        
        # A4: Skill FC@10
        fc = self._skill_fc_at_k(all_decisions, all_contexts, k=10)
        results.append(MetricResult(
            name="Skill FC@10",
            value=fc,
            category=self.CATEGORY,
            description="Top-10 contains ALL ground-truth skills (multi-skill steps)",
            baseline_value=None,
            detail={"source": "SkillRouter"},
        ))
        
        # A5: Skill-Conditioned Routing Accuracy
        scra = self._skill_conditioned_routing_accuracy(trajectories)
        results.append(MetricResult(
            name="Skill-Conditioned Routing Accuracy",
            value=scra,
            category=self.CATEGORY,
            description="Given correct skill, model selection is correct",
            baseline_value=0.80,  # Rubar 7-dim matrix
            detail={"source": "SkillOrchestra Beta-Bernoulli"},
        ))
        
        return results
    
    def _skill_hit_at_1(
        self,
        decisions: List[RoutingDecision],
        contexts: List[StepContext]
    ) -> float:
        """Fraction of steps where top-1 predicted skill is in ground truth."""
        if not decisions:
            return 0.0
        hits = 0
        total = 0
        for dec, ctx in zip(decisions, contexts):
            if not ctx.gt_skills:
                continue
            total += 1
            if dec.predicted_skills and dec.predicted_skills[0] in ctx.gt_skills:
                hits += 1
        return hits / total if total > 0 else 0.0
    
    def _skill_mrr_at_10(
        self,
        decisions: List[RoutingDecision],
        contexts: List[StepContext]
    ) -> float:
        """Mean Reciprocal Rank of first correct skill in top-10."""
        if not decisions:
            return 0.0
        total_rr = 0.0
        count = 0
        for dec, ctx in zip(decisions, contexts):
            if not ctx.gt_skills or not dec.predicted_skills:
                continue
            count += 1
            top_k = dec.predicted_skills[:10]
            for rank, skill in enumerate(top_k, 1):
                if skill in ctx.gt_skills:
                    total_rr += 1.0 / rank
                    break
        return total_rr / count if count > 0 else 0.0
    
    def _skill_recall_at_k(
        self,
        decisions: List[RoutingDecision],
        contexts: List[StepContext],
        k: int = 10
    ) -> float:
        """Fraction of ground-truth skills recovered in top-K."""
        if not decisions:
            return 0.0
        total_recall = 0.0
        count = 0
        for dec, ctx in zip(decisions, contexts):
            if not ctx.gt_skills:
                continue
            count += 1
            top_k = set(dec.predicted_skills[:k])
            gt = set(ctx.gt_skills)
            if gt:
                total_recall += len(top_k & gt) / len(gt)
        return total_recall / count if count > 0 else 0.0
    
    def _skill_fc_at_k(
        self,
        decisions: List[RoutingDecision],
        contexts: List[StepContext],
        k: int = 10
    ) -> float:
        """Fraction of steps where top-K contains ALL ground-truth skills."""
        if not decisions:
            return 0.0
        full_coverage = 0
        count = 0
        for dec, ctx in zip(decisions, contexts):
            if not ctx.gt_skills:
                continue
            count += 1
            top_k = set(dec.predicted_skills[:k])
            gt = set(ctx.gt_skills)
            if gt.issubset(top_k):
                full_coverage += 1
        return full_coverage / count if count > 0 else 0.0
    
    def _skill_conditioned_routing_accuracy(
        self,
        trajectories: List[Trajectory]
    ) -> float:
        """Given correct skill, is the model selection correct (vs oracle)?"""
        correct = 0
        total = 0
        for traj in trajectories:
            for dec, result, ctx in zip(traj.decisions, traj.results, traj.steps):
                if ctx.gt_model and ctx.gt_skills:
                    total += 1
                    if dec.selected_model == ctx.gt_model:
                        correct += 1
                    elif result.success:
                        correct += 0.5  # Partial credit for correct outcome
        return correct / total if total > 0 else 0.0
