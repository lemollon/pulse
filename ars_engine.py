# ars_engine.py
import math, time, datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class TemplateArm:
    name: str
    def constraint_penalty(self, context: Dict) -> float:
        hour = int(context.get("hour_local", 12))
        ch   = context.get("channel_pref", "email")
        pen = 0.0
        if ch == "sms" and hour < 9: pen += 1.5
        if context.get("weekend") and ch == "email": pen += 0.2
        return pen
    def instantiate_sequence(self, lead: Dict, context: Dict) -> List[Dict]:
        today = dt.date.fromisoformat(context["today"])
        day2  = today + dt.timedelta(days=2)
        day7  = today + dt.timedelta(days=7)
        nm    = lead.get("name","there")
        return [
            {"send_dt": str(today), "channel":"email", "subject":"Thanks for reaching out", "body": f"Hi {nm}, great to connect..."},
            {"send_dt": str(day2),  "channel":"sms",   "body": f"Hi {nm}, quick check-in..."},
            {"send_dt": str(day7),  "channel":"email", "subject":"Next steps", "body": "Following up on our conversation..."},
        ]

class AdaptiveRevenueSequencer:
    def __init__(self, arms: List[TemplateArm], priors: Dict[str, float]):
        self.arms = arms
        self.priors = priors
    def featurize(self, lead: Dict, context: Dict) -> Dict[str, float]:
        hour = float(context.get("hour_local", 12))
        return {
            "pref_email": 1.0 if lead.get("channel_pref") == "email" else 0.0,
            "pref_sms":   1.0 if lead.get("channel_pref") == "sms" else 0.0,
            "weekend":    1.0 if context.get("weekend") else 0.0,
            "morning":    1.0 if 5 <= hour <= 11 else 0.0,
            "holiday":    1.0 if context.get("holiday_flag") else 0.0,
            "sent7":      float(context.get("avg_sentiment_7d", 0.0)),
            "wait_compl": float(context.get("complaint_wait_time", 0.0)),
            "recency":    float(context.get("recency_days", 3.0)),
            "prior_rr":   float(context.get("prior_reply_rate", 0.1)),
        }
    def score(self, arm: TemplateArm, feats: Dict[str, float], weights: Dict[str, float], context: Dict) -> float:
        lin  = sum(weights.get(k, 0.0) * v for k,v in feats.items())
        base = self.priors.get(arm.name, 0.0)
        pen  = arm.constraint_penalty(context)
        return lin + base - pen
    def ucb_pick(self, candidates: List[str], stats: Dict[str, Dict]) -> str:
        now = int(time.time())
        best_name, best_val = candidates[0], float("-inf")
        for name in candidates:
            s = stats.get(name, {"n": 0, "reward_sum": 0.0})
            n = max(1, s["n"]); mean = s["reward_sum"]/n
            bonus = math.sqrt(2 * math.log(max(2, now)) / n)
            val = mean + bonus
            if val > best_val:
                best_name, best_val = name, val
        return best_name
    def plan(self, lead: Dict, context: Dict, cohort_weights: Dict[str, Dict[str, float]], arm_stats: Dict[str, Dict]) -> Tuple[List[Dict], str]:
        feats = self.featurize(lead, context)
        scored = []
        for arm in self.arms:
            w = cohort_weights.get(arm.name, {})
            s = self.score(arm, feats, w, context)
            scored.append((arm, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        topK = [a.name for a,_ in scored[:5]]
        chosen_name = self.ucb_pick(topK, arm_stats)
        chosen = next(a for a,_ in scored if a.name == chosen_name)
        return chosen.instantiate_sequence(lead, context), chosen_name
