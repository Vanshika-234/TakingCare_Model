"""Agentic AI layer: the 'Discharge Planning Agent'.

Agentic concepts demonstrated
  * Tool use / function calling   - 7 tools wrapping ML, DL, RAG and an action (human-review queue)
  * ReAct loop                    - reason -> act (tool) -> observe -> repeat, bounded by MAX_AGENT_STEPS
  * Planning                      - explicit plan in the system prompt (offline: deterministic planner)
  * Memory                        - per-session case memory + persistent JSONL audit log
  * Reflection / critic           - a separate Safety Reviewer pass can force one revision
  * Guardrails + human-in-loop    - no autonomous care changes; HIGH risk / uncertain / anomalous -> human queue
  * Graceful degradation          - runs with Claude (live) or a rule-based planner (offline)
"""
import json
import re
import time

from config import ARTIFACTS, LLM_MODEL, MAX_AGENT_STEPS
from src.gen_models import LLM, generate_summary
from src.system import TOOL_SCHEMAS, CareSystem

AGENT_SYSTEM = """You are CareCompass, a discharge-planning agent supporting hospital clinicians.
GOAL: for the given patient, produce (1) a risk assessment, (2) a prioritised, guideline-cited action plan.
PLAN (follow in order, skipping nothing): get_patient_record -> predict_readmission_risk -> explain_risk_drivers
-> check_data_anomaly -> find_similar_patients -> search_guidelines (one query per major driver) -> decide on review.
RULES: use tools for every fact; never invent values; cite guidelines as [G#]; do NOT recommend starting/stopping
specific drugs; call flag_for_clinician_review if tier is HIGH, uncertainty > 0.08, or the record is anomalous.
Finish with: RISK SUMMARY, KEY DRIVERS, ACTION PLAN (numbered, cited), and the line
'Decision support only - clinician review required.'"""

CRITIC_SYSTEM = """You are a clinical safety reviewer. Given tool evidence and a draft plan, reply 'APPROVED' if the draft
(a) only states facts present in the evidence, (b) cites guidelines, (c) makes no medication start/stop/dose
recommendation, (d) includes the decision-support disclaimer. Otherwise reply 'REVISE:' plus the specific problems."""

BANNED = re.compile(r"\b(start|stop|discontinue|increase|decrease|double|prescribe)\b.{0,25}\b(insulin|metformin|dose|mg|medication)\b", re.I)

FEATURE_QUERY = {"number_inpatient": "repeated hospitalizations case management", "number_emergency": "repeated hospitalizations case management",
                 "num_medications": "medication reconciliation polypharmacy", "meds_per_day": "medication reconciliation polypharmacy",
                 "A1Cresult": "diabetes glycemic insulin HbA1c", "insulin": "diabetes glycemic insulin", "diabetesMed": "diabetes glycemic",
                 "time_in_hospital": "long length of stay functional status", "discharge_group": "discharge to skilled nursing facility handoff",
                 "age_num": "teach-back health literacy follow-up call"}


DIAG_QUERY = {"circulatory": "heart failure discharge bundle", "respiratory": "COPD pneumonia respiratory admissions",
              "diabetes": "diabetes glycemic insulin HbA1c"}


def safety_review_offline(draft: str, guideline_ids: list[str]) -> str:
    problems = []
    if BANNED.search(draft): problems.append("contains a medication change recommendation")
    if "clinician review required" not in draft: problems.append("missing disclaimer")
    if guideline_ids and not re.search(r"\[G\d+\]", draft): problems.append("no guideline citations")
    return "APPROVED" if not problems else "REVISE: " + "; ".join(problems)


class DischargeAgent:
    def __init__(self, system: CareSystem | None = None):
        self.sys = system or CareSystem()
        self.llm = LLM()
        self.memory: dict[int, dict] = {}                      # session memory of finished cases

    @property
    def mode(self):
        return f"LIVE ({LLM_MODEL})" if self.llm.available else "OFFLINE (rule-based planner)"

    # ------------------------------------------------------------------ public API
    def run(self, patient_id: int, verbose=True) -> dict:
        if patient_id in self.memory:                          # memory hit: don't redo work
            return self.memory[patient_id]
        trace: list[dict] = []
        result = self._run_live(patient_id, trace) if self.llm.available else self._run_offline(patient_id, trace)
        result["trace"], result["mode"] = trace, self.mode
        self.memory[patient_id] = result
        with open(ARTIFACTS / "agent_log.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.strftime("%F %T"), "patient_id": patient_id, "mode": self.mode,
                                "tools": [t["tool"] for t in trace], "flagged": result["flagged"]}) + "\n")
        if verbose:
            self.show(result, patient_id)
        return result

    # ------------------------------------------------------------------ LIVE: Claude drives the tools
    def _run_live(self, pid, trace):
        msgs = [{"role": "user", "content": f"Prepare the discharge plan for patient_id={pid}."}]
        text = self._loop(msgs, trace)
        evidence = json.dumps([{t["tool"]: t["result"]} for t in trace])[:6000]
        verdict = self.llm.complete(CRITIC_SYSTEM, f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{text}", 200)
        trace.append({"tool": "safety_reviewer", "args": {}, "result": verdict})
        if not verdict.strip().upper().startswith("APPROVED"):          # reflection: one revision
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": f"Reviewer feedback: {verdict}\nRevise accordingly."}]
            text = self._loop(msgs, trace)
        flagged = any(t["tool"] == "flag_for_clinician_review" for t in trace)
        facts = self._facts_from_trace(trace)
        patient_msg = generate_summary(self.llm, facts, facts["guidelines"], "patient") if facts else ""
        return {"clinician_note": text, "patient_message": patient_msg, "flagged": flagged}

    def _loop(self, msgs, trace):
        for _ in range(MAX_AGENT_STEPS):                                  # bounded loop = guardrail
            r = self.llm.client.messages.create(model=LLM_MODEL, max_tokens=1500, system=AGENT_SYSTEM,
                                                tools=TOOL_SCHEMAS, messages=msgs)
            msgs.append({"role": "assistant", "content": r.content})
            calls = [b for b in r.content if b.type == "tool_use"]
            if not calls:
                return "".join(b.text for b in r.content if b.type == "text")
            results = []
            for c in calls:
                out = self.sys.call(c.name, c.input)
                trace.append({"tool": c.name, "args": c.input, "result": out})
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out)})
            msgs.append({"role": "user", "content": results})
        return "Step limit reached - escalating to a human clinician."

    # ------------------------------------------------------------------ OFFLINE: deterministic planner
    def _run_offline(self, pid, trace):
        def act(tool, **args):
            out = self.sys.call(tool, args)
            trace.append({"tool": tool, "args": args, "result": out})
            return out
        rec = act("get_patient_record", patient_id=pid)
        pred = act("predict_readmission_risk", patient_id=pid)
        expl = act("explain_risk_drivers", patient_id=pid)
        anom = act("check_data_anomaly", patient_id=pid)
        sim = act("find_similar_patients", patient_id=pid)
        # only material drivers (> 0.5 pp of risk) generate retrieval queries; diagnosis drivers use the patient's actual group
        queries = []
        for d in expl["drivers"]:
            if d["effect_on_risk"] > 0.005:
                q = DIAG_QUERY.get(rec["diag_group"]) if d["feature"] == "diag_group" else FEATURE_QUERY.get(d["feature"])
                if q and q not in queries:
                    queries.append(q)
        queries = queries[:3]
        queries += ["follow-up call early clinic visit teach-back"]
        seen, guides = set(), []
        for q in queries:
            for g in act("search_guidelines", query=q, k=2)["results"]:
                if g["source"] not in seen:
                    seen.add(g["source"]); guides.append(g)
        guides = guides[:4]
        facts = {"risk": pred["risk_30d"], "tier": pred["tier"], "uncertainty": pred["uncertainty"],
                 "drivers": expl["drivers"][:4], "segment_rate": sim["neighbor_readmission_rate"], "guidelines": guides}
        must_flag = pred["tier"] == "HIGH" or pred["uncertainty"] > 0.08 or "atypical" in anom["interpretation"]
        flagged = False
        if must_flag:
            why = f"{pred['tier']} risk ({pred['risk_30d']:.1%}), uncertainty {pred['uncertainty']:.1%}, {anom['interpretation']}"
            act("flag_for_clinician_review", patient_id=pid, reason=why, urgency="urgent" if pred["tier"] == "HIGH" else "routine")
            flagged = True
        note = generate_summary(self.llm, facts, guides, "clinician")
        verdict = safety_review_offline(note, [g["source"] for g in guides])
        trace.append({"tool": "safety_reviewer", "args": {}, "result": verdict})
        if verdict != "APPROVED":
            note += "\nDecision support only - clinician review required."
        return {"clinician_note": note, "patient_message": generate_summary(self.llm, facts, guides, "patient"), "flagged": flagged}

    @staticmethod
    def _facts_from_trace(trace):
        g = {t["tool"]: t["result"] for t in trace if isinstance(t["result"], dict)}
        if "predict_readmission_risk" not in g or "explain_risk_drivers" not in g:
            return None
        p = g["predict_readmission_risk"]
        guides = [x for t in trace if t["tool"] == "search_guidelines" for x in t["result"].get("results", [])][:4]
        return {"risk": p["risk_30d"], "tier": p["tier"], "uncertainty": p["uncertainty"], "drivers": g["explain_risk_drivers"]["drivers"][:4],
                "segment_rate": g.get("find_similar_patients", {}).get("neighbor_readmission_rate", float("nan")), "guidelines": guides}

    @staticmethod
    def show(res, pid):
        print(f"\n{'=' * 78}\nPATIENT {pid}   |   agent mode: {res['mode']}\n{'=' * 78}")
        print("AGENT TRACE:", " -> ".join(t["tool"] for t in res["trace"]))
        print(f"HUMAN REVIEW FLAGGED: {res['flagged']}\n\n--- CLINICIAN NOTE ---\n{res['clinician_note']}")
        print(f"\n--- PATIENT MESSAGE ---\n{res['patient_message']}\n")
