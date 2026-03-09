# PRD (Draft): Experiment Control Plane Using OSS GrowthBook Backend

## Overview

This proposal describes an engineering-first experimentation architecture designed for privacy-preserving clean rooms. The system uses the open-source GrowthBook backend exclusively as a **control plane for experiment definitions**, while all assignment, joining, and metric computation occur inside clean rooms such as Snowflake Samooha or AWS Clean Rooms.

GrowthBook is treated as a declarative registry of experiment intent, not as an execution, learning, or analytics engine.

---

## Goals

- Enable declarative, versioned experiment definitions using open-source infrastructure  
- Maintain deterministic, auditable, and replayable experiment assignment  
- Preserve strict clean-room privacy and data-boundary guarantees  
- Support partner-verifiable and regulator-friendly experiment logic  
- Integrate cleanly with Snowflake and AWS Clean Rooms  
- Avoid SaaS lock-in and opaque SDK state  

---

## Non-Goals

- Online learning or bandit execution inside GrowthBook  
- Metric computation or statistical analysis inside GrowthBook  
- Dependence on GrowthBook UI or dashboards  
- Per-request SDK-based assignment in production execution paths  

---

## Experiment Definition Model

Experiments are defined centrally in GrowthBook using feature-level definitions and experiment-type rules.

Each experiment definition includes:
- A stable experiment identifier  
- A rule identifier used for versioning  
- Eligibility logic (regex, attribute predicates, or group membership)  
- A privacy-safe hashing attribute  
- Coverage specifying what fraction of eligible users participate  
- A fixed set of variations (e.g., A/B)  
- Static traffic weights  

Definitions are immutable once published and changes result in new versions.

---

## Assignment Semantics

Experiment assignment is executed entirely inside the clean room and is not performed by GrowthBook SDKs in production paths.

Assignment behavior must be:
- Deterministic across time and systems  
- Stateless and recomputable from raw data  
- Independent of runtime SDK memory  
- Fully replayable for audits and partner verification  

Assignment follows a fixed sequence:
1. Evaluate eligibility conditions  
2. Apply coverage gating using deterministic hashing  
3. Assign a variation using stable hashing logic  

---

## Metrics and Measurement

All metrics are computed from clean-room-approved data sources.

The experiment system assumes:
- Exposure and outcome events are logged downstream  
- Metrics such as click-through rate are derived via SQL  
- No raw event data is ingested by GrowthBook  

Metric computation is SQL-first and supports replay, partner validation, and long-term reproducibility.

---

## Versioning and Reset Semantics

An experiment is considered a new version when any of the following change:
- Experiment identifier  
- Rule identifier  
- Hashing attribute  
- Eligibility namespace or prefix  

Versioning guarantees:
- Historical assignments remain reproducible  
- New versions rebucket cleanly  
- Prior analyses remain valid and comparable  

---

## Security and Privacy Considerations

- GrowthBook never ingests raw user or event data  
- No cross-party joins occur outside clean rooms  
- Assignment logic is transparent and inspectable  
- Partners can independently validate experiment logic  
- No hidden or mutable SDK state influences assignment  

This design aligns with clean-room privacy requirements and multi-party measurement standards.

---

## Operational Workflow

- Engineers define experiments using REST or infrastructure-as-code  
- GrowthBook stores versioned experiment definitions  
- Definitions are periodically exported into Snowflake  
- Clean-room SQL executes assignment and metric aggregation  
- Analysis is performed using reproducible SQL workflows  

---

## Summary

This approach repurposes the open-source GrowthBook backend as a declarative experiment control plane while executing all assignment and measurement inside clean rooms. The result is a deterministic, auditable, privacy-preserving experimentation system aligned with modern clean-room and partner-measurement requirements.
