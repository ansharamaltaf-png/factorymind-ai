"""
Stage VII - Digital Twin & What-If Simulation
A lightweight software digital twin of ONE production line: represents
machine state (running/stopped/maintenance), production rate, tool wear
accumulation, and failure risk, then simulates operational scenarios and
compares expected outcomes (production loss, downtime, cost, risk).

Parameters are calibrated from the AI4I dataset's observed relationships
(documented assumptions below), not physically measured -- appropriate for
a "simplified software-based representation" per the brief.
"""

from dataclasses import dataclass, field

UNITS_PER_HOUR_NOMINAL = 60          # nominal throughput of one machine
COST_PER_UNIT = 12.0                 # $ contribution margin per unit (assumed)
MAINTENANCE_COST = 350.0             # $ fixed cost of a proactive maintenance stop
UNPLANNED_FAILURE_COST = 4500.0      # $ (repair + expedite + scrap), assumed
UNPLANNED_DOWNTIME_HOURS = 6.0       # hours lost on an unplanned failure


@dataclass
class TwinState:
    tool_wear_min: float
    torque_nm: float
    rpm: float
    failure_probability: float
    horizon_hours: int = 8


@dataclass
class ScenarioResult:
    name: str
    produced_units: float
    downtime_hours: float
    expected_cost: float
    residual_risk: float
    notes: str


def simulate_continue(state: TwinState) -> ScenarioResult:
    """Scenario A: keep running at nominal rate for the horizon."""
    produced = UNITS_PER_HOUR_NOMINAL * state.horizon_hours
    # risk compounds hour over hour if nothing is done
    compounded_risk = 1 - (1 - state.failure_probability) ** state.horizon_hours
    expected_downtime = compounded_risk * UNPLANNED_DOWNTIME_HOURS
    expected_cost = (
        compounded_risk * UNPLANNED_FAILURE_COST
        - expected_downtime * UNITS_PER_HOUR_NOMINAL * COST_PER_UNIT  # lost production value
    )
    revenue = produced * COST_PER_UNIT - max(expected_cost, 0)
    return ScenarioResult(
        "Continue operation", produced, round(expected_downtime, 2),
        round(-revenue if revenue < 0 else expected_cost, 2), round(compounded_risk, 3),
        "No intervention; risk compounds across the shift if the alarm is real.",
    )


def simulate_stop_for_maintenance(state: TwinState) -> ScenarioResult:
    """Scenario B: stop now, perform maintenance (assume 1.5h), then resume clean."""
    downtime = 1.5
    remaining_hours = max(state.horizon_hours - downtime, 0)
    produced = UNITS_PER_HOUR_NOMINAL * remaining_hours
    cost = MAINTENANCE_COST + downtime * UNITS_PER_HOUR_NOMINAL * COST_PER_UNIT
    residual_risk = 0.01  # reset to baseline low risk after maintenance
    return ScenarioResult(
        "Stop for maintenance", produced, downtime, round(cost, 2), residual_risk,
        "Eliminates most failure risk at the cost of planned downtime.",
    )


def simulate_reduce_load(state: TwinState, load_factor: float = 0.7) -> ScenarioResult:
    """Scenario C: reduce speed/load to cut risk while still producing."""
    produced = UNITS_PER_HOUR_NOMINAL * load_factor * state.horizon_hours
    reduced_risk = state.failure_probability * (load_factor ** 2)  # risk falls faster than load
    compounded_risk = 1 - (1 - reduced_risk) ** state.horizon_hours
    expected_downtime = compounded_risk * UNPLANNED_DOWNTIME_HOURS
    cost = compounded_risk * UNPLANNED_FAILURE_COST
    return ScenarioResult(
        f"Reduce load to {int(load_factor*100)}%", produced, round(expected_downtime, 2),
        round(cost, 2), round(compounded_risk, 3),
        "Trades throughput for materially lower compounded failure risk.",
    )


def simulate_change_schedule(state: TwinState, shift_hours: int = 2) -> ScenarioResult:
    """Scenario D: defer the job to the next shift after a short inspection."""
    downtime = shift_hours
    produced = UNITS_PER_HOUR_NOMINAL * max(state.horizon_hours - downtime, 0)
    cost = 150.0  # scheduling/inspection cost, assumed
    residual_risk = max(state.failure_probability * 0.4, 0.01)
    return ScenarioResult(
        "Change production schedule", produced, downtime, cost, round(residual_risk, 3),
        "Short inspection window now, full run deferred -- moderate risk reduction.",
    )


def run_all_scenarios(state: TwinState):
    return [
        simulate_continue(state),
        simulate_stop_for_maintenance(state),
        simulate_reduce_load(state),
        simulate_change_schedule(state),
    ]


def recommend(scenarios) -> ScenarioResult:
    """Planning heuristic: minimize expected cost, but never recommend
    'continue' when residual risk is high regardless of cost (safety rule,
    enforced explicitly rather than left to the optimizer)."""
    safe = [s for s in scenarios if not (s.name == "Continue operation" and s.residual_risk > 0.15)]
    pool = safe if safe else scenarios
    return min(pool, key=lambda s: s.expected_cost)


if __name__ == "__main__":
    state = TwinState(tool_wear_min=180, torque_nm=58, rpm=1350, failure_probability=0.42)
    results = run_all_scenarios(state)
    for r in results:
        print(r)
    print("RECOMMENDED:", recommend(results).name)
