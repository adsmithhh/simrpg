from __future__ import annotations

import yaml
import pygame
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import itertools
import os
import json
from datetime import datetime

# --- Load YAML config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "int5.yaml")
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = "int5.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"int5.yaml not found in {os.path.dirname(__file__)} or current directory")

with open(CONFIG_PATH, "r", encoding='utf-8-sig') as f:
    config = yaml.safe_load(f)
CONFIG = config

# Validate and normalize anchors structure
if 'anchors' in config:
    fixed_anchors = {}
    for name, anchor_config in config['anchors'].items():
        if isinstance(anchor_config, dict) and 'norm' in anchor_config:
            fixed_anchors[name] = anchor_config['norm']
        elif isinstance(anchor_config, list) and len(anchor_config) == 2:
            fixed_anchors[name] = anchor_config
        else:
            raise ValueError(f"Invalid anchor format for {name}: {anchor_config}")
    config['anchors'] = fixed_anchors

WIDTH = config["display"]["width"]
HEIGHT = config["display"]["height"]
FPS = config["display"]["fps"]

ANCHORS_NORM = config["anchors"]

ZONE_BASES_NORM = {
    "TERRITORY_1": [0.20, 0.35],
    "TERRITORY_2": [0.80, 0.35],
    "TERRITORY_3": [0.50, 0.80],
}

COLORS = {k: tuple(v) for k, v in config["COLORS"].items()}
BLACK = COLORS["BLACK"]
WHITE = COLORS["WHITE"]
TEAL = COLORS["TEAL"]
ORANGE = COLORS["ORANGE"]
BLUE = COLORS["BLUE"]
PURPLE = COLORS["PURPLE"]
RED_ORANGE = COLORS["RED_ORANGE"]
GREEN = COLORS["GREEN"]
LIME = COLORS["LIME"]
GREEN_TEXT = COLORS["GREEN_TEXT"]
RED_TEXT = COLORS["RED_TEXT"]
YELLOW_TEXT = COLORS["YELLOW_TEXT"]
TRADE_COLOR = COLORS["TRADE_COLOR"]
DEV_COLOR = COLORS["DEV_COLOR"]
FLEX_COLOR = COLORS["FLEX_COLOR"]

PERSONALITY_COLORS = {
    "Anchor": (80, 180, 255),
    "Climber": (255, 180, 80),
    "Connector": (180, 80, 255),
    "Survivor": (80, 255, 120),
}

SHIFT_DURATION = 300
TRAVEL_RATIO = 0.25
TRAVEL_BUDGET = int(SHIFT_DURATION * TRAVEL_RATIO)
WORK_BUDGET = SHIFT_DURATION - TRAVEL_BUDGET
DISS_THRESHOLD = 120.0
WITHDRAWAL_GAP = 7

color_map = {
    "SCIENCE": (100, 200, 255),
    "TRADE": TRADE_COLOR,
    "DEVELOPMENT": DEV_COLOR,
    "FLEX": FLEX_COLOR,
}
INSPECTOR_ZONE_ORDER = ["SCIENCE", "TRADE", "DEVELOPMENT", "FLEX"]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("System Architect - Trust System v2.0")
clock = pygame.time.Clock()

font_header = pygame.font.Font(None, config["fonts"]["header"])
font_large = pygame.font.Font(None, config["fonts"]["large"])
font_small = pygame.font.Font(None, config["fonts"]["small"])
font_tiny = pygame.font.Font(None, config["fonts"]["tiny"])

NPC_ID_GEN = itertools.count(1)
global_tick = 0

ZONE_TRAVEL_COSTS = {
    "SCIENCE": {"energy_cost_per_tick": 0.15, "money_cost_per_tick": 0.05, "description": "To research labs"},
    "TRADE": {"energy_cost_per_tick": 0.10, "money_cost_per_tick": 0.03, "description": "To marketplace"},
    "DEVELOPMENT": {"energy_cost_per_tick": 0.20, "money_cost_per_tick": 0.08, "description": "To industrial zone"},
    "FLEX": {"energy_cost_per_tick": 0.05, "money_cost_per_tick": 0.02, "description": "To flexible workspace"},
    "CENTRAL": {"energy_cost_per_tick": 0.08, "money_cost_per_tick": 0.02, "description": "To central hub"},
    "HOME": {"energy_cost_per_tick": 0.08, "money_cost_per_tick": 0.02, "description": "Returning home"},
}

ZONES = []
for zone_name, zone_cfg in config["zones"].items():
    color = COLORS[zone_cfg["color"]]
    ZONES.append({
        "name": zone_name,
        "color": color,
        "npc_count": zone_cfg.get("npc_count", 3),
        "description": zone_cfg.get("description", "")
    })

@dataclass
class UIState:
    show_statistics: bool = False
    show_npc_inspector: bool = False
    show_combined_view: bool = False
    show_monotony_panel: bool = False  # <-- Add this
    inspector_zone_idx: int = 0
    inspector_page: int = 0  # <-- Add this
    current_world_idx: int = 0

@dataclass
class Commitment:
    key: str
    created_tick: int
    due_tick: int
    strength: float = 0.5
    honored: bool = False
    broken: bool = False
    break_reason: str = ""

@dataclass
class DecisionRecord:
    tick: int
    choice: str
    phase: str

@dataclass
class ZoneStats:
    name: str
    total_work_done: int = 0
    total_npcs_served: int = 0
    current_population: int = 0
    congestion_level: float = 0.0
    market_demand: float = 1.0
    efficiency_rating: float = 1.0
    total_money_generated: float = 0.0
    total_stress_absorbed: float = 0.0
    history_population: deque = field(default_factory=lambda: deque(maxlen=100))
    history_efficiency: deque = field(default_factory=lambda: deque(maxlen=100))

    def update_metrics(self, population: int):
        self.current_population = population
        self.history_population.append(population)
        optimal_population = 5
        self.congestion_level = min(1.0, population / (optimal_population * 2))
        self.efficiency_rating = 1.0 / (1.0 + self.congestion_level * 0.5)
        self.history_efficiency.append(self.efficiency_rating)
        self.market_demand = 0.8 + random.random() * 0.4

@dataclass(frozen=True)
class ZoneStatsView:
    name: str
    current_population: int
    congestion_level: float
    market_demand: float
    efficiency_rating: float
    total_work_done: int
    total_money_generated: float
    total_stress_absorbed: float

@dataclass
class GameEvent:
    name: str
    description: str
    zone_affected: Optional[str]
    duration: int
    effect_type: str
    money_multiplier: float = 1.0
    stress_multiplier: float = 1.0
    energy_multiplier: float = 1.0
    remaining_ticks: int = 0

EVENT_TEMPLATES = [
    {"name": "Market Boom", "zone": "TRADE", "effect": "bonus", "money_mult": 1.5, "duration": 200, "desc": "Trade profits increased!"},
    {"name": "Research Grant", "zone": "SCIENCE", "effect": "bonus", "money_mult": 1.3, "duration": 150, "desc": "Science funding boost!"},
    {"name": "Equipment Failure", "zone": "DEVELOPMENT", "effect": "penalty", "stress_mult": 1.4, "duration": 100, "desc": "Dev zone disrupted!"},
    {"name": "Wellness Program", "zone": "FLEX", "effect": "bonus", "stress_mult": 0.7, "duration": 180, "desc": "Flex zone enhanced!"},
    {"name": "Energy Crisis", "zone": None, "effect": "penalty", "energy_mult": 1.3, "duration": 120, "desc": "Energy costs increased!"},
]

@dataclass
class World:
    npcs: List["NPC"] = field(default_factory=list)
    zone_stats: Dict[str, ZoneStats] = field(default_factory=dict)
    active_events: List[GameEvent] = field(default_factory=list)
    name: str = ""

    def spawn_random_event(self):
        if random.random() < 0.01:
            template = random.choice(EVENT_TEMPLATES)
            event = GameEvent(
                name=template["name"],
                description=template["desc"],
                zone_affected=template.get("zone"),
                duration=template["duration"],
                effect_type=template["effect"],
                money_multiplier=template.get("money_mult", 1.0),
                stress_multiplier=template.get("stress_mult", 1.0),
                energy_multiplier=template.get("energy_mult", 1.0),
                remaining_ticks=template["duration"]
            )
            self.active_events.append(event)

    def update_events(self):
        self.active_events = [e for e in self.active_events if e.remaining_ticks > 0]
        for event in self.active_events:
            event.remaining_ticks -= 1

    def step(self, global_tick):
        self.update_events()
        self.spawn_random_event()
        buffers = simtick(self, global_tick)

@dataclass(frozen=True)
class WorldSnapshot:
    tick: int
    npc: Dict[int, "NPCView"]
    zone_density: Dict[str, int]
    zone_stats: Dict[str, "ZoneStatsView"]
    active_events: Tuple["GameEvent", ...]

def anchors_to_pixels(anchors_norm, width, height):
    return {name: (int(x * width), int(y * height)) for name, (x, y) in anchors_norm.items()}

def build_zone_density(npcs) -> Dict[str, int]:
    dens = {z: 0 for z in ["SCIENCE", "TRADE", "DEVELOPMENT", "FLEX"]}
    for npc in npcs:
        if npc.state == "AT_WORK" and npc.zone in dens:
            dens[npc.zone] += 1
    return dens

def update_zone_stats_mutable(zone_stats_mutable: Dict[str, "ZoneStats"], zone_density: Dict[str, int]):
    for zone_name, pop in zone_density.items():
        if zone_name not in zone_stats_mutable:
            zone_stats_mutable[zone_name] = ZoneStats(name=zone_name)
        zone_stats_mutable[zone_name].update_metrics(pop)

def freeze_zone_stats_views(zone_stats_mutable: Dict[str, "ZoneStats"]) -> Dict[str, ZoneStatsView]:
    out: Dict[str, ZoneStatsView] = {}
    for zn, zs in zone_stats_mutable.items():
        out[zn] = ZoneStatsView(
            name=zs.name,
            current_population=zs.current_population,
            congestion_level=zs.congestion_level,
            market_demand=zs.market_demand,
            efficiency_rating=zs.efficiency_rating,
            total_work_done=zs.total_work_done,
            total_money_generated=zs.total_money_generated,
            total_stress_absorbed=zs.total_stress_absorbed,
        )
    return out

@dataclass(frozen=True)
class NPCView:
    id: int
    x: float
    y: float
    state: str
    zone: str
    target: Optional[str]
    next_intended_zone: Optional[str]
    travel_budget: int
    work_budget: int
    shift_offset: int
    speed: float
    stress_endured: float
    money: float
    energy: float
    is_collapsing: bool
    skills: Dict[str, float]
    zone_visit_count: Dict[str, int]
    zone_fatigue: Dict[str, int]  # <-- Add this line


    @staticmethod
    def from_npc(npc: "NPC") -> "NPCView":
        return NPCView(
            id=npc.id,
            x=npc.x,
            y=npc.y,
            state=npc.state,
            zone=npc.zone,
            target=npc.target,
            speed=npc.speed,
            next_intended_zone=npc.next_intended_zone,
            travel_budget=npc.travel_budget,
            work_budget=npc.work_budget,
            shift_offset=npc.shift_offset,
            stress_endured=npc.stress_endured,
            money=npc.money,
            energy=npc.energy,
            is_collapsing=npc.is_collapsing,
            skills=dict(npc.skills),
            zone_visit_count=dict(npc.zone_visit_count),
            zone_fatigue=dict(npc.zone_fatigue),  # <-- Add this line
        )

def build_snapshot(world, tick: int) -> WorldSnapshot:
    zone_density = build_zone_density(world.npcs)
    update_zone_stats_mutable(world.zone_stats, zone_density)
    zone_stats_view = freeze_zone_stats_views(world.zone_stats)
    npc_view = {npc.id: NPCView.from_npc(npc) for npc in world.npcs}
    return WorldSnapshot(
        tick=tick,
        npc=npc_view,
        zone_density=zone_density,
        zone_stats=zone_stats_view,
        active_events=tuple(world.active_events),
    )

@dataclass
class NPC:
    x: float
    y: float
    zone: str = ""
    state: str = "AT_HOME"
    target: Optional[str] = None
    next_intended_zone: Optional[str] = None
    speed: float = 4.0
    travel_budget: int = TRAVEL_BUDGET
    work_budget: int = WORK_BUDGET
    shift_offset: int = 0
    just_arrived_home: bool = False
    in_reinterpretation: bool = False
    reinterpretation_ticks: int = 0
    id: int = field(default_factory=lambda: next(NPC_ID_GEN))
    exchange_actions: int = 0
    long_term_commitments: int = 0
    risk_absorptions: int = 0
    stress_endured: float = 0.0
    basic_needs: float = 10.0
    money: float = 190.0
    health: float = 1.0
    energy: float = 100.0
    skills: Dict[str, float] = field(default_factory=lambda: {"SCIENCE": 0.0, "TRADE": 0.0, "DEVELOPMENT": 0.0, "FLEX": 0.0})
    commitments: Dict[str, Commitment] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)
    trust_log: deque = field(default_factory=lambda: deque(maxlen=300))
    decision_history: deque = field(default_factory=lambda: deque(maxlen=80))
    choice_log: deque = field(default_factory=lambda: deque(maxlen=60))
    zone_visit_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    zones_visited_this_cycle: list = field(default_factory=list)
    last_zone_choice: Optional[str] = None
    satisfaction_history: deque = field(default_factory=lambda: deque(maxlen=10))
    trust_log: deque = field(default_factory=lambda: deque(maxlen=300))
    total_jobs_completed: int = 0
    pending_money: float = 0.0
    pending_mood: float = 0.0
    pending_energy: float = 0.0
    pending_needs: float = 0.0
    cycle_phase: str = "RESTING"
    is_collapsing: bool = False
    vx_target: Optional[float] = None
    vy_target: Optional[float] = None
    personality: Dict[str, float] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)

    # In NPC dataclass (add after line 364, after 'derived: Dict[str, float] = field(default_factory=dict)')
    current_zone_ticks: int = 0
    fatigue_penalty_multiplier: float = 1.0
    zone_fatigue: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_zone: Optional[str] = None
    last_zone_change_tick: int = 0
    fatigue_history: deque = field(default_factory=lambda: deque(maxlen=20))
    fatigue_multiplier: float = 1.0

    @property
    def self_esteem(self):
        # Use derived trait if available, else fallback to 0.0
        return self.derived.get("self_esteem_base", 0.0)
    @property
    def trade_dissonance(self):
        return 0.0  # or some computed value if you want
    @property
    def trust(self):
        if self.trust_log:
            return sum(self.trust_log) / len(self.trust_log)
        return 0.0

    def set_target_anchor(self, anchor_pos):
        self.vx_target, self.vy_target = anchor_pos

    def visual_step(self, speed=2.5):
        if self.vx_target is None or self.vy_target is None:
            return False
        dx = self.vx_target - self.x
        dy = self.vy_target - self.y
        dist = math.hypot(dx, dy)
        if dist <= speed:
            self.x, self.y = self.vx_target, self.vy_target
            self.vx_target = None
            self.vy_target = None
            return True
        self.x += dx / dist * speed
        self.y += dy / dist * speed
        return False

def compute_derived_traits(personality: Dict[str, float]) -> Dict[str, float]:
    r = personality.get("reactivity", 0.5)
    res = personality.get("resilience", 0.5)
    amb = personality.get("ambition", 0.5)
    soc = personality.get("sociability", 0.5)
    stb = personality.get("stability_bias", 0.5)
    return {
        "self_esteem_base": 0.4 - 0.4 * amb + 0.2 * res,
        "goal_stickiness": 0.5 * stb + 0.3 * res + 0.2 * amb,
        "stress_tolerance": 0.6 * res + 0.4 * (1 - r),
        "recovery_rate": 0.7 * res + 0.3 * stb,
        "social_sensitivity": 0.7 * soc + 0.3 * r,
        "risk_tolerance": 0.6 * (1 - stb) + 0.4 * soc,
        "efficiency_invariance": res,
        "anchoring_authority": stb * res,
    }

@dataclass(frozen=True)
class Intent:
    npc_id: int
    desired_zone: str
    phase_label: str
    confidence: float = 0.5

IntentBuffer = Dict[int, Intent]

@dataclass(frozen=True)
class CommitmentPlan:
    npc_id: int
    key: str
    zone: str
    due_tick: int
    strength: float
    forced_reason: str = ""

@dataclass(frozen=True)
class BreakPlan:
    npc_id: int
    key: str
    reason: str

CommitmentBuffer = Dict[int, CommitmentPlan]
BreakBuffer = List[BreakPlan]

@dataclass(frozen=True)
class MovementResult:
    npc_id: int
    new_pos: Tuple[float, float]
    arrived: bool
    energy_cost: float = 0.0
    money_cost: float = 0.0

MoveBuffer = Dict[int, MovementResult]

@dataclass(frozen=True)
class WorkResult:
    npc_id: int
    zone: str
    money_delta: float = 0.0
    stress_delta: float = 0.0
    energy_delta: float = 0.0
    skills_delta: Dict[str, float] = field(default_factory=dict)

WorkBuffer = Dict[int, WorkResult]

@dataclass(frozen=True)
class InfluenceDelta:
    npc_id: int
    doctrine_bias_delta: float = 0.0
    stress_delta: float = 0.0
    pull_flag: Optional[str] = None

InfluenceBuffer = Dict[int, InfluenceDelta]

@dataclass(frozen=True)
class CollapseFlag:
    npc_id: int
    reason: str
    severity: float = 1.0

CollapseBuffer = Dict[int, CollapseFlag]

@dataclass
class TickBuffers:
    snapshot: WorldSnapshot
    intents: IntentBuffer = field(default_factory=dict)
    commitments: CommitmentBuffer = field(default_factory=dict)
    breaks: BreakBuffer = field(default_factory=list)
    moves: MoveBuffer = field(default_factory=dict)
    work: WorkBuffer = field(default_factory=dict)
    influence: InfluenceBuffer = field(default_factory=dict)
    collapse: CollapseBuffer = field(default_factory=dict)

def allowed_work_zones(npc_view: NPCView, diss_threshold: float):
    allowed = {"SCIENCE", "TRADE", "DEVELOPMENT", "FLEX"}
    if npc_view.energy < 15.0:
        return set()
    if npc_view.energy < 30.0:
        allowed -= {"SCIENCE", "DEVELOPMENT"}
    if npc_view.money < 3.0:
        allowed -= {"SCIENCE"}
    if npc_view.stress_endured > 85.0:
        allowed -= {"DEVELOPMENT"}
    return allowed

def personality_affinity(personality, zone_name, derived):
    affinity = 0.0
    if zone_name == "SCIENCE":
        affinity += personality.get("ambition", 0.0) * 0.3
        affinity += personality.get("sociability", 0.0) * 0.2
    elif zone_name == "TRADE":
        affinity += personality.get("ambition", 0.0) * 0.6
        affinity -= personality.get("stability_bias", 0.0) * 0.3
    elif zone_name == "DEVELOPMENT":
        affinity += personality.get("stability_bias", 0.0) * 0.4
        affinity += personality.get("resilience", 0.0) * 0.3
    elif zone_name == "FLEX":
        affinity += derived.get("risk_tolerance", 0.0) * 0.5
        affinity += personality.get("sociability", 0.0) * 0.3
    return affinity

def decide_intent(
    npc_view: NPCView,
    zone_stats,
    zone_density,
    tick,
    diss_threshold
):
    allowed = allowed_work_zones(npc_view, diss_threshold)
    if not allowed:
        return Intent(npc_id=npc_view.id, desired_zone="HOME", phase_label="NO_ENERGY", confidence=0.0)
    if npc_view.state == "AT_WORK":
        # Stay at work, no new decision
        return Intent(npc_id=npc_view.id, desired_zone=npc_view.zone, phase_label="WORKING", confidence=1.0)
    npc = None
    # If you have access to the full NPC object, use it; otherwise, pass personality/derived as part of NPCView
    if hasattr(npc_view, "personality") and hasattr(npc_view, "derived"):
        personality = npc_view.personality
        derived = npc_view.derived
    else:
        # fallback: look up NPC by id if needed
        personality = {}
        derived = {}
    # ... rest of logic for other states ...
    if npc_view.money < 30:
        phase = "SURVIVAL"
    elif npc_view.stress_endured > 40:
        phase = "STABILIZATION"
    elif npc_view.energy > 65:
        phase = "EXPLORATION"
    else:
        phase = "GROWTH"
    scores = {}
    for zone_name in ["SCIENCE", "TRADE", "DEVELOPMENT", "FLEX"]:
        if zone_name not in allowed:
            continue
        score = 0.0
        if zone_name == "SCIENCE":
            score += max(0.0, (100.0 - npc_view.energy) / 100.0)
        elif zone_name == "TRADE":
            score += max(0.0, (50.0 - npc_view.money) / 50.0) * 2.2
            score += (npc_view.stress_endured / 100.0) * 0.4
        elif zone_name == "DEVELOPMENT":
            need_energy = (100.0 - npc_view.energy) / 100.0
            score += need_energy * 0.8
        elif zone_name == "FLEX":
            stress_pressure = npc_view.stress_endured / 100.0
            desperation = 1.0
            score += stress_pressure * (1.0 + desperation)
        skill_bonus = npc_view.skills.get(zone_name, 0.0) * 1.5
        score += skill_bonus
        occupancy_penalty = npc_view.zone_fatigue.get(zone_name, 0) / 40.0
        score -= occupancy_penalty
        personality_bonus = personality_affinity(personality, zone_name, derived)
        score += personality_bonus * 1.0
        if zone_name in zone_stats:
            score *= zone_stats[zone_name].market_demand
            congestion_penalty = zone_stats[zone_name].congestion_level * 0.6
            score *= (1.0 - congestion_penalty)
        if npc_view.zone_visit_count.get(zone_name, 0) < 2:
            score += 0.2
        score += random.uniform(0.0, 0.1)
        scores[zone_name] = score
    if not scores:
        return Intent(npc_id=npc_view.id, desired_zone="HOME", phase_label=phase, confidence=0.0)
    choice = max(scores, key=scores.get)
    confidence = min(1.0, scores[choice] / (sum(scores.values()) + 1e-6))
    return Intent(npc_id=npc_view.id, desired_zone=choice, phase_label=phase, confidence=confidence)

def compute_work_result(npc_view: NPCView, zone, active_events):
    load = (npc_view.stress_endured / 100.0) + (1.0 - (npc_view.energy / 100.0))
    efficiency = max(0.1, min(1.0, 1.0 - (load * 0.3)))
    skill_bonus = npc_view.skills.get(zone, 0.0) * 1.0
    efficiency *= (1.0 + skill_bonus)
    money_mult = 1.0
    stress_mult = 1.0
    energy_mult = 1.0
    fatigue_multiplier = getattr(npc_view, "fatigue_penalty_multiplier", 1.0)
    money_mult *= fatigue_multiplier  # Before using money_mult
    for event in active_events:
        if event.zone_affected is None or event.zone_affected == zone:
            money_mult *= event.money_multiplier
            stress_mult *= event.stress_multiplier
            energy_mult *= event.energy_multiplier
    money_delta = 0.0
    stress_delta = 0.0
    energy_delta = 0.0
    skills_delta = {}
    if zone == "SCIENCE":
        money_delta = 0.5 * efficiency
        stress_delta = npc_view.stress_endured * -0.05
    elif zone == "TRADE":
        money_delta = 8.0 * efficiency * money_mult
        stress_delta = 2.0 * stress_mult
    elif zone == "FLEX":
        stress_delta = -4.0 * efficiency * stress_mult
        energy_delta = -3.0 * energy_mult
    elif zone == "DEVELOPMENT":
        energy_delta = min(100.0 - npc_view.energy, 15.0 * efficiency * energy_mult)
    if zone in npc_view.skills:
        skills_delta[zone] = 0.01
    return WorkResult(
        npc_id=npc_view.id,
        zone=zone,
        money_delta=money_delta,
        stress_delta=stress_delta,
        energy_delta=energy_delta,
        skills_delta=skills_delta
    )

def compute_influence_delta(npc_view: NPCView):
    return InfluenceDelta(npc_id=npc_view.id)

def compute_collapse_flag(npc_view: NPCView, stress_threshold=90.0, withdrawal_gap=15):
    if npc_view.stress_endured > stress_threshold:
        return CollapseFlag(
            npc_id=npc_view.id,
            reason="stress",
            severity=min(1.0, (npc_view.stress_endured - stress_threshold) / 50.0)
        )
    return None

def commit_phase(snapshot, intents):
    commitments = {}
    breaks = []
    for npc_id, intent in intents.items():
        npc = snapshot.npc[npc_id]
        if intent.desired_zone == "HOME":
            plan = CommitmentPlan(
                npc_id=npc_id,
                key="REST:HOME",
                zone="HOME",
                due_tick=snapshot.tick + 30,
                strength=0.8
            )

        if intent.desired_zone not in ["SCIENCE", "TRADE", "DEVELOPMENT", "FLEX"]:
            continue
        key = f"WORK:{intent.desired_zone}"
        zone_eff = snapshot.zone_stats.get(intent.desired_zone)
        zone_efficiency = zone_eff.efficiency_rating if zone_eff else 1.0
        plan = CommitmentPlan(
            npc_id=npc_id,
            key=key,
            zone=intent.desired_zone,
            due_tick=snapshot.tick + npc.work_budget,
            strength=0.5 + 0.5 * zone_efficiency,
            forced_reason=""
        )
        commitments[npc_id] = plan
    return commitments, breaks

def move_phase(snapshot, commitments):
    moves = {}
    for npc_id, plan in commitments.items():
        if plan.zone not in ANCHORS:
            continue
        moves[npc_id] = MovementResult(
            npc_id=npc_id,
            new_pos=ANCHORS[plan.zone],
            arrived=False
        )
    return moves

def apply_commit_reality(world, buffers):
    for npc_id, move in buffers.moves.items():
        npc = next((n for n in world.npcs if n.id == npc_id), None)
        if npc:
            plan = buffers.commitments.get(npc_id)
            if not plan:
                continue
            npc.target = plan.zone
            npc.state = "TRAVELING"
            npc.set_target_anchor(ANCHORS[plan.zone])
            if move.arrived:
                npc.state = "AT_WORK"
                npc.zone = buffers.commitments[npc_id].zone
    for npc_id, work in buffers.work.items():
        npc = next((n for n in world.npcs if n.id == npc_id), None)
        if npc:
            npc.money += work.money_delta
            npc.stress_endured = max(0.0, min(100.0, npc.stress_endured + work.stress_delta))
            npc.energy = max(0.0, min(100.0, npc.energy + work.energy_delta))
            for skill, delta in work.skills_delta.items():
                npc.skills[skill] = min(1.0, npc.skills.get(skill, 0.0) + delta)
    for npc_id, infl in buffers.influence.items():
        npc = next((n for n in world.npcs if n.id == npc_id), None)
        if npc:
            npc.stress_endured = max(0.0, min(100.0, npc.stress_endured + infl.stress_delta))
    for npc_id, flag in buffers.collapse.items():
        npc = next((n for n in world.npcs if n.id == npc_id), None)
        if npc:
            npc.is_collapsing = True
            npc.cycle_phase = "COLLAPSED"
    for npc in world.npcs:
        if npc.is_collapsing and npc.stress_endured < 70.0:
            npc.is_collapsing = False
            npc.cycle_phase = "RESTING"
    for npc_id, plan in buffers.commitments.items():
        npc = next((n for n in world.npcs if n.id == npc_id), None)
        if npc:
            npc.commitments[plan.key] = Commitment(
                key=plan.key,
                created_tick=buffers.snapshot.tick,
                due_tick=plan.due_tick,
                strength=plan.strength
            )

def draw_npcs(surface, world, anchors, fonts):
    for npc in world.npcs:
        pos = (int(npc.x), int(npc.y))
        # Draw personality color as outline
        outline_color = PERSONALITY_COLORS.get(getattr(npc, "personality_name", ""), WHITE)
        pygame.draw.circle(surface, outline_color, pos, 7)
        # Draw state/zone color as fill
        if npc.state == "AT_HOME":
            color = ORANGE
        elif npc.state == "TRAVELING":
            color = RED_TEXT
        elif npc.state == "AT_WORK":
            color = ZONE_MECHANICS.get(npc.zone, {}).get("color", WHITE)
        else:
            color = (180, 180, 180)
        pygame.draw.circle(surface, color, pos, 5)

def update_zone_occupancy(world, tick):
    for npc in world.npcs:
        if npc.state == "AT_WORK":
            # Track how long in current zone
            if npc.last_zone != npc.zone:
                npc.current_zone_ticks = 1
                npc.last_zone = npc.zone
                npc.last_zone_change_tick = tick
            else:
                npc.current_zone_ticks += 1
            npc.zone_fatigue[npc.zone] = npc.current_zone_ticks
            # Fatigue penalty after 120 ticks
            if npc.zone != npc.last_zone:
                npc.zones_visited_this_cycle.append(npc.zone)
                if len(npc.zones_visited_this_cycle) > 10:
                    npc.zones_visited_this_cycle.pop(0)
                npc.last_zone = npc.zone
            if npc.current_zone_ticks > 120:
                penalty = min(0.3, (npc.current_zone_ticks - 120) / 200)
                npc.fatigue_penalty_multiplier = 1.0 - penalty
            else:
                npc.fatigue_penalty_multiplier = 1.0
            npc.fatigue_history.append(npc.fatigue_penalty_multiplier)
        else:
            npc.current_zone_ticks = 0
            npc.fatigue_penalty_multiplier = 1.0


def simtick(world, tick):
    update_zone_occupancy(world, tick)
    snapshot = build_snapshot(world, tick)
    buffers = TickBuffers(snapshot=snapshot)
    # P1: Decide
    for npc_id, npc_view in snapshot.npc.items():
           
        intent = decide_intent(
            npc_view,
            snapshot.zone_stats,
            snapshot.zone_density,
            tick,
            DISS_THRESHOLD
        )
        buffers.intents[npc_id] = intent

    # P2: Commit
    buffers.commitments, buffers.breaks = commit_phase(snapshot, buffers.intents)
    # P3: Move
    buffers.moves = move_phase(snapshot, buffers.commitments)
    # P4: Work
    for npc_id, npc_view in snapshot.npc.items():
        if npc_view.state == "AT_WORK":
            buffers.work[npc_id] = compute_work_result(
                npc_view, npc_view.zone, snapshot.active_events
            )
    # P5: Influence
    for npc_id, npc_view in snapshot.npc.items():
        buffers.influence[npc_id] = compute_influence_delta(npc_view)
    # P6: Collapse
    for npc_id, npc_view in snapshot.npc.items():
        flag = compute_collapse_flag(npc_view, 90.0, WITHDRAWAL_GAP)
        if flag:
            buffers.collapse[npc_id] = flag
    # P7: COMMIT_REALITY
    apply_commit_reality(world, buffers)
    return buffers

def draw_anchors(surface, anchors, font_small, zone_stats):
    for name, (x, y) in anchors.items():
        if name in zone_stats:
            pop = zone_stats[name].current_population
            color = (100, 255, 100) if pop > 0 else TEAL
        else:
            color = TEAL
        pygame.draw.circle(surface, color, (int(x), int(y)), 8, 2)
        label = font_small.render(name, True, WHITE)
        surface.blit(label, (int(x) + 15, int(y) - 10))

# ... (rest of rendering and main loop code remains unchanged) ...

# PHYSICAL PHASE (The Body) - update work_budget and state
    for w in worlds:
        for npc in w.npcs:
            if npc.state == "TRAVELING":
                if npc.visual_step(speed=3.5):
                    npc.state = "AT_WORK"
                    npc.zone = npc.target
            elif npc.state == "AT_WORK":
                npc.energy -= 0.05
                npc.work_budget -= 1
                if npc.work_budget <= 0:
                    npc.state = "AT_HOME"
                    npc.work_budget = WORK_BUDGET

# ... (rest of main loop and rendering code) ...

def draw_statistics_dashboard(surface, world, zone_stats, active_events, fonts):
    """Comprehensive analytics overlay"""
    panel_w, panel_h = 1530, 870
    panel_x = WIDTH // 2 - panel_w // 2
    panel_y = HEIGHT // 2 - panel_h // 2

    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((20, 20, 40, 230))

    title = fonts['header'].render("📊 WORLD ANALYTICS", True, LIME)
    panel_surf.blit(title, (panel_w//2 - title.get_width()//2, 750))

    y_offset = 70

    section_title = fonts['large'].render("🌍 World Overview", True, YELLOW_TEXT)
    panel_surf.blit(section_title, (20, y_offset))
    y_offset += 40

    y_offset += 20

    total_money = sum(npc.money for npc in world.npcs)
    avg_stress = sum(npc.stress_endured for npc in world.npcs) / len(world.npcs) if world.npcs else 0
    collapsed = sum(1 for npc in world.npcs if npc.is_collapsing)

    overview_stats = [
        f"Total NPCs: {len(world.npcs)}",
        f"Total Wealth: ${total_money:.1f}",
        f"Avg Stress: {avg_stress:.1f}",
        f"Collapsed NPCs: {collapsed}",
    ]

    for stat in overview_stats:
        text = fonts['small'].render(stat, True, WHITE)
        panel_surf.blit(text, (40, y_offset))
        y_offset += 25

    y_offset += 20

    section_title = fonts['large'].render("🏭 Zone Performance", True, YELLOW_TEXT)
    panel_surf.blit(section_title, (20, y_offset))
    y_offset += 40

    for zone_name, stats in zone_stats.items():
        zone_line = f"{zone_name}: Pop={stats.current_population} Eff={stats.efficiency_rating:.2f} Work={stats.total_work_done}"
        color_map = {
            "SCIENCE": (100, 200, 255),
            "TRADE": TRADE_COLOR,
            "DEVELOPMENT": DEV_COLOR,
            "FLEX": FLEX_COLOR,
           
        }
        color = color_map.get(zone_name, WHITE)
        text = fonts['small'].render(zone_line, True, color)
        panel_surf.blit(text, (40, y_offset))
        y_offset += 25

    surface.blit(panel_surf, (panel_x, panel_y))


    for npc in world.npcs:
        # Always draw at current position
        pos = (int(npc.x), int(npc.y))

        # Color is semantic
        if npc.state == "AT_HOME":
            color = ORANGE
        elif npc.state == "TRAVELING":
            color = RED_TEXT
        elif npc.state == "AT_WORK":
            color = color_map.get(npc.zone, WHITE)
        else:
            color = (180, 180, 180)  # fallback

        pygame.draw.circle(screen, color, pos, 5)

INSPECTOR_ZONE_ORDER = ["SCIENCE", "TRADE", "DEVELOPMENT", "FLEX",]
def draw_npc_inspector(surface, world, inspector_zone_idx, fonts, zone_stats):
    """Enhanced NPC inspector with zone rotation AND personality traits."""

    panel_w, panel_h = 1520, 860
    panel_x = WIDTH // 2 - panel_w // 2
    panel_y = HEIGHT // 2 - panel_h // 2

    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((20, 20, 40, 230))
    row_idx = -1
    
    # Title
    title = fonts['header'].render("🔍 NPC INSPECTOR [PERSONALITY & ZONE ROTATION]", True, LIME)
    panel_surf.blit(title, (700, 700))

    # Personality legend
    y_offset = 55
    personality_title = fonts['small'].render("Personality Archetypes:", True, YELLOW_TEXT)
    panel_surf.blit(personality_title, (20, y_offset))
    y_offset += 25

    arch_colors = {
        "Anchor": (80, 180, 255),
        "Climber": (255, 180, 80),
        "Connector": (180, 80, 255),
        "Survivor": (80, 255, 120),
    }
    arch_x = 20
    for arch, color in arch_colors.items():
        pygame.draw.circle(panel_surf, color, (arch_x + 8, y_offset + 8), 5)
        pygame.draw.circle(panel_surf, WHITE, (arch_x + 8, y_offset + 8), 5, 1)
        text = fonts['tiny'].render(f"= {arch}", True, color)
        panel_surf.blit(text, (arch_x + 20, y_offset))
        arch_x += 160

    y_offset += 25
    pygame.draw.line(panel_surf, YELLOW_TEXT, (20, y_offset), (panel_w - 20, y_offset), 1)
    y_offset += 25

    # Headers
    headers = ["ID", "Personality", "Zone", "Phase", "Stress", "$", "E", "Ticks"]
    header_spacing = [10, 20, 20, 20, 15, 15, 15, 15]
    header_text = ""
    for h, sp in zip(headers, header_spacing):
        header_text += f"{h:<{sp}}"
    header_render = fonts['small'].render(header_text, True, YELLOW_TEXT)
    panel_surf.blit(header_render, (40, y_offset))
    y_offset += 25
    pygame.draw.line(panel_surf, YELLOW_TEXT, (20, y_offset), (panel_w - 20, y_offset), 1)
    y_offset += 25

    # Get NPCs
    npcs = world.npcs
    if inspector_zone_idx is not None and 0 <= inspector_zone_idx < len(INSPECTOR_ZONE_ORDER):
        zone_name = INSPECTOR_ZONE_ORDER[inspector_zone_idx]
        npcs = [n for n in npcs if n.zone == zone_name]
    npcs = sorted(npcs, key=lambda n: n.stress_endured, reverse=True)
    
    # Pagination logic
    PAGE_SIZE = 8  # Reduced to accommodate multi-line rows
    total_npcs = len(npcs)
    total_pages = max(1, math.ceil(total_npcs / PAGE_SIZE))
    
    # Clamp current page
    if ui_state.inspector_page >= total_pages:
        ui_state.inspector_page = total_pages - 1
    if ui_state.inspector_page < 0:
        ui_state.inspector_page = 0
        
    start_idx = ui_state.inspector_page * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    displayed_npcs = npcs[start_idx:end_idx]

    zone_colors = {
        "SCIENCE": (100, 200, 255),
        "TRADE": TRADE_COLOR,
        "DEVELOPMENT": DEV_COLOR,
        "FLEX": FLEX_COLOR,
        "HOME": ORANGE,
    }
    
    
    # Draw NPC rows
    for row_idx, npc in enumerate(displayed_npcs):
        npc_y_pos = y_offset + (row_idx * 75)  # Increased from 24 to avoid overlap
        phase = npc.decision_history[-1].phase if npc.decision_history else "---"
        ticks_in_zone = getattr(npc, 'current_zone_ticks', 0)
        personality_name = getattr(npc, 'personality_name', '?')
                                                        
        row_text = (
            f"{npc.id:<10} "
            f"{personality_name:<20} "
            f"{npc.zone[:10]:<20} "
            f"{phase[:12]:<20} "
            f"{npc.stress_endured:<15.0f} "
            f"{npc.money:<15.0f} "
            f"{npc.energy:<15.0f} "
            f"{ticks_in_zone:<15}"
        )
            # ... your row rendering code ...
        if row_idx >= 0:
                legend_y = y_offset + (row_idx * 20) + 8
           
        # Color by stress
        if npc.stress_endured > 85:
            row_color = (255, 0, 0)
        elif npc.stress_endured > 70:
            row_color = (255, 150, 0)
        elif npc.stress_endured > 50:
            row_color = (200, 200, 0)
        elif npc.is_collapsing:
            row_color = (255, 0, 150)
        else:
            row_color = WHITE

        row_render = fonts['small'].render(row_text, True, row_color)
        panel_surf.blit(row_render, (40, npc_y_pos))

        # Draw zone rotation balls
        zones_visited = getattr(npc, 'zones_visited_this_cycle', [])
        recent_zones = zones_visited[-4:] if zones_visited else []
        ball_x = 700
        ball_y = npc_y_pos + 8
        ball_size = 5
        ball_spacing = 14
        for ball_idx, zone in enumerate(recent_zones):
            color = zone_colors.get(zone, WHITE)
            x_pos = ball_x + (ball_idx * ball_spacing)
            pygame.draw.circle(panel_surf, color, (x_pos, ball_y), ball_size)
            pygame.draw.circle(panel_surf, WHITE, (x_pos, ball_y), ball_size, 1)

        # Draw monotony bar
        if ticks_in_zone > 0:
            monotony_x = ball_x + 90
            monotony_bar_w = 50
            monotony_bar_h = 3
            max_ticks = 80
            pygame.draw.rect(panel_surf, (50, 50, 50), (monotony_x, ball_y, monotony_bar_w, monotony_bar_h))
            fill_width = int(monotony_bar_w * min(1.0, ticks_in_zone / max_ticks))
            if ticks_in_zone < 40:
                bar_color = (0, 255, 0)
            elif ticks_in_zone < 60:
                bar_color = (255, 200, 0)
            else:
                bar_color = (255, 0, 0)
            pygame.draw.rect(panel_surf, bar_color, (monotony_x, ball_y, fill_width, monotony_bar_h))
            ticks_text = fonts['tiny'].render(f"{ticks_in_zone}⏱", True, bar_color)
            panel_surf.blit(ticks_text, (monotony_x + monotony_bar_w + 8, npc_y_pos))

        # Display traits below main row
        personality = getattr(npc, 'personality', {})
        derived = getattr(npc, 'derived', {})
        traits_y = npc_y_pos + 20
        personality_text = (
            f"→ Reactivity:{personality.get('reactivity', 0):.2f} "
            f"Resilience:{personality.get('resilience', 0):.2f} "
            f"Ambition:{personality.get('ambition', 0):.2f} "
            f"Sociability:{personality.get('sociability', 0):.2f} "
            f"Stability:{personality.get('stability_bias', 0):.2f}"
        )
        traits_render = fonts['small'].render(personality_text, True, (150, 200, 150))
        panel_surf.blit(traits_render, (40, traits_y))
        derived_text = (
            f"  └─ Stress_Tol:{derived.get('stress_tolerance', 0):.2f} "
            f"Recovery:{derived.get('recovery_rate', 0):.2f} "
            f"Risk_Tol:{derived.get('risk_tolerance', 0):.2f} "
            f"Goal_Sticky:{derived.get('goal_stickiness', 0):.2f} "
            f"Efficiency:{derived.get('efficiency_invariance', 0):.2f}"
        )
        derived_render = fonts['small'].render(derived_text, True, (150, 150, 200))
        panel_surf.blit(derived_render, (40, traits_y + 40))

    # --- LEGEND & TRAIT PANEL ANCHORED TO BOTTOM ---
    legend_margin = 30
    legend_y = panel_h - 180  # Anchor legend 180px from bottom (adjust as needed)
    pygame.draw.line(panel_surf, YELLOW_TEXT, (20, legend_y - 10), (panel_w - 10, legend_y - 10), 1)
    legend_title = fonts['small'].render("📖 LEGEND & TRAIT DESCRIPTIONS", True, YELLOW_TEXT)
    panel_surf.blit(legend_title, (20, legend_y))
    legend_y += 25
    legend_text = fonts['small'].render(
        "Zones: 🔵=SCIENCE  🟡=TRADE  🟣=DEVELOPMENT  🔴=FLEX",
        True, WHITE
    )
    panel_surf.blit(legend_text, (20, legend_y))
    legend_y += 28
    legend_text = fonts['small'].render(
        "Personality: Reactivity(responsive), Resilience(tough), Ambition(driven), Sociability(social), Stability(predictable)",
        True, (150, 200, 150)
    )
    panel_surf.blit(legend_text, ( 20, legend_y))
    legend_y += 28
    legend_text = fonts['small'].render(
        "Derived: Stress_Tol(handle pressure), Recovery(heal fast), Risk_Tol(take chances), Goal_Sticky(stay focused), Efficiency(consistent)",
        True, (150, 150, 200)
    )
    panel_surf.blit(legend_text, (20, legend_y))
    legend_y += 28
    stress_legend = [
        ("Stress: ", WHITE),
        ("🟢<50=Healthy ", (0, 255, 0)),
        ("🟡50-70=Moderate ", (200, 200, 0)),
        ("🟠70-85=High ", (255, 150, 0)),
        ("🔴>85=Critical", (255, 0, 0)),
    ]
    stress_x = 28
    for label, color in stress_legend:
        text = fonts['small'].render(label, True, color)
        panel_surf.blit(text, (stress_x, legend_y))
        stress_x += text.get_width() + 10
    legend_y += 28
    legend_text = fonts['small'].render(
        f"Zone Fatigue: 🟢<40 ticks=Fresh(stress↓) | 🟡40-60=Warning | 🟠60-80=Change soon | 🔴>80=FORCED change",
        True, WHITE
    )
    panel_surf.blit(legend_text, (20, legend_y))

    # Pagination Indicator (keep at bottom right)
    page_text = fonts['small'].render(f"NPCs: {total_npcs} | PAGE {ui_state.inspector_page + 1} / {total_pages}", True, LIME)
    hint_text = fonts['tiny'].render("(Use PAGE UP / PAGE DOWN keys to browse all NPCs)", True, WHITE)
    panel_surf.blit(page_text, (panel_w - page_text.get_width() - 40, panel_h - 60))
    panel_surf.blit(hint_text, (panel_w - hint_text.get_width() - 40, panel_h - 35))

    surface.blit(panel_surf, (panel_x, panel_y))
def draw_color_legend_corner(surface, x, y, corner_size, font_small, font_tiny):
    """Draw NPC state color legend"""
    pygame.draw.rect(surface, (30, 30, 30), (x, y, corner_size, corner_size))
    
    title = font_small.render("NPC Colors", True, YELLOW_TEXT)
    surface.blit(title, (x + 16, y + 16))
    
    y_offset = y + 50
    line_height = 22
    
    legend_entries = [
        ("HOME (Rest)", ORANGE),
        ("HOME (Extract)", LIME),
        ("→ Work", RED_TEXT),
        ("← Home", (255, 100, 255)),
        ("Hub", BLUE),
        ("Science", (100, 200, 255)),
        ("Trade", TRADE_COLOR),
        ("Dev", DEV_COLOR),
        ("Flex", FLEX_COLOR),
    ]
    
    for i, (label, color) in enumerate(legend_entries):
        current_y = y_offset + i * line_height
        circle_x = x + 20
        circle_y = current_y + 8
        pygame.draw.circle(surface, color, (circle_x, circle_y), 6)
        pygame.draw.circle(surface, WHITE, (circle_x, circle_y), 6, 1)
        text = font_tiny.render(label, True, WHITE)
        surface.blit(text, (x + 35, current_y))

def save_npc_state(world, filename=None):
    """Save all NPC state to JSON file for analysis."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"npc_state_{timestamp}.json"
    npc_data = []
    for npc in world.npcs:
        data = {
            "id": npc.id,
            "personality_name": getattr(npc, 'personality_name', 'Unknown'),
            "personality": getattr(npc, 'personality', {}),
            "derived": getattr(npc, 'derived', {}),
            "state": npc.state,
            "zone": npc.zone,
            "x": npc.x,
            "y": npc.y,
     }       

def draw_combined_view(surface, all_worlds, fonts, global_tick):
    """Enhanced combined view showing all territories"""
    surface.fill(BLACK)
    
    title = fonts['header'].render("🌐 COMBINED ZONE VIEW - ALL TERRITORIES", True, LIME)
    surface.blit(title, (WIDTH // 2 - title.get_width() // 2, 800))
    
    TERRITORY_COLORS = {
        "TERRITORY_1": (100, 200, 255),
        "TERRITORY_2": (255, 180, 100),
        "TERRITORY_3": (150, 100, 200),
    }
    
    ANCHORS_PX = anchors_to_pixels(ZONE_BASES_NORM, WIDTH, HEIGHT)
    
    for i, (base_name, (bx, by)) in enumerate(ANCHORS_PX.items()):
        if i >= len(all_worlds):
            break
        world = all_worlds[i]
        zone_color = TERRITORY_COLORS.get(base_name, WHITE)
        
        pygame.draw.circle(surface, zone_color, (int(bx), int(by)), 50, 6)
        pygame.draw.circle(surface, zone_color, (int(bx), int(by)), 30, 3)
        
        name_text = fonts['small'].render(world.name, True, zone_color)
        surface.blit(name_text, (int(bx) - name_text.get_width() // 2, int(by) - 70))
        
        active = sum(1 for n in world.npcs if n.state != "AT_HOME")
        pop_text = fonts['header'].render(f"{active}/{len(world.npcs)}", True, WHITE)
        surface.blit(pop_text, (int(bx) - pop_text.get_width() // 2, int(by) - pop_text.get_height() // 2))


def draw_corner_panels(screen, world, fonts, tick):
    """Draw info panels in all four corners"""
    corner_size = 220
    
    # Top-left: World info
    pygame.draw.rect(screen, (30, 30, 30, 200), (0, 0, corner_size, corner_size))
    tl_lines = [
        f"Tick: {tick}",
        f"World: {world.name}",
        f"NPCs: {len(world.npcs)}",
        f"Active: {sum(1 for n in world.npcs if n.state != 'AT_HOME')}",
    ]
    y = 10
    for line in tl_lines:
        text = fonts['tiny'].render(line, True, LIME)
        screen.blit(text, (10, y))
        y += 25
    
    # Top-right: Stats
    total_wealth = sum(npc.money for npc in world.npcs)
    collapsed = sum(1 for npc in world.npcs if npc.is_collapsing)
   
    pygame.draw.rect(screen, (30, 30, 30, 200), (WIDTH - corner_size, 0, corner_size, corner_size))
    tr_lines = [
        f"World: {world.name}",
        f"NPCs: {len(world.npcs)}",
        f"Wealth: ${total_wealth:.0f}",
        f"Collapsed: {collapsed}",
    ]
    y = 10
    for line in tr_lines:
        text = fonts['tiny'].render(line, True, GREEN_TEXT)
        screen.blit(text, (WIDTH - corner_size + 10, y))
        y += 25
    
    # Bottom-left: Color legend
    draw_color_legend_corner(screen, 0, HEIGHT - corner_size, corner_size, fonts['small'], fonts['tiny'])
    
    # Bottom-right: Personality legend (replaces stats)
    pygame.draw.rect(screen, (30, 30, 30, 200), (WIDTH - corner_size, HEIGHT - corner_size, corner_size, corner_size))
    legend_title = fonts['small'].render("Personality Types", True, YELLOW_TEXT)
    screen.blit(legend_title, (WIDTH - corner_size + 16, HEIGHT - corner_size + 16))
    y_offset = HEIGHT - corner_size + 50
    line_height = 28
    for name, color in PERSONALITY_COLORS.items():
        pygame.draw.circle(screen, color, (WIDTH - corner_size + 28, y_offset + 8), 9)
        text = fonts['tiny'].render(name, True, WHITE)
        screen.blit(text, (WIDTH - corner_size + 45, y_offset))
        y_offset += line_height





def render_view(screen, world, worlds, ui_state, fonts, tick):
    """Main rendering orchestrator - dispatches to specific draw functions"""
    draw_anchors(screen, ANCHORS, fonts['small'], world.zone_stats)
    
    # Draw NPCs on main map
    draw_npcs(screen, world, ANCHORS, fonts)
    
    # Draw corner panels (always visible)
    draw_corner_panels(screen, world, fonts, tick)

    if ui_state.show_statistics:
        draw_statistics_dashboard(
            screen, world, world.zone_stats, world.active_events, fonts
        )

    if ui_state.show_npc_inspector:
        draw_npc_inspector(
            screen, world, ui_state.inspector_zone_idx, fonts, world.zone_stats
        )

    if ui_state.show_monotony_panel:
        # draw_monotony_panel(screen, active_world, fonts_dict)
        pass
       

    if ui_state.show_combined_view:
        draw_combined_view(screen, worlds, fonts, tick)
        

    # Render visual layers for NPCs
    for npc in world.npcs:
        pos = (int(npc.x), int(npc.y))
        if npc.state == "TRAVELING":
            pygame.draw.circle(screen, TEAL, pos, 4, 1)
        elif npc.state == "AT_WORK":
            pygame.draw.circle(screen, GREEN, pos, 6, 2)

    


# ---------- INITIALIZE WORLDS ----------
print("Initializing trust system...\n")

ANCHORS = anchors_to_pixels(ANCHORS_NORM, WIDTH, HEIGHT)
ZONE_BASES = anchors_to_pixels(ZONE_BASES_NORM, WIDTH, HEIGHT)

# ===== ZONE MECHANICS (from top-level config) =====
ZONE_MECHANICS = {}
for zone_name, zone_cfg in config["zones"].items():
    ZONE_MECHANICS[zone_name] = {
        "name": zone_name,
        "color": COLORS.get(zone_cfg.get("color", "WHITE"), WHITE),
        "description": zone_cfg.get("description", ""),
    }

print(f"✅ Loaded {len(ZONE_MECHANICS)} zone mechanics: {list(ZONE_MECHANICS.keys())}\n")

home_x, home_y = ANCHORS["HOME"]

# ===== INITIALIZE WORLDS FROM TERRITORIES =====
territories = config["territories"]
worlds = []

# ... world initialization loop ...
for terr_key, terr_cfg in territories.items():
    world = World(name=terr_cfg.get("name", terr_key))
    terr_zones = terr_cfg.get("zones", {})
    for zone_name, zone_spawn_cfg in terr_zones.items():
        if zone_name not in ZONE_MECHANICS:
            continue
        npc_count = zone_spawn_cfg.get("npc_count", 0)
        initial_resources = zone_spawn_cfg.get("initial_resources", {})

        # Pick a personality archetype
        personality_pool = config.get("npc_personalities", {})
        personality_name = random.choice(list(personality_pool.keys()))
        personality = personality_pool[personality_name]

        for _ in range(npc_count):
            shift_offset = random.randint(0, SHIFT_DURATION)
            npc = NPC(
                x=home_x,
                y=home_y,
                zone=zone_name,
                state="AT_HOME",  # Explicitly set
                shift_offset=shift_offset,
                money=initial_resources.get("MATERIAL", 190.0),
                energy=initial_resources.get("ENERGY", 100.0),
                stress_endured=100.0 - initial_resources.get("STABILITY", 10.0),
            )
            npc.personality = personality
            npc.derived = compute_derived_traits(personality)
            npc.personality_name = personality_name
            world.npcs.append(npc)
    worlds.append(world)
    total_npcs = len(world.npcs)
    
    # Print territory summary
    print(f"  ✅ Territory: '{terr_cfg.get('name', terr_key)}'")
    print(f"     Total: {total_npcs} NPCs")
    
    # Break down by zone
    zone_counts = {}
    for npc in world.npcs:
        zone_counts[npc.zone] = zone_counts.get(npc.zone, 0) + 1
    
    for zone_name in sorted(zone_counts.keys()):
        count = zone_counts[zone_name]
        print(f"       • {zone_name}: {count:2d} NPCs")
    print()

print(f"🚀 Total: {len(worlds)} territories, {sum(len(w.npcs) for w in worlds)} NPCs")
print(f"✅ All systems initialized. Press 'S' for stats, 'I' for inspector, 'C' for combined view\n")


# ---------- UI STATE ----------
ui_state = UIState()

def draw_personality_legend(surface, x, y, font_small, font_tiny):
    pygame.draw.rect(surface, (30, 30, 30), (x, y, 180, 120))
    title = font_small.render("Personality", True, YELLOW_TEXT)
    surface.blit(title, (x + 10, y + 10))
    y_offset = y + 40
    line_height = 22
    
    legend_entries = [
        ("Anchor", PERSONALITY_COLORS["Anchor"]),
        ("Climber", PERSONALITY_COLORS["Climber"]),
        ("Connector", PERSONALITY_COLORS["Connector"]),
        ("Survivor", PERSONALITY_COLORS["Survivor"]),
    ]
    
    for i, (label, color) in enumerate(legend_entries):
        current_y = y_offset + i * line_height
        pygame.draw.circle(surface, color, (x + 20, current_y + 8), 6)
        pygame.draw.circle(surface, WHITE, (x + 20, current_y + 8), 6, 1)
        text = font_tiny.render(label, True, WHITE)
        surface.blit(text, (x + 35, current_y))

    # Fade effect
    fade_rect = pygame.Rect(x, y, 180, 120)
    fade_surface = pygame.Surface(fade_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(fade_surface, (0, 0, 0, 180), fade_rect)
    surface.blit(fade_surface, fade_rect.topleft)

    # Border
    pygame.draw.rect(surface, WHITE, (x, y, 180, 120), 2)


# ---------- MAIN GAME LOOP ----------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_TAB:
                ui_state.current_world_idx = (ui_state.current_world_idx + 1) % len(worlds)
                ui_state.inspector_page = 0  # Reset page when switching world
                print(f"🌍 Switched to: {worlds[ui_state.current_world_idx].name}")

            elif event.key == pygame.K_m:
                ui_state.show_monotony_panel = not ui_state.show_monotony_panel
                print(f"🟠 Monotony Panel: {'ON' if ui_state.show_monotony_panel else 'OFF'}")

            elif event.key == pygame.K_b:
                ui_state.show_npc_inspector = not ui_state.show_npc_inspector           
                print(f"🔍 Inspector: {'ON' if ui_state.show_npc_inspector else 'OFF'}")

            elif event.key == pygame.K_s:
                ui_state.show_statistics = not ui_state.show_statistics            
                print(f"📊 Statistics: {'ON' if ui_state.show_statistics else 'OFF'}")

            elif event.key == pygame.K_c:
                ui_state.show_combined_view = not ui_state.show_combined_view         
                print(f"🌐 Combined View: {'ON' if ui_state.show_combined_view else 'OFF'}")

            elif event.key == pygame.K_PAGEUP:
                ui_state.show_npc_inspector -= 1
            elif event.key == pygame.K_PAGEDOWN:
                ui_state.show_npc_inspector += 1

            elif event.key == pygame.K_0:
                ui_state.inspector_zone_idx = None
                ui_state.inspector_page = 0
            elif event.key == pygame.K_1:
                ui_state.inspector_zone_idx = 0
                ui_state.inspector_page = 0
            elif event.key == pygame.K_2:
                ui_state.inspector_zone_idx = 1
                ui_state.inspector_page = 0
            elif event.key == pygame.K_3:
                ui_state.inspector_zone_idx = 2
                ui_state.inspector_page = 0
            elif event.key == pygame.K_4:
                ui_state.inspector_zone_idx = 3
                ui_state.inspector_page = 0
            elif event.key == pygame.K_ESCAPE:
                running = False
      

    # 2. LOGIC PHASE (The Multiverse)
    for _ in range(2):
        global_tick += 1
        for w in worlds: # Using 'w' so we don't shadow the 'world' variable
            w.step(global_tick)

    # 3. PHYSICAL PHASE (The Body)
    for w in worlds:
        for npc in w.npcs:
            if npc.state == "TRAVELING":
                if npc.visual_step(speed=3.5):
                    npc.state = "AT_WORK"
                    npc.zone = npc.target
            elif npc.state == "AT_WORK":
                npc.energy -= 0.05

    fonts_dict = {
        'header': font_header,
        'large': font_large,
        'small': font_small,
        'tiny': font_tiny
    }

    # 4. RENDER PHASE (The Eyes) - MULTILAYERED
    active_world = worlds[ui_state.current_world_idx]
    screen.fill(PURPLE)
    
    # Layer 1: Base World View
    render_view(
        screen=screen,
        world=active_world,
        worlds=worlds,
        ui_state=ui_state, 
        fonts=fonts_dict,
        tick=global_tick
    )

    # Layer 2: UI Overlays (Drawn on top)
    if ui_state.show_npc_inspector:
        draw_npc_inspector(screen, active_world, ui_state.inspector_zone_idx, fonts_dict, active_world.zone_stats)

    elif ui_state.show_statistics:
        draw_statistics_dashboard(screen, active_world, active_world.zone_stats, active_world.active_events, fonts_dict)

    elif ui_state.show_combined_view:
        draw_combined_view(screen, worlds, fonts_dict, global_tick)

    # Layer 3: Title (Always on top)
    world_name_text = fonts_dict['header'].render(
        f"Active World: {active_world.name} ({ui_state.current_world_idx + 1}/{len(worlds)})",
        True, LIME
    )
    screen.blit(
        world_name_text,
        (WIDTH // 2 - world_name_text.get_width() // 2, 10)
    )

    pygame.display.flip()
    clock.tick(FPS)