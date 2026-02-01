
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.S)
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

HABIT_MARKERS = [
    "habit",
    "routine",
    "daily",
    "every day",
    "every morning",
    "every night",
    "weekly",
    "monthly",
    "often",
    "always",
    "usually",
    "regularly",
    "tend to",
    "keep",
]

REPETITION_MARKERS = [
    "repeat",
    "repetition",
    "repeated",
    "repeatedly",
    "recurring",
    "recurrence",
    "again",
    "over and over",
    "cycle",
    "cyclical",
    "loop",
    "every day",
    "each day",
    "daily",
    "weekly",
    "monthly",
    "yearly",
]

SEQUENCE_MARKERS = [
    "first",
    "second",
    "third",
    "fourth",
    "then",
    "next",
    "after",
    "before",
    "finally",
    "last",
    "later",
    "and then",
    "prior",
]

SEQUENCE_CONNECTORS = {
    "then",
    "next",
    "after",
    "before",
    "finally",
    "last",
    "later",
}

DEFAULT_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "sleep",
        "markers": ["sleep", "bed", "wake", "rest", "nap"],
    },
    {
        "name": "fitness",
        "markers": ["exercise", "workout", "run", "walk", "gym", "train", "stretch", "yoga"],
    },
    {
        "name": "nutrition",
        "markers": ["eat", "meal", "food", "cook", "diet", "snack", "drink", "water"],
    },
    {
        "name": "hygiene",
        "markers": ["shower", "bath", "brush", "teeth", "wash", "groom", "clean"],
    },
    {
        "name": "work",
        "markers": ["work", "job", "project", "task", "meeting", "deadline", "email"],
    },
    {
        "name": "learning",
        "markers": ["learn", "study", "read", "practice", "review", "train", "lesson"],
    },
    {
        "name": "read_understood",
        "markers": [
            "read",
            "reading",
            "understand",
            "understood",
            "comprehend",
            "comprehension",
            "learned",
            "studied",
            "grasp",
            "grasped",
        ],
    },
    {
        "name": "episodic_learning",
        "markers": [
            "episode",
            "episodes",
            "episodic",
            "series of episodes",
            "episode series",
            "tv series",
            "tv show",
            "show episodes",
            "season finale",
            "season premiere",
            "series finale",
            "all seasons",
            "complete series",
            "entire series",
            "binge watch",
            "binge watching",
        ],
    },
    {
        "name": "book_learning",
        "markers": [
            "book",
            "books",
            "book series",
            "series of books",
            "novel",
            "novels",
            "chapter",
            "chapters",
            "book volumes",
            "volume of books",
            "volumes of books",
        ],
    },
    {
        "name": "movie_learning",
        "markers": [
            "movie",
            "movies",
            "film",
            "films",
            "cinema",
            "cinematic",
            "movie series",
            "film series",
            "series of movies",
            "trilogy",
            "saga",
        ],
    },
    {
        "name": "anime_learning",
        "markers": [
            "anime",
            "anime series",
            "anime episodes",
            "anime season",
            "anime seasons",
            "manga",
            "otaku",
        ],
    },
    {
        "name": "experience",
        "markers": [
            "experience",
            "experienced",
            "feel",
            "felt",
            "feeling",
            "feelings",
            "emotion",
            "emotional",
            "affect",
            "affective",
            "sensation",
            "sensations",
            "sensory",
            "visceral",
            "memory",
            "memories",
            "lived",
        ],
    },
    {
        "name": "emotional_affect",
        "markers": [
            "emotion",
            "emotional",
            "affect",
            "affective",
            "mood",
            "feeling",
            "feelings",
        ],
    },
    {
        "name": "personal_experience",
        "markers": [
            "my experience",
            "my own experience",
            "own experience",
            "personal experience",
            "lived experience",
            "firsthand",
            "personally",
        ],
    },
    {
        "name": "hardship",
        "markers": [
            "hardship",
            "hardships",
            "struggle",
            "struggles",
            "struggled",
            "pain",
            "trauma",
            "loss",
            "grief",
            "adversity",
            "suffering",
            "my hardship",
            "my hardships",
        ],
    },
    {
        "name": "social",
        "markers": ["call", "talk", "meet", "friend", "family", "hangout", "chat"],
    },
    {
        "name": "finance",
        "markers": ["budget", "save", "spend", "invest", "pay", "bill", "expense"],
    },
    {
        "name": "household",
        "markers": ["laundry", "dishes", "vacuum", "clean", "shop", "groceries", "cook"],
    },
    {
        "name": "digital",
        "markers": ["scroll", "browse", "screen", "phone", "social media", "message", "email"],
    },
    {
        "name": "mindset",
        "markers": ["meditate", "journal", "reflect", "plan", "gratitude", "mindful"],
    },
    {
        "name": "spiritual",
        "markers": [
            "divine",
            "divinity",
            "holy",
            "sacred",
            "spiritual",
            "spirituality",
            "deity",
            "god",
            "gods",
            "heavenly",
            "heaven",
            "prayer",
            "blessing",
            "blessed",
        ],
    },
    {
        "name": "leisure",
        "markers": ["game", "play", "music", "movie", "tv", "hobby", "relax"],
    },
]

ORIGIN_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "home",
        "markers": ["home", "house", "bedroom", "kitchen", "family", "parent", "parents"],
    },
    {
        "name": "school",
        "markers": ["school", "class", "classroom", "college", "university", "campus", "teacher"],
    },
    {
        "name": "workplace",
        "markers": ["work", "job", "office", "coworker", "manager", "client", "shift"],
    },
    {
        "name": "community",
        "markers": ["community", "neighborhood", "friends", "group", "club", "team"],
    },
    {
        "name": "environment",
        "markers": ["environment", "environmental", "setting", "surroundings", "context"],
    },
    {
        "name": "digital",
        "markers": ["online", "internet", "web", "app", "phone", "device", "social media"],
    },
    {
        "name": "outdoors",
        "markers": ["outdoors", "nature", "park", "trail", "weather", "season"],
    },
    {
        "name": "healthcare",
        "markers": ["doctor", "clinic", "hospital", "therapy", "medical", "rehab"],
    },
    {
        "name": "religious",
        "markers": ["church", "temple", "mosque", "prayer", "faith", "religious"],
    },
    {
        "name": "divine",
        "markers": [
            "divine",
            "divinity",
            "holy",
            "sacred",
            "spiritual",
            "spirituality",
            "deity",
            "god",
            "gods",
            "heavenly",
            "heaven",
        ],
    },
    {
        "name": "training",
        "markers": ["training", "coach", "mentor", "workshop", "bootcamp"],
    },
    {
        "name": "travel",
        "markers": ["travel", "trip", "vacation", "airport", "hotel", "journey"],
    },
]

DENDRITE_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "dendritic",
        "markers": ["dendrite", "dendritic", "branching"],
    },
    {
        "name": "synaptic",
        "markers": ["synapse", "synaptic", "synapses"],
    },
    {
        "name": "neural",
        "markers": ["neuron", "neuronal", "neural", "brain"],
    },
    {
        "name": "cortical",
        "markers": ["cortex", "cortical", "prefrontal"],
    },
    {
        "name": "hippocampal",
        "markers": ["hippocampus", "hippocampal"],
    },
    {
        "name": "prefrontal",
        "markers": ["prefrontal", "prefrontal cortex"],
    },
    {
        "name": "basal_ganglia",
        "markers": ["basal ganglia"],
    },
    {
        "name": "striatal",
        "markers": ["striatum", "striatal"],
    },
    {
        "name": "cerebellum",
        "markers": ["cerebellum", "cerebellar"],
    },
    {
        "name": "amygdala",
        "markers": ["amygdala"],
    },
    {
        "name": "thalamic",
        "markers": ["thalamus", "thalamic"],
    },
    {
        "name": "motor_cortex",
        "markers": ["motor cortex"],
    },
    {
        "name": "sensory_cortex",
        "markers": ["sensory cortex"],
    },
    {
        "name": "brain_region",
        "markers": ["brain region", "neuroanatomy", "neuroanatomical"],
    },
    {
        "name": "myelin",
        "markers": ["myelin", "myelination"],
    },
]

TIMEFRAME_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "time_of_day",
        "markers": [
            "morning",
            "afternoon",
            "evening",
            "night",
            "tonight",
            "midnight",
            "noon",
            "dawn",
            "dusk",
        ],
    },
    {
        "name": "daily",
        "markers": ["today", "yesterday", "tomorrow", "daily", "each day", "every day"],
    },
    {
        "name": "weekly",
        "markers": ["week", "weekly", "last week", "next week", "this week"],
    },
    {
        "name": "monthly",
        "markers": ["month", "monthly", "last month", "next month", "this month"],
    },
    {
        "name": "yearly",
        "markers": ["year", "yearly", "last year", "next year", "this year"],
    },
    {
        "name": "seasonal",
        "markers": ["spring", "summer", "fall", "autumn", "winter", "season"],
    },
    {
        "name": "recent",
        "markers": ["recent", "recently", "lately", "earlier", "previously"],
    },
    {
        "name": "long_term",
        "markers": ["long term", "over time", "for years", "decades"],
    },
]

ONSET_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "childhood",
        "markers": ["childhood", "as a child", "kid", "kids"],
    },
    {
        "name": "teen",
        "markers": ["teen", "teenager", "adolescence", "high school"],
    },
    {
        "name": "adult",
        "markers": ["adult", "adulthood"],
    },
    {
        "name": "college",
        "markers": ["college", "university", "campus"],
    },
    {
        "name": "early",
        "markers": ["early", "initial", "beginning", "start"],
    },
    {
        "name": "late",
        "markers": ["late", "later", "recently"],
    },
]

ONSET_MARKERS = [
    "started",
    "start",
    "began",
    "begin",
    "since",
    "from",
    "when i started",
    "when we started",
    "developed",
    "developing",
    "formed",
    "forming",
]

FREQUENCY_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "rare",
        "markers": ["rarely", "seldom", "hardly ever"],
    },
    {
        "name": "occasional",
        "markers": ["occasionally", "once in a while", "sometimes"],
    },
    {
        "name": "regular",
        "markers": ["regularly", "consistent", "routine"],
    },
    {
        "name": "frequent",
        "markers": ["often", "frequently", "many times"],
    },
    {
        "name": "daily",
        "markers": ["daily", "every day", "each day"],
    },
    {
        "name": "weekly",
        "markers": ["weekly", "every week", "each week"],
    },
    {
        "name": "monthly",
        "markers": ["monthly", "every month", "each month"],
    },
    {
        "name": "yearly",
        "markers": ["yearly", "every year", "each year"],
    },
    {
        "name": "hourly",
        "markers": ["hourly", "every hour", "each hour"],
    },
    {
        "name": "multi_per_day",
        "markers": ["twice", "three times", "four times", "several times"],
    },
]

NEUROPLASTIC_MARKERS = [
    "neuroplastic",
    "neuroplasticity",
    "plasticity",
    "synaptic plasticity",
    "long-term potentiation",
    "long-term depression",
    "ltp",
    "ltd",
    "rewire",
    "rewiring",
]

NEUROTRANSMITTER_MARKERS = [
    "neurotransmitter",
    "neurotransmitters",
    "neuromodulator",
    "neuromodulation",
    "dopamine",
    "serotonin",
    "gaba",
    "glutamate",
    "acetylcholine",
    "norepinephrine",
    "noradrenaline",
    "epinephrine",
    "adrenaline",
    "oxytocin",
    "endorphin",
    "histamine",
]

NEURAL_FIRING_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "firing_rate",
        "markers": ["firing", "firing rate", "spike rate"],
    },
    {
        "name": "neuroplasticity",
        "markers": NEUROPLASTIC_MARKERS,
    },
    {
        "name": "neurotransmitter",
        "markers": NEUROTRANSMITTER_MARKERS,
    },
    {
        "name": "spiking",
        "markers": ["spike", "spiking", "spike train"],
    },
    {
        "name": "bursting",
        "markers": ["burst", "bursting", "burst rate"],
    },
    {
        "name": "synaptic",
        "markers": ["synaptic firing", "synapse", "synaptic"],
    },
    {
        "name": "oscillation",
        "markers": ["oscillation", "oscillatory", "rhythm", "gamma", "theta", "alpha", "beta", "delta"],
    },
]

MUSCLE_MEMORY_MARKERS = [
    "muscle",
    "muscles",
    "muscle memory",
    "motor memory",
    "kinesthetic",
    "proprioception",
    "proprioceptive",
    "movement",
    "motion",
    "reflex",
    "reaction",
    "coordination",
    "motor skill",
    "motor skills",
    "motor control",
    "motor program",
    "motor pattern",
    "motor learning",
]

CHANGE_MARKERS = [
    "change",
    "changes",
    "changed",
    "changing",
    "shift",
    "shifted",
    "adapt",
    "adapted",
    "adjust",
    "adjusted",
    "improve",
    "improved",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "slower",
    "faster",
]

AUTOMATIC_MARKERS = [
    "automatic",
    "automatically",
    "instinct",
    "instinctive",
    "reflex",
    "reflexive",
    "involuntary",
    "autopilot",
]

SUBCONSCIOUS_MARKERS = [
    "subconscious",
    "unconscious",
    "nonconscious",
    "implicit",
]

SUBLIMINAL_MARKERS = [
    "subliminal",
    "subthreshold",
    "below awareness",
]

AUTOIMMUNE_MARKERS = [
    "autoimmune",
    "immune",
    "immunity",
    "immune system",
    "autoantibody",
    "inflammation",
    "cytokine",
]

REINFORCEMENT_MARKERS = [
    "reinforce",
    "reinforced",
    "reinforcement",
    "reward",
    "rewarded",
    "conditioning",
    "operant",
]

MISTAKE_MARKERS = [
    "mistake",
    "mistakes",
    "error",
    "errors",
    "misstep",
    "missteps",
    "fault",
    "faults",
]

MOTOR_SKILL_MARKERS = [
    "motor skill",
    "motor skills",
    "motor control",
    "motor program",
    "motor pattern",
    "motor learning",
]

MOTOR_ACTIVATION_MARKERS = [
    "activate",
    "activation",
    "initiated",
    "initiate",
    "trigger",
    "triggered",
    "engage",
    "engaged",
]

BRAIN_REGION_MARKERS = [
    "brain region",
    "neuroanatomy",
    "neuroanatomical",
    "prefrontal",
    "prefrontal cortex",
    "basal ganglia",
    "striatum",
    "striatal",
    "cerebellum",
    "cerebellar",
    "amygdala",
    "thalamus",
    "thalamic",
    "motor cortex",
    "sensory cortex",
    "cortical",
    "hippocampus",
    "hippocampal",
]

RESPONSE_MARKERS = [
    "response",
    "responses",
    "respond",
    "responded",
    "responding",
    "react",
    "reacted",
    "reacting",
    "reaction",
]

PASSIVE_MARKERS = [
    "passive",
    "passively",
    "inactive",
    "inactivity",
    "idle",
    "low effort",
]

AGGRESSIVE_MARKERS = [
    "aggressive",
    "aggression",
    "hostile",
    "angry",
    "anger",
    "rage",
    "attack",
    "violent",
]

INTUITIVE_MARKERS = [
    "intuitive",
    "intuition",
    "gut feeling",
    "sixth sense",
]

PRECOGNITIVE_MARKERS = [
    "precognitive",
    "premonition",
    "foresee",
    "foresight",
    "foretelling",
]

AFFECT_MARKERS = [
    "feel",
    "felt",
    "feeling",
    "feelings",
    "emotion",
    "emotional",
    "affect",
    "affective",
    "mood",
    "visceral",
]

PATTERN_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "pattern",
        "markers": ["pattern", "patterned"],
    },
    {
        "name": "cycle",
        "markers": ["cycle", "cyclical", "loop"],
    },
    {
        "name": "rhythm",
        "markers": ["rhythm", "rhythmic", "tempo", "cadence"],
    },
    {
        "name": "sequence",
        "markers": ["sequence", "ordered", "steps", "stepwise"],
    },
    {
        "name": "structure",
        "markers": ["structure", "structured", "pattern structure"],
    },
    {
        "name": "consistency",
        "markers": [
            "consistent",
            "consistency",
            "reinforced",
            "reinforcement",
            "operant conditioning",
            "conditioning",
        ],
    },
]

EVENT_MARKERS = [
    "event",
    "events",
    "occurrence",
    "occurrences",
    "incident",
    "episode",
    "situation",
    "scenario",
    "happening",
    "happened",
    "chain",
]

STRUCTURE_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "cue",
        "markers": ["cue", "trigger", "prompt", "stimulus", "signal"],
    },
    {
        "name": "routine",
        "markers": ["routine", "ritual", "procedure", "process", "steps", "stepwise"],
    },
    {
        "name": "reward",
        "markers": ["reward", "reinforce", "reinforcement", "payoff", "benefit"],
    },
    {
        "name": "loop",
        "markers": ["habit loop", "loop", "cycle", "cyclical"],
    },
    {
        "name": "sequence",
        "markers": ["sequence", "ordered", "first", "next", "then", "after"],
    },
    {
        "name": "structure",
        "markers": ["structure", "structured", "framework", "format"],
    },
    {
        "name": "filtering",
        "markers": [
            "filter",
            "filtered",
            "filtering",
            "exclude",
            "excluded",
            "include",
            "included",
            "selection",
            "selective",
            "screened",
        ],
    },
]

EXTERNAL_FORCE_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "social_pressure",
        "markers": [
            "peer pressure",
            "social pressure",
            "others",
            "people",
            "friends",
            "family",
            "expectation",
        ],
    },
    {
        "name": "environmental",
        "markers": [
            "environment",
            "surroundings",
            "context",
            "setting",
            "place",
            "weather",
            "noise",
            "temperature",
        ],
    },
    {
        "name": "cultural",
        "markers": ["culture", "cultural", "tradition", "norms", "society"],
    },
    {
        "name": "authority",
        "markers": ["authority", "boss", "manager", "teacher", "rules", "policy", "law"],
    },
    {
        "name": "media",
        "markers": ["media", "advertising", "ads", "marketing", "social media", "news"],
    },
    {
        "name": "financial",
        "markers": ["financial", "money", "cost", "expense", "budget", "price"],
    },
    {
        "name": "time_pressure",
        "markers": ["deadline", "schedule", "time pressure", "time constraint"],
    },
    {
        "name": "external_force",
        "markers": ["external", "outside", "forced", "pressure", "pressured", "coerced"],
    },
]

INFLUENCE_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "trigger",
        "markers": ["trigger", "cue", "prompt", "stimulus", "signal"],
    },
    {
        "name": "emotion",
        "markers": ["stress", "anxiety", "bored", "boredom", "anger", "sad", "happy", "emotion"],
    },
    {
        "name": "motivation",
        "markers": ["motivation", "desire", "want", "goal", "intention"],
    },
    {
        "name": "reward",
        "markers": ["reward", "reinforcement", "payoff", "benefit"],
    },
    {
        "name": "pressure",
        "markers": ["pressure", "pressured", "forced", "coerced"],
    },
    {
        "name": "environment",
        "markers": ["environment", "setting", "context", "place", "weather"],
    },
    {
        "name": "social",
        "markers": ["social", "friends", "family", "others", "peer"],
    },
    {
        "name": "biological",
        "markers": ["sleep", "hunger", "tired", "fatigue", "energy", "hormone", "craving"],
    },
    {
        "name": "autoimmune_response",
        "markers": AUTOIMMUNE_MARKERS,
    },
    {
        "name": "cognitive",
        "markers": ["belief", "thought", "mindset", "decision", "plan"],
    },
    {
        "name": "cause",
        "markers": ["because", "due to", "caused by", "as a result", "therefore"],
    },
]

SOCIAL_LEARNING_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "observational",
        "markers": ["observe", "observed", "watch", "watching", "seeing", "saw", "noticed"],
    },
    {
        "name": "instructional",
        "markers": ["taught", "teach", "instruction", "guidance", "coach", "mentor", "teacher"],
    },
    {
        "name": "imitative",
        "markers": ["imitate", "imitation", "copy", "copied", "mimic", "modeled", "modeling"],
    },
    {
        "name": "peer",
        "markers": ["peer", "friend", "coworker", "classmate", "team"],
    },
    {
        "name": "family",
        "markers": ["parent", "mother", "father", "family", "sibling"],
    },
    {
        "name": "influence",
        "markers": ["learned from", "influenced by", "picked up from"],
    },
    {
        "name": "unsupervised",
        "markers": [
            "unsupervised",
            "unguided",
            "self guided",
            "self-guided",
            "self directed",
            "self-directed",
            "autonomous",
            "no supervision",
            "without supervision",
        ],
    },
    {
        "name": "unsupervised_retention",
        "markers": ["unsupervised retention", "retention without supervision"],
    },
    {
        "name": "unsupervised_learning",
        "markers": [
            "unsupervised learning",
            "self learning",
            "self-learning",
            "self taught",
            "self-taught",
        ],
    },
    {
        "name": "unsupervised_behavior",
        "markers": ["unsupervised behavior", "unsupervised behaviors", "autonomous behavior"],
    },
]

PROVENANCE_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "family_origin",
        "markers": ["parent", "mother", "father", "family", "sibling"],
    },
    {
        "name": "mentor_origin",
        "markers": ["mentor", "coach", "teacher", "trainer", "advisor"],
    },
    {
        "name": "peer_origin",
        "markers": ["friend", "peer", "coworker", "classmate", "teammate"],
    },
    {
        "name": "authority_origin",
        "markers": ["boss", "manager", "leader", "authority", "supervisor"],
    },
    {
        "name": "partner_origin",
        "markers": ["partner", "spouse", "wife", "husband"],
    },
    {
        "name": "self_origin",
        "markers": ["self", "myself", "independent", "on my own"],
    },
]

PROVENANCE_MARKERS = [
    "from",
    "origin",
    "originated",
    "stem",
    "stemmed",
    "source",
    "learned from",
    "taught by",
]

ACTIVATION_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "trigger",
        "markers": ["trigger", "cue", "prompt", "stimulus", "signal"],
    },
    {
        "name": "initiation",
        "markers": ["activate", "activation", "initiated", "initiate", "kick off", "start"],
    },
    {
        "name": "switch",
        "markers": ["switch", "flip", "turn on", "turns on"],
    },
    {
        "name": "arousal",
        "markers": ["arousal", "energize", "energized", "wake up"],
    },
]

DENDRO_MOTOR_MARKERS = [
    "response",
    "reaction",
    "reflex",
    "output",
    "activation",
    "activated",
    "activate",
]

DENDRITE_ACTION_MARKERS = [
    "action",
    "actions",
    "act",
    "acting",
    "activation",
    "activate",
    "activated",
]

MUSIC_MARKERS = [
    "music",
    "song",
    "songs",
    "listen",
    "listening",
    "listen to",
    "track",
    "tune",
    "melody",
    "rhythm",
    "beat",
    "lyrics",
    "playlist",
    "album",
    "chorus",
    "hook",
    "same song",
    "repeat song",
    "replay",
    "on repeat",
]

INTENT_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "intent",
        "markers": ["intent", "intention", "intend"],
    },
    {
        "name": "goal",
        "markers": ["goal", "target", "objective"],
    },
    {
        "name": "purpose",
        "markers": ["purpose", "reason", "aim"],
    },
    {
        "name": "volition",
        "markers": ["volition", "willpower", "will", "desire", "want"],
    },
    {
        "name": "motivation",
        "markers": ["motivation", "motivated", "drive", "driven", "ambition", "determined"],
    },
    {
        "name": "decision",
        "markers": ["decide", "decision", "choose", "choice"],
    },
    {
        "name": "control",
        "markers": [
            "control",
            "controlled",
            "manage",
            "managed",
            "regulate",
            "regulated",
            "agency",
            "autonomy",
            "govern",
        ],
    },
    {
        "name": "discipline_focus",
        "markers": [
            "discipline",
            "disciplined",
            "focus",
            "focused",
            "concentrate",
            "concentrated",
            "concentration",
            "attention",
            "attentive",
        ],
    },
]

ACTION_MARKERS = [
    "action",
    "actions",
    "act",
    "acted",
    "acting",
    "behavior",
    "behaviour",
    "response",
    "responses",
    "respond",
    "responded",
    "responding",
    "reaction",
    "react",
    "reacted",
    "reacting",
    "do",
    "does",
    "did",
    "make",
    "makes",
    "made",
    "perform",
    "performed",
    "performing",
]

WORD_MARKERS = [
    "word",
    "words",
    "say",
    "said",
    "says",
    "spoke",
    "spoken",
    "speaking",
    "tell",
    "told",
    "telling",
    "speak",
    "speech",
    "talk",
    "phrase",
    "statement",
    "message",
    "instruction",
    "command",
    "request",
    "ask",
    "asked",
    "asking",
    "verbal",
    "language",
    "utter",
    "uttered",
    "utterance",
    "vocalized",
]

UNDERSTAND_MARKERS = [
    "understand",
    "understood",
    "comprehend",
    "comprehended",
    "interpret",
    "interpreted",
    "grasp",
    "grasped",
]

PERFORM_MARKERS = [
    "perform",
    "performed",
    "performing",
    "execute",
    "executed",
    "executing",
    "carry out",
    "carried out",
]

SOCIAL_MARKERS = [
    "someone",
    "person",
    "people",
    "others",
    "another",
    "they",
    "their",
    "friend",
    "family",
    "parent",
    "teacher",
    "boss",
    "coworker",
    "partner",
    "spouse",
]

SELF_MARKERS = [
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
]

BELIEF_MARKERS = [
    "belief",
    "beliefs",
    "believe",
    "believed",
    "faith",
    "value",
    "values",
    "moral",
    "morals",
    "principle",
    "principles",
    "conviction",
    "ideology",
]

DIVINE_MARKERS = [
    "divine",
    "divinity",
    "holy",
    "sacred",
    "spiritual",
    "spirituality",
    "deity",
    "god",
    "gods",
    "godly",
    "heaven",
    "heavenly",
    "blessing",
    "blessed",
]

TRUTH_MARKERS = [
    "truth",
    "truths",
    "truthful",
    "truthfulness",
    "veracity",
]

VIRTUE_MARKERS = [
    "virtue",
    "virtues",
    "kindness",
    "patience",
    "courage",
    "honesty",
    "integrity",
    "temperance",
    "justice",
    "compassion",
    "discipline",
]

OUTER_THINKING_MARKERS = [
    "outer thinking",
    "external thinking",
    "outside thinking",
    "external thought",
    "outer thoughts",
    "surface thinking",
    "perception",
    "observing",
    "observer",
]

INNER_BELIEF_MARKERS = [
    "inner",
    "internal",
    "inward",
    "within",
    "inside",
    "self",
    "personal",
    "core",
    "private",
]

OUTER_BELIEF_MARKERS = [
    "outer",
    "external",
    "outside",
    "outward",
    "public",
    "social",
    "cultural",
    "community",
    "societal",
]

INNER_THINKING_MARKERS = [
    "inner thinking",
    "inner thought",
    "inner thoughts",
    "internal thinking",
    "internal thought",
    "internal thoughts",
    "inner mind",
    "inner monologue",
    "inner voice",
    "self talk",
    "self-talk",
    "introspective",
    "introspection",
    "self reflection",
    "self-reflection",
    "ruminate",
    "rumination",
]

MUSIC_RESONANCE_MARKERS = [
    "resonate",
    "resonates",
    "resonance",
    "echo",
    "echoes",
    "align",
    "aligned",
    "sync",
    "in sync",
    "harmony",
    "harmonic",
]

INFLUENCE_MARKERS = [
    "influence",
    "influenced",
    "impact",
    "impacted",
    "affect",
    "affected",
    "effect",
    "pressure",
    "persuade",
    "persuaded",
    "sway",
    "shape",
    "shaped",
]

DEEP_RESONANCE_MARKERS = [
    "deep",
    "deeply",
    "profound",
    "fundamental",
    "foundational",
    "root",
    "core",
]

CONCENTRATION_MARKERS = [
    "concentrate",
    "concentrated",
    "concentration",
    "focus",
    "focused",
    "intense",
    "intensely",
    "single-minded",
    "singleminded",
]

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
DURATION_RE = re.compile(r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\b", re.I)
FREQUENCY_RE = re.compile(r"\b\d+\s*(?:x|times)\s*(?:a|per)?\s*(day|week|month|year)\b", re.I)
FREQUENCY_SLASH_RE = re.compile(r"\b\d+\s*/\s*(day|week|month|year)\b", re.I)


class ContextHabitCategorizationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum bytes across utterances."
    )
    max_utterances: int = Field(default=500, description="Maximum utterances to process.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=10, description="Max keywords per habit.")
    max_categories_per_habit: int = Field(
        default=3, description="Max categories per habit."
    )
    max_category_terms: int = Field(
        default=8, description="Max terms per category bucket."
    )
    max_origin_categories_per_habit: int = Field(
        default=2, description="Max origin categories per habit."
    )
    max_origin_terms: int = Field(
        default=8, description="Max terms per origin bucket."
    )
    max_dendrite_categories_per_habit: int = Field(
        default=2, description="Max dendrite categories per habit."
    )
    max_dendrite_terms: int = Field(
        default=8, description="Max terms per dendrite bucket."
    )
    max_timeframe_categories_per_habit: int = Field(
        default=2, description="Max timeframe categories per habit."
    )
    max_timeframe_terms: int = Field(
        default=8, description="Max terms per timeframe bucket."
    )
    max_onset_categories_per_habit: int = Field(
        default=2, description="Max onset categories per habit."
    )
    max_onset_terms: int = Field(
        default=8, description="Max terms per onset bucket."
    )
    max_frequency_categories_per_habit: int = Field(
        default=2, description="Max frequency categories per habit."
    )
    max_frequency_terms: int = Field(
        default=8, description="Max terms per frequency bucket."
    )
    max_neural_firing_categories_per_habit: int = Field(
        default=2, description="Max neural firing categories per habit."
    )
    max_neural_firing_terms: int = Field(
        default=8, description="Max terms per neural firing bucket."
    )
    max_muscle_memory_categories_per_habit: int = Field(
        default=2, description="Max muscle memory categories per habit."
    )
    max_muscle_memory_terms: int = Field(
        default=8, description="Max terms per muscle memory bucket."
    )
    max_pattern_categories_per_habit: int = Field(
        default=2, description="Max habit pattern categories per habit."
    )
    max_pattern_terms: int = Field(
        default=8, description="Max terms per habit pattern bucket."
    )
    max_structure_categories_per_habit: int = Field(
        default=2, description="Max habit structure categories per habit."
    )
    max_structure_terms: int = Field(
        default=8, description="Max terms per habit structure bucket."
    )
    max_external_force_categories_per_habit: int = Field(
        default=2, description="Max external force categories per habit."
    )
    max_external_force_terms: int = Field(
        default=8, description="Max terms per external force bucket."
    )
    max_influence_categories_per_habit: int = Field(
        default=2, description="Max influence categories per habit."
    )
    max_influence_terms: int = Field(
        default=8, description="Max terms per influence bucket."
    )
    max_external_repetition_categories_per_habit: int = Field(
        default=2, description="Max external repetition categories per habit."
    )
    max_external_repetition_terms: int = Field(
        default=8, description="Max terms per external repetition bucket."
    )
    max_event_sequence_categories_per_habit: int = Field(
        default=2, description="Max event sequence categories per habit."
    )
    max_event_sequence_terms: int = Field(
        default=8, description="Max terms per event sequence bucket."
    )
    max_social_learning_categories_per_habit: int = Field(
        default=2, description="Max social learning categories per habit."
    )
    max_social_learning_terms: int = Field(
        default=8, description="Max terms per social learning bucket."
    )
    max_provenance_categories_per_habit: int = Field(
        default=2, description="Max provenance categories per habit."
    )
    max_provenance_terms: int = Field(
        default=8, description="Max terms per provenance bucket."
    )
    max_activation_categories_per_habit: int = Field(
        default=2, description="Max activation categories per habit."
    )
    max_activation_terms: int = Field(
        default=8, description="Max terms per activation bucket."
    )
    max_dendro_motor_categories_per_habit: int = Field(
        default=2, description="Max dendro-motor categories per habit."
    )
    max_dendro_motor_terms: int = Field(
        default=8, description="Max terms per dendro-motor bucket."
    )
    max_music_categories_per_habit: int = Field(
        default=2, description="Max music habit categories per habit."
    )
    max_music_terms: int = Field(
        default=8, description="Max terms per music habit bucket."
    )
    max_intent_categories_per_habit: int = Field(
        default=2, description="Max intent categories per habit."
    )
    max_intent_terms: int = Field(
        default=8, description="Max terms per intent bucket."
    )
    max_action_categories_per_habit: int = Field(
        default=2, description="Max action categories per habit."
    )
    max_action_terms: int = Field(
        default=8, description="Max terms per action bucket."
    )
    max_word_action_categories_per_habit: int = Field(
        default=2, description="Max word-action categories per habit."
    )
    max_word_action_terms: int = Field(
        default=8, description="Max terms per word-action bucket."
    )
    max_social_word_categories_per_habit: int = Field(
        default=2, description="Max social word influence categories per habit."
    )
    max_social_word_terms: int = Field(
        default=8, description="Max terms per social word influence bucket."
    )
    max_music_belief_action_categories_per_habit: int = Field(
        default=2, description="Max music-belief-action categories per habit."
    )
    max_music_belief_action_terms: int = Field(
        default=8, description="Max terms per music-belief-action bucket."
    )
    max_belief_categories_per_habit: int = Field(
        default=2, description="Max belief categories per habit."
    )
    max_belief_terms: int = Field(
        default=8, description="Max terms per belief bucket."
    )
    max_virtue_categories_per_habit: int = Field(
        default=2, description="Max virtue categories per habit."
    )
    max_virtue_terms: int = Field(
        default=8, description="Max terms per virtue bucket."
    )
    max_outer_thinking_categories_per_habit: int = Field(
        default=2, description="Max outer thinking categories per habit."
    )
    max_outer_thinking_terms: int = Field(
        default=8, description="Max terms per outer thinking bucket."
    )
    max_inner_thinking_categories_per_habit: int = Field(
        default=2, description="Max inner thinking categories per habit."
    )
    max_inner_thinking_terms: int = Field(
        default=8, description="Max terms per inner thinking bucket."
    )
    max_topics: int = Field(default=15, description="Max topics to include.")
    max_sequences: int = Field(default=20, description="Max sequences to include.")
    max_steps_per_sequence: int = Field(
        default=10, description="Max steps per sequence."
    )
    max_category_segments: int = Field(
        default=8, description="Max category speech segments."
    )
    max_sequence_segments: int = Field(
        default=8, description="Max sequence speech segments."
    )
    max_origin_segments: int = Field(
        default=8, description="Max origin speech segments."
    )
    max_dendrite_segments: int = Field(
        default=8, description="Max dendrite speech segments."
    )
    max_timeframe_segments: int = Field(
        default=8, description="Max timeframe speech segments."
    )
    max_onset_segments: int = Field(
        default=8, description="Max onset speech segments."
    )
    max_frequency_segments: int = Field(
        default=8, description="Max frequency speech segments."
    )
    max_neural_firing_segments: int = Field(
        default=8, description="Max neural firing speech segments."
    )
    max_muscle_memory_segments: int = Field(
        default=8, description="Max muscle memory speech segments."
    )
    max_pattern_segments: int = Field(
        default=8, description="Max habit pattern speech segments."
    )
    max_structure_segments: int = Field(
        default=8, description="Max habit structure speech segments."
    )
    max_external_force_segments: int = Field(
        default=8, description="Max external force speech segments."
    )
    max_influence_segments: int = Field(
        default=8, description="Max influence speech segments."
    )
    max_external_repetition_segments: int = Field(
        default=8, description="Max external repetition speech segments."
    )
    max_event_sequence_segments: int = Field(
        default=8, description="Max event sequence speech segments."
    )
    max_social_learning_segments: int = Field(
        default=8, description="Max social learning speech segments."
    )
    max_provenance_segments: int = Field(
        default=8, description="Max provenance speech segments."
    )
    max_activation_segments: int = Field(
        default=8, description="Max activation speech segments."
    )
    max_dendro_motor_segments: int = Field(
        default=8, description="Max dendro-motor speech segments."
    )
    max_music_segments: int = Field(
        default=8, description="Max music habit speech segments."
    )
    max_intent_segments: int = Field(
        default=8, description="Max intent speech segments."
    )
    max_action_segments: int = Field(
        default=8, description="Max action speech segments."
    )
    max_word_action_segments: int = Field(
        default=8, description="Max word-action speech segments."
    )
    max_social_word_segments: int = Field(
        default=8, description="Max social word influence speech segments."
    )
    max_music_belief_action_segments: int = Field(
        default=8, description="Max music-belief-action speech segments."
    )
    max_belief_segments: int = Field(
        default=8, description="Max belief speech segments."
    )
    max_virtue_segments: int = Field(
        default=8, description="Max virtue speech segments."
    )
    max_outer_thinking_segments: int = Field(
        default=8, description="Max outer thinking speech segments."
    )
    max_inner_thinking_segments: int = Field(
        default=8, description="Max inner thinking speech segments."
    )
    max_repetition_segments: int = Field(
        default=8, description="Max repetition speech segments."
    )
    max_habit_segments: int = Field(
        default=8, description="Max habit speech segments."
    )
    max_speech_segments: int = Field(
        default=40, description="Max total speech segments."
    )
    default_split_mode: str = Field(
        default="lines", description="lines or sentences."
    )
    include_singleton_sequences: bool = Field(
        default=False, description="Include single-step sequences."
    )


class ContextHabitCategorizationState(BaseToolState):
    pass


class HabitUtterance(BaseModel):
    text: str = Field(description="Habit statement text.")
    speaker: str | None = Field(default=None, description="Speaker label.")
    timestamp: str | None = Field(default=None, description="Optional timestamp.")


class ContextHabitCategorizationArgs(BaseModel):
    content: str | None = Field(default=None, description="Conversation content.")
    path: str | None = Field(default=None, description="Path to transcript file.")
    utterances: list[HabitUtterance] | None = Field(
        default=None, description="Explicit habit utterances."
    )
    split_mode: str | None = Field(
        default=None, description="lines or sentences."
    )
    include_sequences: bool = Field(
        default=True, description="Include sequence categorization."
    )
    include_singleton_sequences: bool | None = Field(
        default=None, description="Override include_singleton_sequences."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_utterances: int | None = Field(
        default=None, description="Override max_utterances."
    )
    min_token_length: int | None = Field(
        default=None, description="Override min_token_length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )
    max_categories_per_habit: int | None = Field(
        default=None, description="Override max_categories_per_habit."
    )
    max_category_terms: int | None = Field(
        default=None, description="Override max_category_terms."
    )
    max_origin_categories_per_habit: int | None = Field(
        default=None, description="Override max_origin_categories_per_habit."
    )
    max_origin_terms: int | None = Field(
        default=None, description="Override max_origin_terms."
    )
    max_dendrite_categories_per_habit: int | None = Field(
        default=None, description="Override max_dendrite_categories_per_habit."
    )
    max_dendrite_terms: int | None = Field(
        default=None, description="Override max_dendrite_terms."
    )
    max_timeframe_categories_per_habit: int | None = Field(
        default=None, description="Override max_timeframe_categories_per_habit."
    )
    max_timeframe_terms: int | None = Field(
        default=None, description="Override max_timeframe_terms."
    )
    max_onset_categories_per_habit: int | None = Field(
        default=None, description="Override max_onset_categories_per_habit."
    )
    max_onset_terms: int | None = Field(
        default=None, description="Override max_onset_terms."
    )
    max_frequency_categories_per_habit: int | None = Field(
        default=None, description="Override max_frequency_categories_per_habit."
    )
    max_frequency_terms: int | None = Field(
        default=None, description="Override max_frequency_terms."
    )
    max_neural_firing_categories_per_habit: int | None = Field(
        default=None, description="Override max_neural_firing_categories_per_habit."
    )
    max_neural_firing_terms: int | None = Field(
        default=None, description="Override max_neural_firing_terms."
    )
    max_muscle_memory_categories_per_habit: int | None = Field(
        default=None, description="Override max_muscle_memory_categories_per_habit."
    )
    max_muscle_memory_terms: int | None = Field(
        default=None, description="Override max_muscle_memory_terms."
    )
    max_pattern_categories_per_habit: int | None = Field(
        default=None, description="Override max_pattern_categories_per_habit."
    )
    max_pattern_terms: int | None = Field(
        default=None, description="Override max_pattern_terms."
    )
    max_structure_categories_per_habit: int | None = Field(
        default=None, description="Override max_structure_categories_per_habit."
    )
    max_structure_terms: int | None = Field(
        default=None, description="Override max_structure_terms."
    )
    max_external_force_categories_per_habit: int | None = Field(
        default=None, description="Override max_external_force_categories_per_habit."
    )
    max_external_force_terms: int | None = Field(
        default=None, description="Override max_external_force_terms."
    )
    max_influence_categories_per_habit: int | None = Field(
        default=None, description="Override max_influence_categories_per_habit."
    )
    max_influence_terms: int | None = Field(
        default=None, description="Override max_influence_terms."
    )
    max_external_repetition_categories_per_habit: int | None = Field(
        default=None, description="Override max_external_repetition_categories_per_habit."
    )
    max_external_repetition_terms: int | None = Field(
        default=None, description="Override max_external_repetition_terms."
    )
    max_event_sequence_categories_per_habit: int | None = Field(
        default=None, description="Override max_event_sequence_categories_per_habit."
    )
    max_event_sequence_terms: int | None = Field(
        default=None, description="Override max_event_sequence_terms."
    )
    max_social_learning_categories_per_habit: int | None = Field(
        default=None, description="Override max_social_learning_categories_per_habit."
    )
    max_social_learning_terms: int | None = Field(
        default=None, description="Override max_social_learning_terms."
    )
    max_provenance_categories_per_habit: int | None = Field(
        default=None, description="Override max_provenance_categories_per_habit."
    )
    max_provenance_terms: int | None = Field(
        default=None, description="Override max_provenance_terms."
    )
    max_activation_categories_per_habit: int | None = Field(
        default=None, description="Override max_activation_categories_per_habit."
    )
    max_activation_terms: int | None = Field(
        default=None, description="Override max_activation_terms."
    )
    max_dendro_motor_categories_per_habit: int | None = Field(
        default=None, description="Override max_dendro_motor_categories_per_habit."
    )
    max_dendro_motor_terms: int | None = Field(
        default=None, description="Override max_dendro_motor_terms."
    )
    max_music_categories_per_habit: int | None = Field(
        default=None, description="Override max_music_categories_per_habit."
    )
    max_music_terms: int | None = Field(
        default=None, description="Override max_music_terms."
    )
    max_intent_categories_per_habit: int | None = Field(
        default=None, description="Override max_intent_categories_per_habit."
    )
    max_intent_terms: int | None = Field(
        default=None, description="Override max_intent_terms."
    )
    max_action_categories_per_habit: int | None = Field(
        default=None, description="Override max_action_categories_per_habit."
    )
    max_action_terms: int | None = Field(
        default=None, description="Override max_action_terms."
    )
    max_word_action_categories_per_habit: int | None = Field(
        default=None, description="Override max_word_action_categories_per_habit."
    )
    max_word_action_terms: int | None = Field(
        default=None, description="Override max_word_action_terms."
    )
    max_social_word_categories_per_habit: int | None = Field(
        default=None, description="Override max_social_word_categories_per_habit."
    )
    max_social_word_terms: int | None = Field(
        default=None, description="Override max_social_word_terms."
    )
    max_music_belief_action_categories_per_habit: int | None = Field(
        default=None, description="Override max_music_belief_action_categories_per_habit."
    )
    max_music_belief_action_terms: int | None = Field(
        default=None, description="Override max_music_belief_action_terms."
    )
    max_belief_categories_per_habit: int | None = Field(
        default=None, description="Override max_belief_categories_per_habit."
    )
    max_belief_terms: int | None = Field(
        default=None, description="Override max_belief_terms."
    )
    max_virtue_categories_per_habit: int | None = Field(
        default=None, description="Override max_virtue_categories_per_habit."
    )
    max_virtue_terms: int | None = Field(
        default=None, description="Override max_virtue_terms."
    )
    max_outer_thinking_categories_per_habit: int | None = Field(
        default=None, description="Override max_outer_thinking_categories_per_habit."
    )
    max_outer_thinking_terms: int | None = Field(
        default=None, description="Override max_outer_thinking_terms."
    )
    max_inner_thinking_categories_per_habit: int | None = Field(
        default=None, description="Override max_inner_thinking_categories_per_habit."
    )
    max_inner_thinking_terms: int | None = Field(
        default=None, description="Override max_inner_thinking_terms."
    )
    max_topics: int | None = Field(default=None, description="Override max_topics.")
    max_sequences: int | None = Field(default=None, description="Override max_sequences.")
    max_steps_per_sequence: int | None = Field(
        default=None, description="Override max_steps_per_sequence."
    )
    max_category_segments: int | None = Field(
        default=None, description="Override max_category_segments."
    )
    max_sequence_segments: int | None = Field(
        default=None, description="Override max_sequence_segments."
    )
    max_origin_segments: int | None = Field(
        default=None, description="Override max_origin_segments."
    )
    max_dendrite_segments: int | None = Field(
        default=None, description="Override max_dendrite_segments."
    )
    max_timeframe_segments: int | None = Field(
        default=None, description="Override max_timeframe_segments."
    )
    max_onset_segments: int | None = Field(
        default=None, description="Override max_onset_segments."
    )
    max_frequency_segments: int | None = Field(
        default=None, description="Override max_frequency_segments."
    )
    max_neural_firing_segments: int | None = Field(
        default=None, description="Override max_neural_firing_segments."
    )
    max_muscle_memory_segments: int | None = Field(
        default=None, description="Override max_muscle_memory_segments."
    )
    max_pattern_segments: int | None = Field(
        default=None, description="Override max_pattern_segments."
    )
    max_structure_segments: int | None = Field(
        default=None, description="Override max_structure_segments."
    )
    max_external_force_segments: int | None = Field(
        default=None, description="Override max_external_force_segments."
    )
    max_influence_segments: int | None = Field(
        default=None, description="Override max_influence_segments."
    )
    max_external_repetition_segments: int | None = Field(
        default=None, description="Override max_external_repetition_segments."
    )
    max_event_sequence_segments: int | None = Field(
        default=None, description="Override max_event_sequence_segments."
    )
    max_social_learning_segments: int | None = Field(
        default=None, description="Override max_social_learning_segments."
    )
    max_provenance_segments: int | None = Field(
        default=None, description="Override max_provenance_segments."
    )
    max_activation_segments: int | None = Field(
        default=None, description="Override max_activation_segments."
    )
    max_dendro_motor_segments: int | None = Field(
        default=None, description="Override max_dendro_motor_segments."
    )
    max_music_segments: int | None = Field(
        default=None, description="Override max_music_segments."
    )
    max_intent_segments: int | None = Field(
        default=None, description="Override max_intent_segments."
    )
    max_action_segments: int | None = Field(
        default=None, description="Override max_action_segments."
    )
    max_word_action_segments: int | None = Field(
        default=None, description="Override max_word_action_segments."
    )
    max_social_word_segments: int | None = Field(
        default=None, description="Override max_social_word_segments."
    )
    max_music_belief_action_segments: int | None = Field(
        default=None, description="Override max_music_belief_action_segments."
    )
    max_belief_segments: int | None = Field(
        default=None, description="Override max_belief_segments."
    )
    max_virtue_segments: int | None = Field(
        default=None, description="Override max_virtue_segments."
    )
    max_outer_thinking_segments: int | None = Field(
        default=None, description="Override max_outer_thinking_segments."
    )
    max_inner_thinking_segments: int | None = Field(
        default=None, description="Override max_inner_thinking_segments."
    )
    max_repetition_segments: int | None = Field(
        default=None, description="Override max_repetition_segments."
    )
    max_habit_segments: int | None = Field(
        default=None, description="Override max_habit_segments."
    )
    max_speech_segments: int | None = Field(
        default=None, description="Override max_speech_segments."
    )
    include_opening: bool = Field(
        default=True, description="Include speech opening."
    )
    include_closing: bool = Field(
        default=True, description="Include speech closing."
    )


class HabitEntry(BaseModel):
    index: int
    speaker: str | None
    text: str
    preview: str
    habit_like: bool
    habit_markers: list[str]
    categories: list[str]
    origin_categories: list[str]
    origin_markers: list[str]
    dendrite_categories: list[str]
    dendrite_markers: list[str]
    timeframe_categories: list[str]
    timeframe_markers: list[str]
    onset_categories: list[str]
    onset_markers: list[str]
    frequency_categories: list[str]
    frequency_markers: list[str]
    neural_firing_categories: list[str]
    neural_firing_markers: list[str]
    muscle_memory_categories: list[str]
    muscle_memory_markers: list[str]
    pattern_categories: list[str]
    pattern_markers: list[str]
    structure_categories: list[str]
    structure_markers: list[str]
    external_force_categories: list[str]
    external_force_markers: list[str]
    influence_categories: list[str]
    influence_markers: list[str]
    external_repetition_categories: list[str]
    external_repetition_markers: list[str]
    event_sequence_categories: list[str]
    event_sequence_markers: list[str]
    social_learning_categories: list[str]
    social_learning_markers: list[str]
    provenance_categories: list[str]
    provenance_markers: list[str]
    activation_categories: list[str]
    activation_markers: list[str]
    dendro_motor_categories: list[str]
    dendro_motor_markers: list[str]
    music_categories: list[str]
    music_markers: list[str]
    intent_categories: list[str]
    intent_markers: list[str]
    action_categories: list[str]
    action_markers: list[str]
    word_action_categories: list[str]
    word_action_markers: list[str]
    social_word_categories: list[str]
    social_word_markers: list[str]
    music_belief_action_categories: list[str]
    music_belief_action_markers: list[str]
    belief_categories: list[str]
    belief_markers: list[str]
    virtue_categories: list[str]
    virtue_markers: list[str]
    outer_thinking_categories: list[str]
    outer_thinking_markers: list[str]
    inner_thinking_categories: list[str]
    inner_thinking_markers: list[str]
    keywords: list[str]
    sequence_markers: list[str]
    repetition_like: bool
    repetition_markers: list[str]


class HabitCategoryBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitTopicBucket(BaseModel):
    term: str
    count: int
    habit_indices: list[int]


class HabitOriginBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitDendriteBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitTimeframeBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitOnsetBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitFrequencyBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitNeuralFiringBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitMuscleMemoryBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitPatternBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitStructureBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitExternalForceBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitInfluenceBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitExternalRepetitionBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitEventSequenceBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitSocialLearningBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitProvenanceBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitActivationBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitDendroMotorBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitMusicBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitIntentBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitActionBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitWordActionBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitSocialWordBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitMusicBeliefActionBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitBeliefBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitVirtueBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitOuterThinkingBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitInnerThinkingBucket(BaseModel):
    category: str
    count: int
    habit_indices: list[int]
    top_terms: list[str]


class HabitSequenceStep(BaseModel):
    index: int
    habit_index: int
    preview: str
    keywords: list[str]
    sequence_markers: list[str]


class HabitSequence(BaseModel):
    index: int
    step_count: int
    habit_indices: list[int]
    sequence_markers: list[str]
    steps: list[HabitSequenceStep]
    cue: str


class HabitSpeechSegment(BaseModel):
    index: int
    kind: str
    cue: str
    habit_indices: list[int]
    categories: list[str]
    topics: list[str]


class ContextHabitCategorizationResult(BaseModel):
    habit_categorization_term: str
    habit_categorization_description: str
    habit_sequence_term: str
    habit_sequence_description: str
    habit_repetition_term: str
    habit_repetition_description: str
    habit_origin_term: str
    habit_origin_description: str
    habit_dendrite_term: str
    habit_dendrite_description: str
    habit_timeframe_term: str
    habit_timeframe_description: str
    habit_onset_term: str
    habit_onset_description: str
    habit_frequency_term: str
    habit_frequency_description: str
    habit_neural_firing_term: str
    habit_neural_firing_description: str
    habit_muscle_memory_term: str
    habit_muscle_memory_description: str
    habit_pattern_term: str
    habit_pattern_description: str
    habit_structure_term: str
    habit_structure_description: str
    habit_external_force_term: str
    habit_external_force_description: str
    habit_influence_term: str
    habit_influence_description: str
    habit_external_repetition_term: str
    habit_external_repetition_description: str
    habit_event_sequence_term: str
    habit_event_sequence_description: str
    habit_social_learning_term: str
    habit_social_learning_description: str
    habit_provenance_term: str
    habit_provenance_description: str
    habit_activation_term: str
    habit_activation_description: str
    habit_dendro_motor_term: str
    habit_dendro_motor_description: str
    habit_music_term: str
    habit_music_description: str
    habit_intent_term: str
    habit_intent_description: str
    habit_action_term: str
    habit_action_description: str
    habit_word_action_term: str
    habit_word_action_description: str
    habit_social_word_term: str
    habit_social_word_description: str
    habit_music_belief_action_term: str
    habit_music_belief_action_description: str
    habit_belief_term: str
    habit_belief_description: str
    habit_virtue_term: str
    habit_virtue_description: str
    habit_outer_thinking_term: str
    habit_outer_thinking_description: str
    habit_inner_thinking_term: str
    habit_inner_thinking_description: str
    habits: list[HabitEntry]
    habit_count: int
    habit_like_count: int
    repetition_like_count: int
    dendrite_like_count: int
    timeframe_like_count: int
    onset_like_count: int
    frequency_like_count: int
    neural_firing_like_count: int
    muscle_memory_like_count: int
    pattern_like_count: int
    structure_like_count: int
    external_force_like_count: int
    influence_like_count: int
    external_repetition_like_count: int
    event_sequence_like_count: int
    social_learning_like_count: int
    provenance_like_count: int
    activation_like_count: int
    dendro_motor_like_count: int
    music_like_count: int
    intent_like_count: int
    action_like_count: int
    word_action_like_count: int
    social_word_like_count: int
    music_belief_action_like_count: int
    belief_like_count: int
    virtue_like_count: int
    outer_thinking_like_count: int
    inner_thinking_like_count: int
    origin_buckets: list[HabitOriginBucket]
    origin_count: int
    dendrite_buckets: list[HabitDendriteBucket]
    dendrite_count: int
    timeframe_buckets: list[HabitTimeframeBucket]
    timeframe_count: int
    onset_buckets: list[HabitOnsetBucket]
    onset_count: int
    frequency_buckets: list[HabitFrequencyBucket]
    frequency_count: int
    neural_firing_buckets: list[HabitNeuralFiringBucket]
    neural_firing_count: int
    muscle_memory_buckets: list[HabitMuscleMemoryBucket]
    muscle_memory_count: int
    pattern_buckets: list[HabitPatternBucket]
    pattern_count: int
    structure_buckets: list[HabitStructureBucket]
    structure_count: int
    external_force_buckets: list[HabitExternalForceBucket]
    external_force_count: int
    influence_buckets: list[HabitInfluenceBucket]
    influence_count: int
    external_repetition_buckets: list[HabitExternalRepetitionBucket]
    external_repetition_count: int
    event_sequence_buckets: list[HabitEventSequenceBucket]
    event_sequence_count: int
    social_learning_buckets: list[HabitSocialLearningBucket]
    social_learning_count: int
    provenance_buckets: list[HabitProvenanceBucket]
    provenance_count: int
    activation_buckets: list[HabitActivationBucket]
    activation_count: int
    dendro_motor_buckets: list[HabitDendroMotorBucket]
    dendro_motor_count: int
    music_buckets: list[HabitMusicBucket]
    music_count: int
    intent_buckets: list[HabitIntentBucket]
    intent_count: int
    action_buckets: list[HabitActionBucket]
    action_count: int
    word_action_buckets: list[HabitWordActionBucket]
    word_action_count: int
    social_word_buckets: list[HabitSocialWordBucket]
    social_word_count: int
    music_belief_action_buckets: list[HabitMusicBeliefActionBucket]
    music_belief_action_count: int
    belief_buckets: list[HabitBeliefBucket]
    belief_count: int
    virtue_buckets: list[HabitVirtueBucket]
    virtue_count: int
    outer_thinking_buckets: list[HabitOuterThinkingBucket]
    outer_thinking_count: int
    inner_thinking_buckets: list[HabitInnerThinkingBucket]
    inner_thinking_count: int
    category_buckets: list[HabitCategoryBucket]
    category_count: int
    topic_buckets: list[HabitTopicBucket]
    topic_count: int
    sequences: list[HabitSequence]
    sequence_count: int
    speech_opening: str
    speech_segments: list[HabitSpeechSegment]
    speech_closing: str
    truncated: bool
    warnings: list[str]

class ContextHabitCategorization(
    BaseTool[
        ContextHabitCategorizationArgs,
        ContextHabitCategorizationResult,
        ContextHabitCategorizationConfig,
        ContextHabitCategorizationState,
    ],
    ToolUIData[
        ContextHabitCategorizationArgs,
        ContextHabitCategorizationResult,
    ],
):
    description: ClassVar[str] = (
        "Categorize habits and optionally organize them into sequences."
    )

    async def run(
        self, args: ContextHabitCategorizationArgs
    ) -> ContextHabitCategorizationResult:
        max_source_bytes = args.max_source_bytes or self.config.max_source_bytes
        max_total_bytes = args.max_total_bytes or self.config.max_total_bytes
        max_utterances = (
            args.max_utterances if args.max_utterances is not None else self.config.max_utterances
        )
        min_token_length = args.min_token_length or self.config.min_token_length
        max_keywords = args.max_keywords or self.config.max_keywords
        max_categories = (
            args.max_categories_per_habit
            if args.max_categories_per_habit is not None
            else self.config.max_categories_per_habit
        )
        max_category_terms = (
            args.max_category_terms
            if args.max_category_terms is not None
            else self.config.max_category_terms
        )
        max_origin_categories = (
            args.max_origin_categories_per_habit
            if args.max_origin_categories_per_habit is not None
            else self.config.max_origin_categories_per_habit
        )
        max_origin_terms = (
            args.max_origin_terms
            if args.max_origin_terms is not None
            else self.config.max_origin_terms
        )
        max_dendrite_categories = (
            args.max_dendrite_categories_per_habit
            if args.max_dendrite_categories_per_habit is not None
            else self.config.max_dendrite_categories_per_habit
        )
        max_dendrite_terms = (
            args.max_dendrite_terms
            if args.max_dendrite_terms is not None
            else self.config.max_dendrite_terms
        )
        max_timeframe_categories = (
            args.max_timeframe_categories_per_habit
            if args.max_timeframe_categories_per_habit is not None
            else self.config.max_timeframe_categories_per_habit
        )
        max_timeframe_terms = (
            args.max_timeframe_terms
            if args.max_timeframe_terms is not None
            else self.config.max_timeframe_terms
        )
        max_onset_categories = (
            args.max_onset_categories_per_habit
            if args.max_onset_categories_per_habit is not None
            else self.config.max_onset_categories_per_habit
        )
        max_onset_terms = (
            args.max_onset_terms
            if args.max_onset_terms is not None
            else self.config.max_onset_terms
        )
        max_frequency_categories = (
            args.max_frequency_categories_per_habit
            if args.max_frequency_categories_per_habit is not None
            else self.config.max_frequency_categories_per_habit
        )
        max_frequency_terms = (
            args.max_frequency_terms
            if args.max_frequency_terms is not None
            else self.config.max_frequency_terms
        )
        max_neural_firing_categories = (
            args.max_neural_firing_categories_per_habit
            if args.max_neural_firing_categories_per_habit is not None
            else self.config.max_neural_firing_categories_per_habit
        )
        max_neural_firing_terms = (
            args.max_neural_firing_terms
            if args.max_neural_firing_terms is not None
            else self.config.max_neural_firing_terms
        )
        max_muscle_memory_categories = (
            args.max_muscle_memory_categories_per_habit
            if args.max_muscle_memory_categories_per_habit is not None
            else self.config.max_muscle_memory_categories_per_habit
        )
        max_muscle_memory_terms = (
            args.max_muscle_memory_terms
            if args.max_muscle_memory_terms is not None
            else self.config.max_muscle_memory_terms
        )
        max_pattern_categories = (
            args.max_pattern_categories_per_habit
            if args.max_pattern_categories_per_habit is not None
            else self.config.max_pattern_categories_per_habit
        )
        max_pattern_terms = (
            args.max_pattern_terms
            if args.max_pattern_terms is not None
            else self.config.max_pattern_terms
        )
        max_structure_categories = (
            args.max_structure_categories_per_habit
            if args.max_structure_categories_per_habit is not None
            else self.config.max_structure_categories_per_habit
        )
        max_structure_terms = (
            args.max_structure_terms
            if args.max_structure_terms is not None
            else self.config.max_structure_terms
        )
        max_external_force_categories = (
            args.max_external_force_categories_per_habit
            if args.max_external_force_categories_per_habit is not None
            else self.config.max_external_force_categories_per_habit
        )
        max_external_force_terms = (
            args.max_external_force_terms
            if args.max_external_force_terms is not None
            else self.config.max_external_force_terms
        )
        max_influence_categories = (
            args.max_influence_categories_per_habit
            if args.max_influence_categories_per_habit is not None
            else self.config.max_influence_categories_per_habit
        )
        max_influence_terms = (
            args.max_influence_terms
            if args.max_influence_terms is not None
            else self.config.max_influence_terms
        )
        max_external_repetition_categories = (
            args.max_external_repetition_categories_per_habit
            if args.max_external_repetition_categories_per_habit is not None
            else self.config.max_external_repetition_categories_per_habit
        )
        max_external_repetition_terms = (
            args.max_external_repetition_terms
            if args.max_external_repetition_terms is not None
            else self.config.max_external_repetition_terms
        )
        max_event_sequence_categories = (
            args.max_event_sequence_categories_per_habit
            if args.max_event_sequence_categories_per_habit is not None
            else self.config.max_event_sequence_categories_per_habit
        )
        max_event_sequence_terms = (
            args.max_event_sequence_terms
            if args.max_event_sequence_terms is not None
            else self.config.max_event_sequence_terms
        )
        max_social_learning_categories = (
            args.max_social_learning_categories_per_habit
            if args.max_social_learning_categories_per_habit is not None
            else self.config.max_social_learning_categories_per_habit
        )
        max_social_learning_terms = (
            args.max_social_learning_terms
            if args.max_social_learning_terms is not None
            else self.config.max_social_learning_terms
        )
        max_provenance_categories = (
            args.max_provenance_categories_per_habit
            if args.max_provenance_categories_per_habit is not None
            else self.config.max_provenance_categories_per_habit
        )
        max_provenance_terms = (
            args.max_provenance_terms
            if args.max_provenance_terms is not None
            else self.config.max_provenance_terms
        )
        max_activation_categories = (
            args.max_activation_categories_per_habit
            if args.max_activation_categories_per_habit is not None
            else self.config.max_activation_categories_per_habit
        )
        max_activation_terms = (
            args.max_activation_terms
            if args.max_activation_terms is not None
            else self.config.max_activation_terms
        )
        max_dendro_motor_categories = (
            args.max_dendro_motor_categories_per_habit
            if args.max_dendro_motor_categories_per_habit is not None
            else self.config.max_dendro_motor_categories_per_habit
        )
        max_dendro_motor_terms = (
            args.max_dendro_motor_terms
            if args.max_dendro_motor_terms is not None
            else self.config.max_dendro_motor_terms
        )
        max_music_categories = (
            args.max_music_categories_per_habit
            if args.max_music_categories_per_habit is not None
            else self.config.max_music_categories_per_habit
        )
        max_music_terms = (
            args.max_music_terms
            if args.max_music_terms is not None
            else self.config.max_music_terms
        )
        max_intent_categories = (
            args.max_intent_categories_per_habit
            if args.max_intent_categories_per_habit is not None
            else self.config.max_intent_categories_per_habit
        )
        max_intent_terms = (
            args.max_intent_terms
            if args.max_intent_terms is not None
            else self.config.max_intent_terms
        )
        max_action_categories = (
            args.max_action_categories_per_habit
            if args.max_action_categories_per_habit is not None
            else self.config.max_action_categories_per_habit
        )
        max_action_terms = (
            args.max_action_terms
            if args.max_action_terms is not None
            else self.config.max_action_terms
        )
        max_word_action_categories = (
            args.max_word_action_categories_per_habit
            if args.max_word_action_categories_per_habit is not None
            else self.config.max_word_action_categories_per_habit
        )
        max_word_action_terms = (
            args.max_word_action_terms
            if args.max_word_action_terms is not None
            else self.config.max_word_action_terms
        )
        max_social_word_categories = (
            args.max_social_word_categories_per_habit
            if args.max_social_word_categories_per_habit is not None
            else self.config.max_social_word_categories_per_habit
        )
        max_social_word_terms = (
            args.max_social_word_terms
            if args.max_social_word_terms is not None
            else self.config.max_social_word_terms
        )
        max_music_belief_action_categories = (
            args.max_music_belief_action_categories_per_habit
            if args.max_music_belief_action_categories_per_habit is not None
            else self.config.max_music_belief_action_categories_per_habit
        )
        max_music_belief_action_terms = (
            args.max_music_belief_action_terms
            if args.max_music_belief_action_terms is not None
            else self.config.max_music_belief_action_terms
        )
        max_belief_categories = (
            args.max_belief_categories_per_habit
            if args.max_belief_categories_per_habit is not None
            else self.config.max_belief_categories_per_habit
        )
        max_belief_terms = (
            args.max_belief_terms
            if args.max_belief_terms is not None
            else self.config.max_belief_terms
        )
        max_virtue_categories = (
            args.max_virtue_categories_per_habit
            if args.max_virtue_categories_per_habit is not None
            else self.config.max_virtue_categories_per_habit
        )
        max_virtue_terms = (
            args.max_virtue_terms
            if args.max_virtue_terms is not None
            else self.config.max_virtue_terms
        )
        max_outer_thinking_categories = (
            args.max_outer_thinking_categories_per_habit
            if args.max_outer_thinking_categories_per_habit is not None
            else self.config.max_outer_thinking_categories_per_habit
        )
        max_outer_thinking_terms = (
            args.max_outer_thinking_terms
            if args.max_outer_thinking_terms is not None
            else self.config.max_outer_thinking_terms
        )
        max_inner_thinking_categories = (
            args.max_inner_thinking_categories_per_habit
            if args.max_inner_thinking_categories_per_habit is not None
            else self.config.max_inner_thinking_categories_per_habit
        )
        max_inner_thinking_terms = (
            args.max_inner_thinking_terms
            if args.max_inner_thinking_terms is not None
            else self.config.max_inner_thinking_terms
        )
        max_topics = args.max_topics if args.max_topics is not None else self.config.max_topics
        max_sequences = args.max_sequences if args.max_sequences is not None else self.config.max_sequences
        max_steps_per_sequence = (
            args.max_steps_per_sequence
            if args.max_steps_per_sequence is not None
            else self.config.max_steps_per_sequence
        )
        max_category_segments = (
            args.max_category_segments
            if args.max_category_segments is not None
            else self.config.max_category_segments
        )
        max_sequence_segments = (
            args.max_sequence_segments
            if args.max_sequence_segments is not None
            else self.config.max_sequence_segments
        )
        max_origin_segments = (
            args.max_origin_segments
            if args.max_origin_segments is not None
            else self.config.max_origin_segments
        )
        max_dendrite_segments = (
            args.max_dendrite_segments
            if args.max_dendrite_segments is not None
            else self.config.max_dendrite_segments
        )
        max_timeframe_segments = (
            args.max_timeframe_segments
            if args.max_timeframe_segments is not None
            else self.config.max_timeframe_segments
        )
        max_onset_segments = (
            args.max_onset_segments
            if args.max_onset_segments is not None
            else self.config.max_onset_segments
        )
        max_frequency_segments = (
            args.max_frequency_segments
            if args.max_frequency_segments is not None
            else self.config.max_frequency_segments
        )
        max_neural_firing_segments = (
            args.max_neural_firing_segments
            if args.max_neural_firing_segments is not None
            else self.config.max_neural_firing_segments
        )
        max_muscle_memory_segments = (
            args.max_muscle_memory_segments
            if args.max_muscle_memory_segments is not None
            else self.config.max_muscle_memory_segments
        )
        max_pattern_segments = (
            args.max_pattern_segments
            if args.max_pattern_segments is not None
            else self.config.max_pattern_segments
        )
        max_structure_segments = (
            args.max_structure_segments
            if args.max_structure_segments is not None
            else self.config.max_structure_segments
        )
        max_external_force_segments = (
            args.max_external_force_segments
            if args.max_external_force_segments is not None
            else self.config.max_external_force_segments
        )
        max_influence_segments = (
            args.max_influence_segments
            if args.max_influence_segments is not None
            else self.config.max_influence_segments
        )
        max_external_repetition_segments = (
            args.max_external_repetition_segments
            if args.max_external_repetition_segments is not None
            else self.config.max_external_repetition_segments
        )
        max_event_sequence_segments = (
            args.max_event_sequence_segments
            if args.max_event_sequence_segments is not None
            else self.config.max_event_sequence_segments
        )
        max_social_learning_segments = (
            args.max_social_learning_segments
            if args.max_social_learning_segments is not None
            else self.config.max_social_learning_segments
        )
        max_provenance_segments = (
            args.max_provenance_segments
            if args.max_provenance_segments is not None
            else self.config.max_provenance_segments
        )
        max_activation_segments = (
            args.max_activation_segments
            if args.max_activation_segments is not None
            else self.config.max_activation_segments
        )
        max_dendro_motor_segments = (
            args.max_dendro_motor_segments
            if args.max_dendro_motor_segments is not None
            else self.config.max_dendro_motor_segments
        )
        max_music_segments = (
            args.max_music_segments
            if args.max_music_segments is not None
            else self.config.max_music_segments
        )
        max_intent_segments = (
            args.max_intent_segments
            if args.max_intent_segments is not None
            else self.config.max_intent_segments
        )
        max_action_segments = (
            args.max_action_segments
            if args.max_action_segments is not None
            else self.config.max_action_segments
        )
        max_word_action_segments = (
            args.max_word_action_segments
            if args.max_word_action_segments is not None
            else self.config.max_word_action_segments
        )
        max_social_word_segments = (
            args.max_social_word_segments
            if args.max_social_word_segments is not None
            else self.config.max_social_word_segments
        )
        max_music_belief_action_segments = (
            args.max_music_belief_action_segments
            if args.max_music_belief_action_segments is not None
            else self.config.max_music_belief_action_segments
        )
        max_belief_segments = (
            args.max_belief_segments
            if args.max_belief_segments is not None
            else self.config.max_belief_segments
        )
        max_virtue_segments = (
            args.max_virtue_segments
            if args.max_virtue_segments is not None
            else self.config.max_virtue_segments
        )
        max_outer_thinking_segments = (
            args.max_outer_thinking_segments
            if args.max_outer_thinking_segments is not None
            else self.config.max_outer_thinking_segments
        )
        max_inner_thinking_segments = (
            args.max_inner_thinking_segments
            if args.max_inner_thinking_segments is not None
            else self.config.max_inner_thinking_segments
        )
        max_repetition_segments = (
            args.max_repetition_segments
            if args.max_repetition_segments is not None
            else self.config.max_repetition_segments
        )
        max_habit_segments = (
            args.max_habit_segments
            if args.max_habit_segments is not None
            else self.config.max_habit_segments
        )
        max_speech_segments = (
            args.max_speech_segments
            if args.max_speech_segments is not None
            else self.config.max_speech_segments
        )
        include_singletons = (
            args.include_singleton_sequences
            if args.include_singleton_sequences is not None
            else self.config.include_singleton_sequences
        )

        split_mode = (args.split_mode or self.config.default_split_mode).strip().lower()
        if split_mode not in {"lines", "sentences"}:
            raise ToolError("split_mode must be lines or sentences.")

        warnings: list[str] = []
        truncated = False

        utterances, input_count, bytes_truncated = self._load_utterances(
            args, max_source_bytes, max_total_bytes, split_mode
        )
        if bytes_truncated:
            warnings.append("Utterances truncated by byte budget.")
            truncated = True

        if max_utterances is not None and max_utterances > 0 and len(utterances) > max_utterances:
            warnings.append("Utterance limit reached; truncating list.")
            utterances = utterances[:max_utterances]
            truncated = True

        if not utterances:
            raise ToolError("No habit statements provided.")

        habit_entries: list[HabitEntry] = []
        habit_tokens: list[list[str]] = []
        habit_categories: list[list[str]] = []
        habit_origins: list[list[str]] = []
        habit_dendrites: list[list[str]] = []
        habit_timeframes: list[list[str]] = []
        habit_onsets: list[list[str]] = []
        habit_frequencies: list[list[str]] = []
        habit_neural_firings: list[list[str]] = []
        habit_muscle_memories: list[list[str]] = []
        habit_patterns: list[list[str]] = []
        habit_structures: list[list[str]] = []
        habit_external_forces: list[list[str]] = []
        habit_influences: list[list[str]] = []
        habit_external_repetitions: list[list[str]] = []
        habit_event_sequences: list[list[str]] = []
        habit_social_learnings: list[list[str]] = []
        habit_provenances: list[list[str]] = []
        habit_activations: list[list[str]] = []
        habit_dendro_motors: list[list[str]] = []
        habit_music: list[list[str]] = []
        habit_intents: list[list[str]] = []
        habit_actions: list[list[str]] = []
        habit_word_actions: list[list[str]] = []
        habit_social_words: list[list[str]] = []
        habit_music_belief_actions: list[list[str]] = []
        habit_beliefs: list[list[str]] = []
        habit_virtues: list[list[str]] = []
        habit_outer_thinking: list[list[str]] = []
        habit_inner_thinking: list[list[str]] = []
        habit_like_count = 0
        repetition_like_count = 0
        dendrite_like_count = 0
        timeframe_like_count = 0
        onset_like_count = 0
        frequency_like_count = 0
        neural_firing_like_count = 0
        muscle_memory_like_count = 0
        pattern_like_count = 0
        structure_like_count = 0
        external_force_like_count = 0
        influence_like_count = 0
        external_repetition_like_count = 0
        event_sequence_like_count = 0
        social_learning_like_count = 0
        provenance_like_count = 0
        activation_like_count = 0
        dendro_motor_like_count = 0
        music_like_count = 0
        intent_like_count = 0
        action_like_count = 0
        word_action_like_count = 0
        social_word_like_count = 0
        music_belief_action_like_count = 0
        belief_like_count = 0
        virtue_like_count = 0
        outer_thinking_like_count = 0
        inner_thinking_like_count = 0

        for idx, utt in enumerate(utterances, start=1):
            tokens = self._tokenize(utt.text, min_token_length)
            habit_markers = self._match_markers(utt.text, tokens, HABIT_MARKERS)
            habit_like = bool(habit_markers)
            if habit_like:
                habit_like_count += 1

            categories = self._assign_categories(utt.text, tokens, max_categories)
            origin_categories, origin_markers = self._assign_origin_categories(
                utt.text, tokens, max_origin_categories
            )
            dendrite_categories, dendrite_markers = self._assign_dendrite_categories(
                utt.text, tokens, max_dendrite_categories
            )
            timeframe_categories, timeframe_markers = self._assign_timeframe_categories(
                utt.text, tokens, max_timeframe_categories
            )
            onset_categories, onset_markers = self._assign_onset_categories(
                utt.text, tokens, timeframe_categories, timeframe_markers, max_onset_categories
            )
            frequency_categories, frequency_markers = self._assign_frequency_categories(
                utt.text, tokens, max_frequency_categories
            )
            neural_firing_categories, neural_firing_markers = self._assign_neural_firing_categories(
                utt.text, tokens, max_neural_firing_categories
            )
            sequence_markers = self._match_markers(utt.text, tokens, SEQUENCE_MARKERS)
            repetition_markers = self._match_markers(utt.text, tokens, REPETITION_MARKERS)
            muscle_memory_categories, muscle_memory_markers = self._assign_muscle_memory_categories(
                utt.text, tokens, max_muscle_memory_categories
            )
            pattern_categories, pattern_markers = self._assign_pattern_categories(
                utt.text,
                tokens,
                sequence_markers,
                repetition_markers,
                max_pattern_categories,
            )
            structure_categories, structure_markers = self._assign_structure_categories(
                utt.text,
                tokens,
                sequence_markers,
                repetition_markers,
                max_structure_categories,
            )
            external_force_categories, external_force_markers = self._assign_external_force_categories(
                utt.text,
                tokens,
                max_external_force_categories,
            )
            influence_categories, influence_markers = self._assign_influence_categories(
                utt.text,
                tokens,
                max_influence_categories,
            )
            external_repetition_categories, external_repetition_markers = (
                self._assign_external_repetition_categories(
                    utt.text,
                    tokens,
                    external_force_markers,
                    repetition_markers,
                    max_external_repetition_categories,
                )
            )
            event_sequence_categories, event_sequence_markers = self._assign_event_sequence_categories(
                utt.text,
                tokens,
                sequence_markers,
                max_event_sequence_categories,
            )
            social_learning_categories, social_learning_markers = self._assign_social_learning_categories(
                utt.text,
                tokens,
                repetition_markers,
                max_social_learning_categories,
            )
            provenance_categories, provenance_markers = self._assign_provenance_categories(
                utt.text,
                tokens,
                max_provenance_categories,
            )
            activation_categories, activation_markers = self._assign_activation_categories(
                utt.text,
                tokens,
                max_activation_categories,
            )
            dendro_motor_categories, dendro_motor_markers = self._assign_dendro_motor_categories(
                utt.text,
                tokens,
                max_dendro_motor_categories,
            )
            music_categories, music_markers = self._assign_music_categories(
                utt.text,
                tokens,
                repetition_markers,
                timeframe_markers,
                max_music_categories,
            )
            intent_categories, intent_markers = self._assign_intent_categories(
                utt.text,
                tokens,
                max_intent_categories,
            )
            action_categories, action_markers = self._assign_action_categories(
                utt.text,
                tokens,
                max_action_categories,
            )
            word_action_categories, word_action_markers = self._assign_word_action_categories(
                utt.text,
                tokens,
                action_markers,
                max_word_action_categories,
            )
            social_word_categories, social_word_markers = self._assign_social_word_categories(
                utt.text,
                tokens,
                max_social_word_categories,
            )
            belief_categories, belief_markers = self._assign_belief_categories(
                utt.text,
                tokens,
                max_belief_categories,
            )
            (
                music_belief_action_categories,
                music_belief_action_markers,
            ) = self._assign_music_belief_action_categories(
                utt.text,
                tokens,
                music_markers,
                belief_markers,
                action_markers,
                max_music_belief_action_categories,
            )
            virtue_categories, virtue_markers = self._assign_virtue_categories(
                utt.text,
                tokens,
                max_virtue_categories,
            )
            outer_thinking_categories, outer_thinking_markers = self._assign_outer_thinking_categories(
                utt.text,
                tokens,
                max_outer_thinking_categories,
            )
            inner_thinking_categories, inner_thinking_markers = self._assign_inner_thinking_categories(
                utt.text,
                tokens,
                max_inner_thinking_categories,
            )
            keywords = self._keywords(tokens, max_keywords)
            repetition_like = bool(repetition_markers)
            if repetition_like:
                repetition_like_count += 1
            dendrite_like = bool(dendrite_markers)
            if dendrite_like:
                dendrite_like_count += 1
            timeframe_like = bool(timeframe_markers)
            if timeframe_like:
                timeframe_like_count += 1
            onset_like = bool(onset_markers)
            if onset_like:
                onset_like_count += 1
            frequency_like = bool(frequency_markers)
            if frequency_like:
                frequency_like_count += 1
            neural_firing_like = bool(neural_firing_markers)
            if neural_firing_like:
                neural_firing_like_count += 1
            muscle_memory_like = bool(muscle_memory_markers)
            if muscle_memory_like:
                muscle_memory_like_count += 1
            pattern_like = bool(pattern_markers)
            if pattern_like:
                pattern_like_count += 1
            structure_like = bool(structure_markers)
            if structure_like:
                structure_like_count += 1
            external_force_like = bool(external_force_markers)
            if external_force_like:
                external_force_like_count += 1
            influence_like = bool(influence_markers)
            if influence_like:
                influence_like_count += 1
            external_repetition_like = bool(external_repetition_markers)
            if external_repetition_like:
                external_repetition_like_count += 1
            event_sequence_like = bool(event_sequence_markers)
            if event_sequence_like:
                event_sequence_like_count += 1
            social_learning_like = bool(social_learning_markers)
            if social_learning_like:
                social_learning_like_count += 1
            provenance_like = bool(provenance_markers)
            if provenance_like:
                provenance_like_count += 1
            activation_like = bool(activation_markers)
            if activation_like:
                activation_like_count += 1
            dendro_motor_like = bool(dendro_motor_markers)
            if dendro_motor_like:
                dendro_motor_like_count += 1
            music_like = bool(music_markers)
            if music_like:
                music_like_count += 1
            intent_like = bool(intent_markers)
            if intent_like:
                intent_like_count += 1
            action_like = bool(action_markers)
            if action_like:
                action_like_count += 1
            word_action_like = bool(word_action_markers)
            if word_action_like:
                word_action_like_count += 1
            social_word_like = bool(social_word_markers)
            if social_word_like:
                social_word_like_count += 1
            music_belief_action_like = bool(music_belief_action_markers)
            if music_belief_action_like:
                music_belief_action_like_count += 1
            belief_like = bool(belief_markers)
            if belief_like:
                belief_like_count += 1
            virtue_like = bool(virtue_markers)
            if virtue_like:
                virtue_like_count += 1
            outer_thinking_like = bool(outer_thinking_markers)
            if outer_thinking_like:
                outer_thinking_like_count += 1
            inner_thinking_like = bool(inner_thinking_markers)
            if inner_thinking_like:
                inner_thinking_like_count += 1

            habit_entries.append(
                HabitEntry(
                    index=idx,
                    speaker=utt.speaker,
                    text=utt.text,
                    preview=self._preview(utt.text),
                    habit_like=habit_like,
                    habit_markers=habit_markers,
                    categories=categories,
                    origin_categories=origin_categories,
                    origin_markers=origin_markers,
                    dendrite_categories=dendrite_categories,
                    dendrite_markers=dendrite_markers,
                    timeframe_categories=timeframe_categories,
                    timeframe_markers=timeframe_markers,
                    onset_categories=onset_categories,
                    onset_markers=onset_markers,
                    frequency_categories=frequency_categories,
                    frequency_markers=frequency_markers,
                    neural_firing_categories=neural_firing_categories,
                    neural_firing_markers=neural_firing_markers,
                    muscle_memory_categories=muscle_memory_categories,
                    muscle_memory_markers=muscle_memory_markers,
                    pattern_categories=pattern_categories,
                    pattern_markers=pattern_markers,
                    structure_categories=structure_categories,
                    structure_markers=structure_markers,
                    external_force_categories=external_force_categories,
                    external_force_markers=external_force_markers,
                    influence_categories=influence_categories,
                    influence_markers=influence_markers,
                    external_repetition_categories=external_repetition_categories,
                    external_repetition_markers=external_repetition_markers,
                    event_sequence_categories=event_sequence_categories,
                    event_sequence_markers=event_sequence_markers,
                    social_learning_categories=social_learning_categories,
                    social_learning_markers=social_learning_markers,
                    provenance_categories=provenance_categories,
                    provenance_markers=provenance_markers,
                    activation_categories=activation_categories,
                    activation_markers=activation_markers,
                    dendro_motor_categories=dendro_motor_categories,
                    dendro_motor_markers=dendro_motor_markers,
                    music_categories=music_categories,
                    music_markers=music_markers,
                    intent_categories=intent_categories,
                    intent_markers=intent_markers,
                    action_categories=action_categories,
                    action_markers=action_markers,
                    word_action_categories=word_action_categories,
                    word_action_markers=word_action_markers,
                    social_word_categories=social_word_categories,
                    social_word_markers=social_word_markers,
                    music_belief_action_categories=music_belief_action_categories,
                    music_belief_action_markers=music_belief_action_markers,
                    belief_categories=belief_categories,
                    belief_markers=belief_markers,
                    virtue_categories=virtue_categories,
                    virtue_markers=virtue_markers,
                    outer_thinking_categories=outer_thinking_categories,
                    outer_thinking_markers=outer_thinking_markers,
                    inner_thinking_categories=inner_thinking_categories,
                    inner_thinking_markers=inner_thinking_markers,
                    keywords=keywords,
                    sequence_markers=sequence_markers,
                    repetition_like=repetition_like,
                    repetition_markers=repetition_markers,
                )
            )
            habit_tokens.append(tokens)
            habit_categories.append(categories)
            habit_origins.append(origin_categories)
            habit_dendrites.append(dendrite_categories)
            habit_timeframes.append(timeframe_categories)
            habit_onsets.append(onset_categories)
            habit_frequencies.append(frequency_categories)
            habit_neural_firings.append(neural_firing_categories)
            habit_muscle_memories.append(muscle_memory_categories)
            habit_patterns.append(pattern_categories)
            habit_structures.append(structure_categories)
            habit_external_forces.append(external_force_categories)
            habit_influences.append(influence_categories)
            habit_external_repetitions.append(external_repetition_categories)
            habit_event_sequences.append(event_sequence_categories)
            habit_social_learnings.append(social_learning_categories)
            habit_provenances.append(provenance_categories)
            habit_activations.append(activation_categories)
            habit_dendro_motors.append(dendro_motor_categories)
            habit_music.append(music_categories)
            habit_intents.append(intent_categories)
            habit_actions.append(action_categories)
            habit_word_actions.append(word_action_categories)
            habit_social_words.append(social_word_categories)
            habit_music_belief_actions.append(music_belief_action_categories)
            habit_beliefs.append(belief_categories)
            habit_virtues.append(virtue_categories)
            habit_outer_thinking.append(outer_thinking_categories)
            habit_inner_thinking.append(inner_thinking_categories)

        category_buckets = self._build_category_buckets(
            habit_categories,
            habit_tokens,
            max_category_terms,
        )
        topic_buckets = self._build_topic_buckets(
            habit_tokens,
            max_topics,
        )
        origin_buckets = self._build_origin_buckets(
            habit_origins,
            habit_tokens,
            max_origin_terms,
        )
        dendrite_buckets = self._build_dendrite_buckets(
            habit_dendrites,
            habit_tokens,
            max_dendrite_terms,
        )
        timeframe_buckets = self._build_timeframe_buckets(
            habit_timeframes,
            habit_tokens,
            max_timeframe_terms,
        )
        onset_buckets = self._build_onset_buckets(
            habit_onsets,
            habit_tokens,
            max_onset_terms,
        )
        frequency_buckets = self._build_frequency_buckets(
            habit_frequencies,
            habit_tokens,
            max_frequency_terms,
        )
        neural_firing_buckets = self._build_neural_firing_buckets(
            habit_neural_firings,
            habit_tokens,
            max_neural_firing_terms,
        )
        muscle_memory_buckets = self._build_muscle_memory_buckets(
            habit_muscle_memories,
            habit_tokens,
            max_muscle_memory_terms,
        )
        pattern_buckets = self._build_pattern_buckets(
            habit_patterns,
            habit_tokens,
            max_pattern_terms,
        )
        structure_buckets = self._build_structure_buckets(
            habit_structures,
            habit_tokens,
            max_structure_terms,
        )
        external_force_buckets = self._build_external_force_buckets(
            habit_external_forces,
            habit_tokens,
            max_external_force_terms,
        )
        influence_buckets = self._build_influence_buckets(
            habit_influences,
            habit_tokens,
            max_influence_terms,
        )
        external_repetition_buckets = self._build_external_repetition_buckets(
            habit_external_repetitions,
            habit_tokens,
            max_external_repetition_terms,
        )
        event_sequence_buckets = self._build_event_sequence_buckets(
            habit_event_sequences,
            habit_tokens,
            max_event_sequence_terms,
        )
        social_learning_buckets = self._build_social_learning_buckets(
            habit_social_learnings,
            habit_tokens,
            max_social_learning_terms,
        )
        provenance_buckets = self._build_provenance_buckets(
            habit_provenances,
            habit_tokens,
            max_provenance_terms,
        )
        activation_buckets = self._build_activation_buckets(
            habit_activations,
            habit_tokens,
            max_activation_terms,
        )
        dendro_motor_buckets = self._build_dendro_motor_buckets(
            habit_dendro_motors,
            habit_tokens,
            max_dendro_motor_terms,
        )
        music_buckets = self._build_music_buckets(
            habit_music,
            habit_tokens,
            max_music_terms,
        )
        intent_buckets = self._build_intent_buckets(
            habit_intents,
            habit_tokens,
            max_intent_terms,
        )
        action_buckets = self._build_action_buckets(
            habit_actions,
            habit_tokens,
            max_action_terms,
        )
        word_action_buckets = self._build_word_action_buckets(
            habit_word_actions,
            habit_tokens,
            max_word_action_terms,
        )
        social_word_buckets = self._build_social_word_buckets(
            habit_social_words,
            habit_tokens,
            max_social_word_terms,
        )
        music_belief_action_buckets = self._build_music_belief_action_buckets(
            habit_music_belief_actions,
            habit_tokens,
            max_music_belief_action_terms,
        )
        belief_buckets = self._build_belief_buckets(
            habit_beliefs,
            habit_tokens,
            max_belief_terms,
        )
        virtue_buckets = self._build_virtue_buckets(
            habit_virtues,
            habit_tokens,
            max_virtue_terms,
        )
        outer_thinking_buckets = self._build_outer_thinking_buckets(
            habit_outer_thinking,
            habit_tokens,
            max_outer_thinking_terms,
        )
        inner_thinking_buckets = self._build_inner_thinking_buckets(
            habit_inner_thinking,
            habit_tokens,
            max_inner_thinking_terms,
        )

        sequences: list[HabitSequence] = []
        if args.include_sequences:
            sequences = self._build_sequences(
                habit_entries,
                include_singletons,
                max_sequences,
                max_steps_per_sequence,
            )

        speech_opening = self._speech_opening(
            args,
            len(habit_entries),
            len(sequences),
            repetition_like_count,
            len(origin_buckets),
            len(dendrite_buckets),
            len(timeframe_buckets),
            len(onset_buckets),
            len(frequency_buckets),
            len(neural_firing_buckets),
            len(muscle_memory_buckets),
            len(pattern_buckets),
            len(structure_buckets),
            len(external_force_buckets),
            len(influence_buckets),
            len(external_repetition_buckets),
            len(event_sequence_buckets),
            len(social_learning_buckets),
            len(provenance_buckets),
            len(activation_buckets),
            len(dendro_motor_buckets),
            len(music_buckets),
            len(intent_buckets),
            len(action_buckets),
            len(word_action_buckets),
            len(social_word_buckets),
            len(music_belief_action_buckets),
            len(belief_buckets),
            len(virtue_buckets),
            len(outer_thinking_buckets),
            len(inner_thinking_buckets),
        )
        speech_segments, segments_truncated = self._speech_segments(
            category_buckets,
            origin_buckets,
            dendrite_buckets,
            timeframe_buckets,
            onset_buckets,
            frequency_buckets,
            neural_firing_buckets,
            muscle_memory_buckets,
            pattern_buckets,
            structure_buckets,
            external_force_buckets,
            influence_buckets,
            external_repetition_buckets,
            event_sequence_buckets,
            social_learning_buckets,
            provenance_buckets,
            activation_buckets,
            dendro_motor_buckets,
            music_buckets,
            intent_buckets,
            action_buckets,
            word_action_buckets,
            social_word_buckets,
            music_belief_action_buckets,
            belief_buckets,
            virtue_buckets,
            outer_thinking_buckets,
            inner_thinking_buckets,
            sequences,
            habit_entries,
            max_category_segments,
            max_sequence_segments,
            max_origin_segments,
            max_dendrite_segments,
            max_timeframe_segments,
            max_onset_segments,
            max_frequency_segments,
            max_neural_firing_segments,
            max_muscle_memory_segments,
            max_pattern_segments,
            max_structure_segments,
            max_external_force_segments,
            max_influence_segments,
            max_external_repetition_segments,
            max_event_sequence_segments,
            max_social_learning_segments,
            max_provenance_segments,
            max_activation_segments,
            max_dendro_motor_segments,
            max_music_segments,
            max_intent_segments,
            max_action_segments,
            max_word_action_segments,
            max_social_word_segments,
            max_music_belief_action_segments,
            max_belief_segments,
            max_virtue_segments,
            max_outer_thinking_segments,
            max_inner_thinking_segments,
            max_repetition_segments,
            max_habit_segments,
            max_speech_segments,
        )
        if segments_truncated:
            warnings.append("Speech segments truncated by limits.")
            truncated = True
        speech_closing = self._speech_closing(args)

        return ContextHabitCategorizationResult(
            habit_categorization_term="habit categorization",
            habit_categorization_description=(
                "The ability to group habits by shared themes, contexts, or behaviors."
            ),
            habit_sequence_term="habit sequence categorization",
            habit_sequence_description=(
                "The ability to organize habits into ordered sequences and routines."
            ),
            habit_repetition_term="habit repetition categorization",
            habit_repetition_description=(
                "The ability to group habits by repeated occurrence and recurrence signals."
            ),
            habit_origin_term="habit formation origin categorization",
            habit_origin_description=(
                "The ability to categorize habits by the context or place where they formed."
            ),
            habit_dendrite_term="habit dendrite categorization",
            habit_dendrite_description=(
                "The ability to categorize habits by dendritic or neural structure signals."
            ),
            habit_timeframe_term="habit timeframe categorization",
            habit_timeframe_description=(
                "The ability to categorize habits by when they occurred."
            ),
            habit_onset_term="habit onset timeframe categorization",
            habit_onset_description=(
                "The ability to categorize habits by when they started developing."
            ),
            habit_frequency_term="habit frequency categorization",
            habit_frequency_description=(
                "The ability to categorize habits by how often they occur."
            ),
            habit_neural_firing_term="habit neural firing categorization",
            habit_neural_firing_description=(
                "The ability to categorize habits by neural firing signals."
            ),
            habit_muscle_memory_term="habit muscle memory change categorization",
            habit_muscle_memory_description=(
                "The ability to categorize habits by changes in muscle memory reactions."
            ),
            habit_pattern_term="habit pattern categorization",
            habit_pattern_description=(
                "The ability to categorize habits by the patterns within the habit itself."
            ),
            habit_structure_term="habit structure categorization",
            habit_structure_description=(
                "The ability to categorize habits by the structure of the habit."
            ),
            habit_external_force_term="habit external force categorization",
            habit_external_force_description=(
                "The ability to categorize habits that come from external forces."
            ),
            habit_influence_term="habit influence categorization",
            habit_influence_description=(
                "The ability to categorize habits by the influence that causes them to happen."
            ),
            habit_external_repetition_term="habit external repetition categorization",
            habit_external_repetition_description=(
                "The ability to categorize habits by external factors that occur through repetition."
            ),
            habit_event_sequence_term="habit event sequence categorization",
            habit_event_sequence_description=(
                "The ability to categorize habits through sequences of events."
            ),
            habit_social_learning_term="habit social learning repetition categorization",
            habit_social_learning_description=(
                "The ability to categorize habits influenced by others and reinforced through repetition."
            ),
            habit_provenance_term="habit provenance categorization",
            habit_provenance_description=(
                "The ability to categorize habits by who they stem from and how they developed."
            ),
            habit_activation_term="habit activation categorization",
            habit_activation_description=(
                "The ability to categorize habits by their activation cues."
            ),
            habit_dendro_motor_term="habit dendro-motor categorization",
            habit_dendro_motor_description=(
                "The ability to categorize habits by dendrite-to-muscle memory response pathways."
            ),
            habit_music_term="habit music causation categorization",
            habit_music_description=(
                "The ability to categorize habits formed by repeated listening to music."
            ),
            habit_intent_term="habit intent categorization",
            habit_intent_description=(
                "The ability to categorize habits by intent and purpose."
            ),
            habit_action_term="habit action categorization",
            habit_action_description=(
                "The ability to categorize habits by actions someone makes."
            ),
            habit_word_action_term="habit word-action categorization",
            habit_word_action_description=(
                "The ability to categorize habits where actions form from words."
            ),
            habit_social_word_term="habit social word influence categorization",
            habit_social_word_description=(
                "The ability to categorize habits influenced by another person's words."
            ),
            habit_music_belief_action_term="habit music belief-action categorization",
            habit_music_belief_action_description=(
                "The ability to categorize habits shaped by music words that resonate "
                "with beliefs and prompt action."
            ),
            habit_belief_term="habit belief categorization",
            habit_belief_description=(
                "The ability to categorize habits by beliefs, including inner and outer beliefs."
            ),
            habit_virtue_term="habit virtue categorization",
            habit_virtue_description=(
                "The ability to categorize habits by virtues that support their development."
            ),
            habit_outer_thinking_term="habit outer thinking categorization",
            habit_outer_thinking_description=(
                "The ability to categorize habits by outer thinking patterns."
            ),
            habit_inner_thinking_term="habit inner thinking categorization",
            habit_inner_thinking_description=(
                "The ability to categorize habits by inner thought patterns."
            ),
            habits=habit_entries,
            habit_count=len(habit_entries),
            habit_like_count=habit_like_count,
            repetition_like_count=repetition_like_count,
            dendrite_like_count=dendrite_like_count,
            timeframe_like_count=timeframe_like_count,
            onset_like_count=onset_like_count,
            frequency_like_count=frequency_like_count,
            neural_firing_like_count=neural_firing_like_count,
            muscle_memory_like_count=muscle_memory_like_count,
            pattern_like_count=pattern_like_count,
            structure_like_count=structure_like_count,
            external_force_like_count=external_force_like_count,
            influence_like_count=influence_like_count,
            external_repetition_like_count=external_repetition_like_count,
            event_sequence_like_count=event_sequence_like_count,
            social_learning_like_count=social_learning_like_count,
            provenance_like_count=provenance_like_count,
            activation_like_count=activation_like_count,
            dendro_motor_like_count=dendro_motor_like_count,
            music_like_count=music_like_count,
            intent_like_count=intent_like_count,
            action_like_count=action_like_count,
            word_action_like_count=word_action_like_count,
            social_word_like_count=social_word_like_count,
            music_belief_action_like_count=music_belief_action_like_count,
            belief_like_count=belief_like_count,
            virtue_like_count=virtue_like_count,
            outer_thinking_like_count=outer_thinking_like_count,
            inner_thinking_like_count=inner_thinking_like_count,
            origin_buckets=origin_buckets,
            origin_count=len(origin_buckets),
            dendrite_buckets=dendrite_buckets,
            dendrite_count=len(dendrite_buckets),
            timeframe_buckets=timeframe_buckets,
            timeframe_count=len(timeframe_buckets),
            onset_buckets=onset_buckets,
            onset_count=len(onset_buckets),
            frequency_buckets=frequency_buckets,
            frequency_count=len(frequency_buckets),
            neural_firing_buckets=neural_firing_buckets,
            neural_firing_count=len(neural_firing_buckets),
            muscle_memory_buckets=muscle_memory_buckets,
            muscle_memory_count=len(muscle_memory_buckets),
            pattern_buckets=pattern_buckets,
            pattern_count=len(pattern_buckets),
            structure_buckets=structure_buckets,
            structure_count=len(structure_buckets),
            external_force_buckets=external_force_buckets,
            external_force_count=len(external_force_buckets),
            influence_buckets=influence_buckets,
            influence_count=len(influence_buckets),
            external_repetition_buckets=external_repetition_buckets,
            external_repetition_count=len(external_repetition_buckets),
            event_sequence_buckets=event_sequence_buckets,
            event_sequence_count=len(event_sequence_buckets),
            social_learning_buckets=social_learning_buckets,
            social_learning_count=len(social_learning_buckets),
            provenance_buckets=provenance_buckets,
            provenance_count=len(provenance_buckets),
            activation_buckets=activation_buckets,
            activation_count=len(activation_buckets),
            dendro_motor_buckets=dendro_motor_buckets,
            dendro_motor_count=len(dendro_motor_buckets),
            music_buckets=music_buckets,
            music_count=len(music_buckets),
            intent_buckets=intent_buckets,
            intent_count=len(intent_buckets),
            action_buckets=action_buckets,
            action_count=len(action_buckets),
            word_action_buckets=word_action_buckets,
            word_action_count=len(word_action_buckets),
            social_word_buckets=social_word_buckets,
            social_word_count=len(social_word_buckets),
            music_belief_action_buckets=music_belief_action_buckets,
            music_belief_action_count=len(music_belief_action_buckets),
            belief_buckets=belief_buckets,
            belief_count=len(belief_buckets),
            virtue_buckets=virtue_buckets,
            virtue_count=len(virtue_buckets),
            outer_thinking_buckets=outer_thinking_buckets,
            outer_thinking_count=len(outer_thinking_buckets),
            inner_thinking_buckets=inner_thinking_buckets,
            inner_thinking_count=len(inner_thinking_buckets),
            category_buckets=category_buckets,
            category_count=len(category_buckets),
            topic_buckets=topic_buckets,
            topic_count=len(topic_buckets),
            sequences=sequences,
            sequence_count=len(sequences),
            speech_opening=speech_opening,
            speech_segments=speech_segments,
            speech_closing=speech_closing,
            truncated=truncated,
            warnings=warnings,
        )

    def _load_utterances(
        self,
        args: ContextHabitCategorizationArgs,
        max_source_bytes: int,
        max_total_bytes: int,
        split_mode: str,
    ) -> tuple[list[HabitUtterance], int, bool]:
        if args.utterances:
            utterances = list(args.utterances)
            total_bytes = 0
            limited: list[HabitUtterance] = []
            truncated = False
            for utt in utterances:
                data = utt.text.encode("utf-8")
                if len(data) > max_source_bytes:
                    raise ToolError(
                        f"utterance exceeds max_source_bytes ({len(data)} > {max_source_bytes})."
                    )
                if total_bytes + len(data) > max_total_bytes:
                    truncated = True
                    break
                total_bytes += len(data)
                limited.append(utt)
            return limited, len(utterances), truncated

        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")

        if args.content is None and args.path is None:
            return [], 0, False

        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {max_source_bytes})."
                )
            content = args.content
        else:
            path = self._resolve_path(args.path or "")
            if path.is_dir():
                raise ToolError(f"Path is a directory: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            content = path.read_text("utf-8", errors="ignore")

        utterances: list[HabitUtterance] = []
        if split_mode == "sentences":
            for sentence in SENTENCE_RE.findall(content):
                text = sentence.strip()
                if text:
                    utterances.append(HabitUtterance(text=text))
            return utterances, len(utterances), False

        for line in content.splitlines():
            text = line.strip()
            if not text:
                continue
            speaker = None
            match = SPEAKER_RE.match(text)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
            utterances.append(HabitUtterance(text=text, speaker=speaker))
        return utterances, len(utterances), False

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _tokenize(self, text: str, min_token_length: int) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= min_token_length
        ]

    def _keywords(self, tokens: list[str], max_keywords: int) -> list[str]:
        candidates = [token for token in tokens if token not in STOPWORDS]
        counter = Counter(candidates)
        most_common = counter.most_common(max_keywords)
        return [token for token, _ in most_common]

    def _match_markers(self, text: str, tokens: list[str], markers: list[str]) -> list[str]:
        lowered = text.lower()
        matches: list[str] = []
        token_set = set(tokens)
        for marker in markers:
            if " " in marker:
                if marker in lowered:
                    matches.append(marker)
            else:
                if marker in token_set:
                    matches.append(marker)
        return matches

    def _assign_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> list[str]:
        lowered = text.lower()
        categories: list[str] = []
        token_set = set(tokens)
        for category in DEFAULT_CATEGORIES:
            name = str(category["name"])
            markers = category.get("markers", [])
            matched = False
            for marker in markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        break
                else:
                    if marker_text in token_set:
                        matched = True
                        break
            if matched:
                categories.append(name)
        if not categories:
            categories.append("general")
        if max_categories is not None and max_categories > 0:
            return categories[:max_categories]
        return categories

    def _assign_origin_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in ORIGIN_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if not categories:
            categories.append("unspecified")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_dendrite_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in DENDRITE_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if not categories:
            categories.append("unspecified")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_timeframe_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in TIMEFRAME_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)

        date_matches = DATE_RE.findall(text)
        if date_matches:
            categories.append("calendar_date")
            markers.extend(date_matches)
        year_matches = YEAR_RE.findall(text)
        if year_matches:
            categories.append("year_marker")
            markers.extend(year_matches)
        time_matches = TIME_RE.findall(text)
        if time_matches:
            categories.append("clock_time")
            markers.extend(time_matches)
        duration_matches = DURATION_RE.findall(text)
        if duration_matches:
            categories.append("duration")
            markers.extend(duration_matches)

        if not categories:
            categories.append("unspecified")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_onset_categories(
        self,
        text: str,
        tokens: list[str],
        timeframe_categories: list[str],
        timeframe_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        onset_markers = self._match_markers(text, tokens, ONSET_MARKERS)
        markers.extend(onset_markers)

        for category in ONSET_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)

        if onset_markers and timeframe_categories and timeframe_categories != ["unspecified"]:
            for category in timeframe_categories:
                if category not in categories:
                    categories.append(category)
            markers.extend(timeframe_markers)

        if onset_markers and not categories:
            categories.append("unspecified")

        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_frequency_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in FREQUENCY_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)

        freq_matches = FREQUENCY_RE.findall(text)
        if freq_matches:
            categories.append("numeric_rate")
            markers.extend(freq_matches)
        slash_matches = FREQUENCY_SLASH_RE.findall(text)
        if slash_matches:
            categories.append("numeric_rate")
            markers.extend(slash_matches)

        if not categories:
            categories.append("unspecified")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_neural_firing_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in NEURAL_FIRING_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                markers.append(marker_text)
            if matched:
                categories.append(name)
        neuro_markers = self._match_markers(text, tokens, NEUROTRANSMITTER_MARKERS)
        change_markers = self._match_markers(text, tokens, CHANGE_MARKERS)
        if neuro_markers and change_markers:
            categories.append("neurotransmitter_change")
            markers.extend(neuro_markers)
            markers.extend(change_markers)
        if not categories:
            categories.append("unspecified")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_muscle_memory_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        muscle_markers = self._match_markers(text, tokens, MUSCLE_MEMORY_MARKERS)
        change_markers = self._match_markers(text, tokens, CHANGE_MARKERS)
        activation_markers = self._match_markers(text, tokens, MOTOR_ACTIVATION_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if muscle_markers:
            if change_markers:
                categories.append("muscle_memory_change")
            else:
                categories.append("muscle_memory")
            if activation_markers:
                categories.append("motor_skill_activation")
                markers.extend(activation_markers)
            markers.extend(muscle_markers)
            markers.extend(change_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_pattern_categories(
        self,
        text: str,
        tokens: list[str],
        sequence_markers: list[str],
        repetition_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in PATTERN_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if sequence_markers and "sequence" not in categories:
            categories.append("sequence")
            markers.extend(sequence_markers)
        if repetition_markers and "cycle" not in categories:
            categories.append("cycle")
            markers.extend(repetition_markers)
        mistake_markers = self._match_markers(text, tokens, MISTAKE_MARKERS)
        if mistake_markers and repetition_markers:
            categories.append("repeated_mistake")
            markers.extend(mistake_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_structure_categories(
        self,
        text: str,
        tokens: list[str],
        sequence_markers: list[str],
        repetition_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in STRUCTURE_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if sequence_markers and "sequence" not in categories:
            categories.append("sequence")
            markers.extend(sequence_markers)
        if repetition_markers and "loop" not in categories:
            categories.append("loop")
            markers.extend(repetition_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_external_force_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in EXTERNAL_FORCE_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_influence_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        influence_markers = self._match_markers(text, tokens, INFLUENCE_MARKERS)
        action_markers = self._match_markers(text, tokens, ACTION_MARKERS)
        for category in INFLUENCE_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if influence_markers and action_markers:
            categories.append("behavior_influence")
            markers.extend(influence_markers)
            markers.extend(action_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_external_repetition_categories(
        self,
        text: str,
        tokens: list[str],
        external_force_markers: list[str],
        repetition_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        categories: list[str] = []
        markers: list[str] = []
        if external_force_markers and repetition_markers:
            categories.append("external_repetition")
            markers.extend(external_force_markers)
            markers.extend(repetition_markers)
        lowered = text.lower()
        if "external repetition" in lowered or "repeated external" in lowered:
            if "external_repetition" not in categories:
                categories.append("external_repetition")
            markers.append("external repetition")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_event_sequence_categories(
        self,
        text: str,
        tokens: list[str],
        sequence_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        event_markers = self._match_markers(text, tokens, EVENT_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        lowered = text.lower()
        has_phrase = "sequence of events" in lowered or "chain of events" in lowered
        if (event_markers and sequence_markers) or has_phrase:
            categories.append("event_sequence")
            markers.extend(event_markers)
            markers.extend(sequence_markers)
            if has_phrase:
                markers.append("sequence of events")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_social_learning_categories(
        self,
        text: str,
        tokens: list[str],
        repetition_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in SOCIAL_LEARNING_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if categories and repetition_markers:
            categories.append("social_learning_repetition")
            markers.extend(repetition_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_provenance_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in PROVENANCE_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        provenance_markers = self._match_markers(text, tokens, PROVENANCE_MARKERS)
        markers.extend(provenance_markers)
        development_markers = self._match_markers(text, tokens, ONSET_MARKERS + CHANGE_MARKERS)
        if development_markers:
            categories.append("development")
            markers.extend(development_markers)
        if not categories and provenance_markers:
            categories.append("origin")
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_activation_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        activation_markers = self._match_markers(text, tokens, MOTOR_ACTIVATION_MARKERS)
        brain_region_markers = self._match_markers(text, tokens, BRAIN_REGION_MARKERS)
        for category in ACTIVATION_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if brain_region_markers and (activation_markers or categories):
            categories.append("brain_region_activation")
            markers.extend(brain_region_markers)
            markers.extend(activation_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_dendro_motor_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        token_set = set(tokens)
        dendrite_markers: list[str] = []
        for category in DENDRITE_CATEGORIES:
            for marker in category.get("markers", []):
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        dendrite_markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        dendrite_markers.append(marker_text)
        motor_markers = self._match_markers(text, tokens, MUSCLE_MEMORY_MARKERS)
        response_markers = self._match_markers(text, tokens, DENDRO_MOTOR_MARKERS)
        action_markers = self._match_markers(text, tokens, DENDRITE_ACTION_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if dendrite_markers and (motor_markers or action_markers):
            categories.append("dendro_motor")
            markers.extend(dendrite_markers)
            markers.extend(motor_markers)
            markers.extend(action_markers)
            markers.extend(response_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_music_categories(
        self,
        text: str,
        tokens: list[str],
        repetition_markers: list[str],
        timeframe_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        music_markers = self._match_markers(text, tokens, MUSIC_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if music_markers:
            categories.append("music_habit")
            markers.extend(music_markers)
            if repetition_markers:
                categories.append("music_repetition")
                markers.extend(repetition_markers)
            if timeframe_markers:
                categories.append("music_timeframe")
                markers.extend(timeframe_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_intent_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        lowered = text.lower()
        categories: list[str] = []
        markers: list[str] = []
        token_set = set(tokens)
        for category in INTENT_CATEGORIES:
            name = str(category["name"])
            cat_markers = category.get("markers", [])
            matched = False
            for marker in cat_markers:
                marker_text = str(marker)
                if " " in marker_text:
                    if marker_text in lowered:
                        matched = True
                        markers.append(marker_text)
                else:
                    if marker_text in token_set:
                        matched = True
                        markers.append(marker_text)
            if matched:
                categories.append(name)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_action_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        action_markers = self._match_markers(text, tokens, ACTION_MARKERS)
        social_markers = self._match_markers(text, tokens, SOCIAL_MARKERS)
        concentration_markers = self._match_markers(text, tokens, CONCENTRATION_MARKERS)
        automatic_markers = self._match_markers(text, tokens, AUTOMATIC_MARKERS)
        subconscious_markers = self._match_markers(text, tokens, SUBCONSCIOUS_MARKERS)
        reinforcement_markers = self._match_markers(text, tokens, REINFORCEMENT_MARKERS)
        response_markers = self._match_markers(text, tokens, RESPONSE_MARKERS)
        passive_markers = self._match_markers(text, tokens, PASSIVE_MARKERS)
        aggressive_markers = self._match_markers(text, tokens, AGGRESSIVE_MARKERS)
        intuitive_markers = self._match_markers(text, tokens, INTUITIVE_MARKERS)
        precognitive_markers = self._match_markers(text, tokens, PRECOGNITIVE_MARKERS)
        affect_markers = self._match_markers(text, tokens, AFFECT_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if action_markers:
            if concentration_markers:
                categories.append("concentrated_action")
                markers.extend(concentration_markers)
            if automatic_markers:
                categories.append("automatic_response")
                markers.extend(automatic_markers)
            if subconscious_markers:
                categories.append("subconscious_behavior")
                markers.extend(subconscious_markers)
            if reinforcement_markers:
                categories.append("reinforced_behavior")
                markers.extend(reinforcement_markers)
            if response_markers and social_markers:
                categories.append("social_response")
                markers.extend(response_markers)
                markers.extend(social_markers)
            if passive_markers:
                categories.append("passive_behavior")
                markers.extend(passive_markers)
            if aggressive_markers:
                categories.append("aggressive_behavior")
                markers.extend(aggressive_markers)
            if intuitive_markers:
                categories.append("intuitive_action")
                markers.extend(intuitive_markers)
            if precognitive_markers:
                categories.append("precognitive_action")
                markers.extend(precognitive_markers)
            if affect_markers:
                categories.append("action_affect")
                markers.extend(affect_markers)
            if social_markers:
                categories.append("social_action")
                markers.extend(social_markers)
            elif not concentration_markers:
                categories.append("action")
            markers.extend(action_markers)
        elif passive_markers:
            categories.append("passive_behavior")
            markers.extend(passive_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_word_action_categories(
        self,
        text: str,
        tokens: list[str],
        action_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        word_markers = self._match_markers(text, tokens, WORD_MARKERS)
        understand_markers = self._match_markers(text, tokens, UNDERSTAND_MARKERS)
        perform_markers = self._match_markers(text, tokens, PERFORM_MARKERS)
        self_markers = self._match_markers(text, tokens, SELF_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if word_markers and understand_markers and action_markers:
            categories.append("spoken_understood_performed")
            markers.extend(word_markers)
            markers.extend(understand_markers)
            markers.extend(action_markers)
            markers.extend(perform_markers)
        if word_markers and self_markers:
            categories.append("self_spoken_words")
            markers.extend(self_markers)
            markers.extend(word_markers)
        if word_markers and action_markers:
            categories.append("word_action")
            markers.extend(word_markers)
            markers.extend(action_markers)
        elif word_markers:
            categories.append("word_influence")
            markers.extend(word_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_social_word_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        word_markers = self._match_markers(text, tokens, WORD_MARKERS)
        social_markers = self._match_markers(text, tokens, SOCIAL_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if word_markers and social_markers:
            categories.append("social_word_influence")
            markers.extend(word_markers)
            markers.extend(social_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_music_belief_action_categories(
        self,
        text: str,
        tokens: list[str],
        music_markers: list[str],
        belief_markers: list[str],
        action_markers: list[str],
        max_categories: int,
    ) -> tuple[list[str], list[str]]:
        word_markers = self._match_markers(text, tokens, WORD_MARKERS)
        resonance_markers = self._match_markers(text, tokens, MUSIC_RESONANCE_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if music_markers and belief_markers and action_markers:
            categories.append("music_belief_action")
            markers.extend(music_markers)
            markers.extend(word_markers)
            markers.extend(belief_markers)
            markers.extend(action_markers)
            markers.extend(resonance_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_belief_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        belief_markers = self._match_markers(text, tokens, BELIEF_MARKERS)
        truth_markers = self._match_markers(text, tokens, TRUTH_MARKERS)
        if truth_markers:
            for marker in truth_markers:
                if marker not in belief_markers:
                    belief_markers.append(marker)
        divine_markers = self._match_markers(text, tokens, DIVINE_MARKERS)
        if divine_markers:
            for marker in divine_markers:
                if marker not in belief_markers:
                    belief_markers.append(marker)
        inner_markers = self._match_markers(text, tokens, INNER_BELIEF_MARKERS)
        outer_markers = self._match_markers(text, tokens, OUTER_BELIEF_MARKERS)
        social_markers = self._match_markers(text, tokens, SOCIAL_MARKERS)
        resonance_markers = self._match_markers(text, tokens, MUSIC_RESONANCE_MARKERS)
        deep_markers = self._match_markers(text, tokens, DEEP_RESONANCE_MARKERS)
        influence_markers = self._match_markers(text, tokens, INFLUENCE_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if belief_markers:
            if divine_markers:
                categories.append("divine_belief")
                markers.extend(divine_markers)
            if inner_markers and social_markers and resonance_markers:
                categories.append("other_inner_belief_resonance")
                markers.extend(inner_markers)
                markers.extend(social_markers)
                markers.extend(resonance_markers)
            if outer_markers and social_markers and influence_markers:
                categories.append("outer_belief_influence")
                markers.extend(outer_markers)
                markers.extend(social_markers)
                markers.extend(influence_markers)
            if resonance_markers and deep_markers:
                categories.append("deep_belief_resonance")
                markers.extend(resonance_markers)
                markers.extend(deep_markers)
            if truth_markers:
                categories.append("truth_belief")
                markers.extend(truth_markers)
            if inner_markers and "inner_belief" not in categories:
                categories.append("inner_belief")
                markers.extend(inner_markers)
            if outer_markers and "outer_belief" not in categories:
                categories.append("outer_belief")
                markers.extend(outer_markers)
            if not inner_markers and not outer_markers:
                categories.append("belief")
            markers.extend(belief_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_virtue_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        virtue_markers = self._match_markers(text, tokens, VIRTUE_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if virtue_markers:
            categories.append("virtue")
            markers.extend(virtue_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_outer_thinking_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        outer_markers = self._match_markers(text, tokens, OUTER_THINKING_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if outer_markers:
            categories.append("outer_thinking")
            markers.extend(outer_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _assign_inner_thinking_categories(
        self, text: str, tokens: list[str], max_categories: int
    ) -> tuple[list[str], list[str]]:
        inner_markers = self._match_markers(text, tokens, INNER_THINKING_MARKERS)
        subconscious_markers = self._match_markers(text, tokens, SUBCONSCIOUS_MARKERS)
        subliminal_markers = self._match_markers(text, tokens, SUBLIMINAL_MARKERS)
        categories: list[str] = []
        markers: list[str] = []
        if inner_markers:
            categories.append("inner_thinking")
            markers.extend(inner_markers)
        if subconscious_markers:
            categories.append("subconscious_thought")
            markers.extend(subconscious_markers)
        if subliminal_markers:
            categories.append("subliminal_thought")
            markers.extend(subliminal_markers)
        if max_categories is not None and max_categories > 0:
            categories = categories[:max_categories]
        unique_markers: list[str] = []
        for marker in markers:
            if marker not in unique_markers:
                unique_markers.append(marker)
        return categories, unique_markers

    def _build_category_buckets(
        self,
        habit_categories: list[list[str]],
        habit_tokens: list[list[str]],
        max_category_terms: int,
    ) -> list[HabitCategoryBucket]:
        category_indices: dict[str, list[int]] = defaultdict(list)
        category_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_categories, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                category_indices[category].append(idx)
                category_terms[category].update(tokens)

        buckets: list[HabitCategoryBucket] = []
        for category, indices in category_indices.items():
            top_terms = [
                term for term, _ in category_terms[category].most_common(max_category_terms)
            ]
            buckets.append(
                HabitCategoryBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_origin_buckets(
        self,
        habit_origins: list[list[str]],
        habit_tokens: list[list[str]],
        max_origin_terms: int,
    ) -> list[HabitOriginBucket]:
        origin_indices: dict[str, list[int]] = defaultdict(list)
        origin_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, origins in enumerate(habit_origins, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for origin in origins:
                origin_indices[origin].append(idx)
                origin_terms[origin].update(tokens)

        buckets: list[HabitOriginBucket] = []
        for origin, indices in origin_indices.items():
            top_terms = [
                term for term, _ in origin_terms[origin].most_common(max_origin_terms)
            ]
            buckets.append(
                HabitOriginBucket(
                    category=origin,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_dendrite_buckets(
        self,
        habit_dendrites: list[list[str]],
        habit_tokens: list[list[str]],
        max_dendrite_terms: int,
    ) -> list[HabitDendriteBucket]:
        dendrite_indices: dict[str, list[int]] = defaultdict(list)
        dendrite_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_dendrites, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                dendrite_indices[category].append(idx)
                dendrite_terms[category].update(tokens)

        buckets: list[HabitDendriteBucket] = []
        for category, indices in dendrite_indices.items():
            top_terms = [
                term for term, _ in dendrite_terms[category].most_common(max_dendrite_terms)
            ]
            buckets.append(
                HabitDendriteBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_timeframe_buckets(
        self,
        habit_timeframes: list[list[str]],
        habit_tokens: list[list[str]],
        max_timeframe_terms: int,
    ) -> list[HabitTimeframeBucket]:
        timeframe_indices: dict[str, list[int]] = defaultdict(list)
        timeframe_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_timeframes, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                timeframe_indices[category].append(idx)
                timeframe_terms[category].update(tokens)

        buckets: list[HabitTimeframeBucket] = []
        for category, indices in timeframe_indices.items():
            top_terms = [
                term for term, _ in timeframe_terms[category].most_common(max_timeframe_terms)
            ]
            buckets.append(
                HabitTimeframeBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_onset_buckets(
        self,
        habit_onsets: list[list[str]],
        habit_tokens: list[list[str]],
        max_onset_terms: int,
    ) -> list[HabitOnsetBucket]:
        onset_indices: dict[str, list[int]] = defaultdict(list)
        onset_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_onsets, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                onset_indices[category].append(idx)
                onset_terms[category].update(tokens)

        buckets: list[HabitOnsetBucket] = []
        for category, indices in onset_indices.items():
            top_terms = [
                term for term, _ in onset_terms[category].most_common(max_onset_terms)
            ]
            buckets.append(
                HabitOnsetBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_frequency_buckets(
        self,
        habit_frequencies: list[list[str]],
        habit_tokens: list[list[str]],
        max_frequency_terms: int,
    ) -> list[HabitFrequencyBucket]:
        frequency_indices: dict[str, list[int]] = defaultdict(list)
        frequency_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_frequencies, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                frequency_indices[category].append(idx)
                frequency_terms[category].update(tokens)

        buckets: list[HabitFrequencyBucket] = []
        for category, indices in frequency_indices.items():
            top_terms = [
                term for term, _ in frequency_terms[category].most_common(max_frequency_terms)
            ]
            buckets.append(
                HabitFrequencyBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_neural_firing_buckets(
        self,
        habit_neural_firings: list[list[str]],
        habit_tokens: list[list[str]],
        max_neural_firing_terms: int,
    ) -> list[HabitNeuralFiringBucket]:
        firing_indices: dict[str, list[int]] = defaultdict(list)
        firing_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_neural_firings, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                firing_indices[category].append(idx)
                firing_terms[category].update(tokens)

        buckets: list[HabitNeuralFiringBucket] = []
        for category, indices in firing_indices.items():
            top_terms = [
                term for term, _ in firing_terms[category].most_common(max_neural_firing_terms)
            ]
            buckets.append(
                HabitNeuralFiringBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_muscle_memory_buckets(
        self,
        habit_muscle_memories: list[list[str]],
        habit_tokens: list[list[str]],
        max_muscle_memory_terms: int,
    ) -> list[HabitMuscleMemoryBucket]:
        muscle_indices: dict[str, list[int]] = defaultdict(list)
        muscle_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_muscle_memories, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                muscle_indices[category].append(idx)
                muscle_terms[category].update(tokens)

        buckets: list[HabitMuscleMemoryBucket] = []
        for category, indices in muscle_indices.items():
            top_terms = [
                term for term, _ in muscle_terms[category].most_common(max_muscle_memory_terms)
            ]
            buckets.append(
                HabitMuscleMemoryBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_pattern_buckets(
        self,
        habit_patterns: list[list[str]],
        habit_tokens: list[list[str]],
        max_pattern_terms: int,
    ) -> list[HabitPatternBucket]:
        pattern_indices: dict[str, list[int]] = defaultdict(list)
        pattern_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_patterns, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                pattern_indices[category].append(idx)
                pattern_terms[category].update(tokens)

        buckets: list[HabitPatternBucket] = []
        for category, indices in pattern_indices.items():
            top_terms = [
                term for term, _ in pattern_terms[category].most_common(max_pattern_terms)
            ]
            buckets.append(
                HabitPatternBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_structure_buckets(
        self,
        habit_structures: list[list[str]],
        habit_tokens: list[list[str]],
        max_structure_terms: int,
    ) -> list[HabitStructureBucket]:
        structure_indices: dict[str, list[int]] = defaultdict(list)
        structure_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_structures, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                structure_indices[category].append(idx)
                structure_terms[category].update(tokens)

        buckets: list[HabitStructureBucket] = []
        for category, indices in structure_indices.items():
            top_terms = [
                term for term, _ in structure_terms[category].most_common(max_structure_terms)
            ]
            buckets.append(
                HabitStructureBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_external_force_buckets(
        self,
        habit_external_forces: list[list[str]],
        habit_tokens: list[list[str]],
        max_external_force_terms: int,
    ) -> list[HabitExternalForceBucket]:
        force_indices: dict[str, list[int]] = defaultdict(list)
        force_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_external_forces, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                force_indices[category].append(idx)
                force_terms[category].update(tokens)

        buckets: list[HabitExternalForceBucket] = []
        for category, indices in force_indices.items():
            top_terms = [
                term for term, _ in force_terms[category].most_common(max_external_force_terms)
            ]
            buckets.append(
                HabitExternalForceBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_influence_buckets(
        self,
        habit_influences: list[list[str]],
        habit_tokens: list[list[str]],
        max_influence_terms: int,
    ) -> list[HabitInfluenceBucket]:
        influence_indices: dict[str, list[int]] = defaultdict(list)
        influence_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_influences, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                influence_indices[category].append(idx)
                influence_terms[category].update(tokens)

        buckets: list[HabitInfluenceBucket] = []
        for category, indices in influence_indices.items():
            top_terms = [
                term for term, _ in influence_terms[category].most_common(max_influence_terms)
            ]
            buckets.append(
                HabitInfluenceBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_external_repetition_buckets(
        self,
        habit_external_repetitions: list[list[str]],
        habit_tokens: list[list[str]],
        max_external_repetition_terms: int,
    ) -> list[HabitExternalRepetitionBucket]:
        repetition_indices: dict[str, list[int]] = defaultdict(list)
        repetition_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_external_repetitions, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                repetition_indices[category].append(idx)
                repetition_terms[category].update(tokens)

        buckets: list[HabitExternalRepetitionBucket] = []
        for category, indices in repetition_indices.items():
            top_terms = [
                term
                for term, _ in repetition_terms[category].most_common(
                    max_external_repetition_terms
                )
            ]
            buckets.append(
                HabitExternalRepetitionBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_event_sequence_buckets(
        self,
        habit_event_sequences: list[list[str]],
        habit_tokens: list[list[str]],
        max_event_sequence_terms: int,
    ) -> list[HabitEventSequenceBucket]:
        sequence_indices: dict[str, list[int]] = defaultdict(list)
        sequence_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_event_sequences, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                sequence_indices[category].append(idx)
                sequence_terms[category].update(tokens)

        buckets: list[HabitEventSequenceBucket] = []
        for category, indices in sequence_indices.items():
            top_terms = [
                term for term, _ in sequence_terms[category].most_common(max_event_sequence_terms)
            ]
            buckets.append(
                HabitEventSequenceBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_social_learning_buckets(
        self,
        habit_social_learnings: list[list[str]],
        habit_tokens: list[list[str]],
        max_social_learning_terms: int,
    ) -> list[HabitSocialLearningBucket]:
        learning_indices: dict[str, list[int]] = defaultdict(list)
        learning_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_social_learnings, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                learning_indices[category].append(idx)
                learning_terms[category].update(tokens)

        buckets: list[HabitSocialLearningBucket] = []
        for category, indices in learning_indices.items():
            top_terms = [
                term
                for term, _ in learning_terms[category].most_common(max_social_learning_terms)
            ]
            buckets.append(
                HabitSocialLearningBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_provenance_buckets(
        self,
        habit_provenances: list[list[str]],
        habit_tokens: list[list[str]],
        max_provenance_terms: int,
    ) -> list[HabitProvenanceBucket]:
        provenance_indices: dict[str, list[int]] = defaultdict(list)
        provenance_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_provenances, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                provenance_indices[category].append(idx)
                provenance_terms[category].update(tokens)

        buckets: list[HabitProvenanceBucket] = []
        for category, indices in provenance_indices.items():
            top_terms = [
                term for term, _ in provenance_terms[category].most_common(max_provenance_terms)
            ]
            buckets.append(
                HabitProvenanceBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_activation_buckets(
        self,
        habit_activations: list[list[str]],
        habit_tokens: list[list[str]],
        max_activation_terms: int,
    ) -> list[HabitActivationBucket]:
        activation_indices: dict[str, list[int]] = defaultdict(list)
        activation_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_activations, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                activation_indices[category].append(idx)
                activation_terms[category].update(tokens)

        buckets: list[HabitActivationBucket] = []
        for category, indices in activation_indices.items():
            top_terms = [
                term for term, _ in activation_terms[category].most_common(max_activation_terms)
            ]
            buckets.append(
                HabitActivationBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_dendro_motor_buckets(
        self,
        habit_dendro_motors: list[list[str]],
        habit_tokens: list[list[str]],
        max_dendro_motor_terms: int,
    ) -> list[HabitDendroMotorBucket]:
        dendro_indices: dict[str, list[int]] = defaultdict(list)
        dendro_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_dendro_motors, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                dendro_indices[category].append(idx)
                dendro_terms[category].update(tokens)

        buckets: list[HabitDendroMotorBucket] = []
        for category, indices in dendro_indices.items():
            top_terms = [
                term
                for term, _ in dendro_terms[category].most_common(max_dendro_motor_terms)
            ]
            buckets.append(
                HabitDendroMotorBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_music_buckets(
        self,
        habit_music: list[list[str]],
        habit_tokens: list[list[str]],
        max_music_terms: int,
    ) -> list[HabitMusicBucket]:
        music_indices: dict[str, list[int]] = defaultdict(list)
        music_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_music, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                music_indices[category].append(idx)
                music_terms[category].update(tokens)

        buckets: list[HabitMusicBucket] = []
        for category, indices in music_indices.items():
            top_terms = [
                term for term, _ in music_terms[category].most_common(max_music_terms)
            ]
            buckets.append(
                HabitMusicBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_intent_buckets(
        self,
        habit_intents: list[list[str]],
        habit_tokens: list[list[str]],
        max_intent_terms: int,
    ) -> list[HabitIntentBucket]:
        intent_indices: dict[str, list[int]] = defaultdict(list)
        intent_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_intents, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                intent_indices[category].append(idx)
                intent_terms[category].update(tokens)

        buckets: list[HabitIntentBucket] = []
        for category, indices in intent_indices.items():
            top_terms = [
                term for term, _ in intent_terms[category].most_common(max_intent_terms)
            ]
            buckets.append(
                HabitIntentBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_action_buckets(
        self,
        habit_actions: list[list[str]],
        habit_tokens: list[list[str]],
        max_action_terms: int,
    ) -> list[HabitActionBucket]:
        action_indices: dict[str, list[int]] = defaultdict(list)
        action_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_actions, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                action_indices[category].append(idx)
                action_terms[category].update(tokens)

        buckets: list[HabitActionBucket] = []
        for category, indices in action_indices.items():
            top_terms = [
                term for term, _ in action_terms[category].most_common(max_action_terms)
            ]
            buckets.append(
                HabitActionBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_word_action_buckets(
        self,
        habit_word_actions: list[list[str]],
        habit_tokens: list[list[str]],
        max_word_action_terms: int,
    ) -> list[HabitWordActionBucket]:
        word_indices: dict[str, list[int]] = defaultdict(list)
        word_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_word_actions, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                word_indices[category].append(idx)
                word_terms[category].update(tokens)

        buckets: list[HabitWordActionBucket] = []
        for category, indices in word_indices.items():
            top_terms = [
                term for term, _ in word_terms[category].most_common(max_word_action_terms)
            ]
            buckets.append(
                HabitWordActionBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_social_word_buckets(
        self,
        habit_social_words: list[list[str]],
        habit_tokens: list[list[str]],
        max_social_word_terms: int,
    ) -> list[HabitSocialWordBucket]:
        social_indices: dict[str, list[int]] = defaultdict(list)
        social_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_social_words, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                social_indices[category].append(idx)
                social_terms[category].update(tokens)

        buckets: list[HabitSocialWordBucket] = []
        for category, indices in social_indices.items():
            top_terms = [
                term for term, _ in social_terms[category].most_common(max_social_word_terms)
            ]
            buckets.append(
                HabitSocialWordBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_music_belief_action_buckets(
        self,
        habit_music_belief_actions: list[list[str]],
        habit_tokens: list[list[str]],
        max_music_belief_action_terms: int,
    ) -> list[HabitMusicBeliefActionBucket]:
        music_indices: dict[str, list[int]] = defaultdict(list)
        music_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_music_belief_actions, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                music_indices[category].append(idx)
                music_terms[category].update(tokens)

        buckets: list[HabitMusicBeliefActionBucket] = []
        for category, indices in music_indices.items():
            top_terms = [
                term
                for term, _ in music_terms[category].most_common(max_music_belief_action_terms)
            ]
            buckets.append(
                HabitMusicBeliefActionBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_belief_buckets(
        self,
        habit_beliefs: list[list[str]],
        habit_tokens: list[list[str]],
        max_belief_terms: int,
    ) -> list[HabitBeliefBucket]:
        belief_indices: dict[str, list[int]] = defaultdict(list)
        belief_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_beliefs, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                belief_indices[category].append(idx)
                belief_terms[category].update(tokens)

        buckets: list[HabitBeliefBucket] = []
        for category, indices in belief_indices.items():
            top_terms = [
                term for term, _ in belief_terms[category].most_common(max_belief_terms)
            ]
            buckets.append(
                HabitBeliefBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_virtue_buckets(
        self,
        habit_virtues: list[list[str]],
        habit_tokens: list[list[str]],
        max_virtue_terms: int,
    ) -> list[HabitVirtueBucket]:
        virtue_indices: dict[str, list[int]] = defaultdict(list)
        virtue_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_virtues, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                virtue_indices[category].append(idx)
                virtue_terms[category].update(tokens)

        buckets: list[HabitVirtueBucket] = []
        for category, indices in virtue_indices.items():
            top_terms = [
                term for term, _ in virtue_terms[category].most_common(max_virtue_terms)
            ]
            buckets.append(
                HabitVirtueBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_outer_thinking_buckets(
        self,
        habit_outer_thinking: list[list[str]],
        habit_tokens: list[list[str]],
        max_outer_thinking_terms: int,
    ) -> list[HabitOuterThinkingBucket]:
        outer_indices: dict[str, list[int]] = defaultdict(list)
        outer_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_outer_thinking, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                outer_indices[category].append(idx)
                outer_terms[category].update(tokens)

        buckets: list[HabitOuterThinkingBucket] = []
        for category, indices in outer_indices.items():
            top_terms = [
                term for term, _ in outer_terms[category].most_common(max_outer_thinking_terms)
            ]
            buckets.append(
                HabitOuterThinkingBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_inner_thinking_buckets(
        self,
        habit_inner_thinking: list[list[str]],
        habit_tokens: list[list[str]],
        max_inner_thinking_terms: int,
    ) -> list[HabitInnerThinkingBucket]:
        inner_indices: dict[str, list[int]] = defaultdict(list)
        inner_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(habit_inner_thinking, start=1):
            tokens = [token for token in habit_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                inner_indices[category].append(idx)
                inner_terms[category].update(tokens)

        buckets: list[HabitInnerThinkingBucket] = []
        for category, indices in inner_indices.items():
            top_terms = [
                term
                for term, _ in inner_terms[category].most_common(max_inner_thinking_terms)
            ]
            buckets.append(
                HabitInnerThinkingBucket(
                    category=category,
                    count=len(indices),
                    habit_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_topic_buckets(
        self,
        habit_tokens: list[list[str]],
        max_topics: int,
    ) -> list[HabitTopicBucket]:
        token_counter: Counter = Counter()
        term_indices: dict[str, list[int]] = defaultdict(list)
        for idx, tokens in enumerate(habit_tokens, start=1):
            filtered = [token for token in tokens if token not in STOPWORDS]
            token_counter.update(filtered)
            for token in set(filtered):
                term_indices[token].append(idx)

        buckets: list[HabitTopicBucket] = []
        for term, count in token_counter.most_common(max_topics):
            indices = term_indices.get(term, [])
            buckets.append(
                HabitTopicBucket(
                    term=term,
                    count=count,
                    habit_indices=indices,
                )
            )
        return buckets

    def _build_sequences(
        self,
        habits: list[HabitEntry],
        include_singletons: bool,
        max_sequences: int,
        max_steps_per_sequence: int,
    ) -> list[HabitSequence]:
        sequences: list[HabitSequence] = []
        current: list[HabitEntry] = []
        markers: list[str] = []
        assigned: set[int] = set()

        def flush() -> None:
            nonlocal current, markers
            if not current:
                return
            sequences.append(
                self._make_sequence(current, markers, max_steps_per_sequence)
            )
            for entry in current:
                assigned.add(entry.index)
            current = []
            markers = []

        for entry in habits:
            entry_markers = entry.sequence_markers
            has_connector = self._has_sequence_connector(entry)
            if entry_markers or (current and has_connector):
                current.append(entry)
                markers.extend(entry_markers)
            else:
                if current:
                    flush()
        if current:
            flush()

        if include_singletons:
            for entry in habits:
                if entry.index in assigned:
                    continue
                sequences.append(
                    self._make_sequence([entry], entry.sequence_markers, max_steps_per_sequence)
                )

        if max_sequences and len(sequences) > max_sequences:
            sequences = sequences[:max_sequences]

        for idx, sequence in enumerate(sequences, start=1):
            sequence.index = idx
        return sequences

    def _has_sequence_connector(self, entry: HabitEntry) -> bool:
        tokens = {token.lower() for token in WORD_RE.findall(entry.text)}
        return any(token in tokens for token in SEQUENCE_CONNECTORS)

    def _make_sequence(
        self,
        entries: list[HabitEntry],
        markers: list[str],
        max_steps_per_sequence: int,
    ) -> HabitSequence:
        limited = entries[:max_steps_per_sequence]
        steps: list[HabitSequenceStep] = []
        habit_indices = [entry.index for entry in limited]
        for idx, entry in enumerate(limited, start=1):
            steps.append(
                HabitSequenceStep(
                    index=idx,
                    habit_index=entry.index,
                    preview=entry.preview,
                    keywords=entry.keywords,
                    sequence_markers=entry.sequence_markers,
                )
            )
        markers_unique = []
        for marker in markers:
            if marker not in markers_unique:
                markers_unique.append(marker)
        cue = "Sequence steps: " + ", ".join(str(idx) for idx in habit_indices)
        if markers_unique:
            cue = f"{cue}. Markers: {', '.join(markers_unique[:4])}."
        return HabitSequence(
            index=0,
            step_count=len(steps),
            habit_indices=habit_indices,
            sequence_markers=markers_unique,
            steps=steps,
            cue=cue,
        )

    def _speech_opening(
        self,
        args: ContextHabitCategorizationArgs,
        habit_count: int,
        sequence_count: int,
        repetition_count: int,
        origin_count: int,
        dendrite_count: int,
        timeframe_count: int,
        onset_count: int,
        frequency_count: int,
        neural_firing_count: int,
        muscle_memory_count: int,
        pattern_count: int,
        structure_count: int,
        external_force_count: int,
        influence_count: int,
        external_repetition_count: int,
        event_sequence_count: int,
        social_learning_count: int,
        provenance_count: int,
        activation_count: int,
        dendro_motor_count: int,
        music_count: int,
        intent_count: int,
        action_count: int,
        word_action_count: int,
        social_word_count: int,
        music_belief_action_count: int,
        belief_count: int,
        virtue_count: int,
        outer_thinking_count: int,
        inner_thinking_count: int,
    ) -> str:
        if not args.include_opening:
            return ""
        parts = [
            f"Habit categorization identifies {habit_count} habit statement(s).",
            "Habit sequence categorization organizes ordered routines.",
        ]
        if args.include_sequences and sequence_count:
            parts.append(f"Detected {sequence_count} sequence(s).")
        if repetition_count:
            parts.append(f"Detected {repetition_count} repetition signal(s).")
        if origin_count:
            parts.append(f"Detected {origin_count} origin context(s).")
        if dendrite_count:
            parts.append(f"Detected {dendrite_count} dendrite context(s).")
        if timeframe_count:
            parts.append(f"Detected {timeframe_count} timeframe context(s).")
        if onset_count:
            parts.append(f"Detected {onset_count} onset context(s).")
        if frequency_count:
            parts.append(f"Detected {frequency_count} frequency context(s).")
        if neural_firing_count:
            parts.append(f"Detected {neural_firing_count} neural firing context(s).")
        if muscle_memory_count:
            parts.append(f"Detected {muscle_memory_count} muscle memory context(s).")
        if pattern_count:
            parts.append(f"Detected {pattern_count} habit pattern context(s).")
        if structure_count:
            parts.append(f"Detected {structure_count} habit structure context(s).")
        if external_force_count:
            parts.append(f"Detected {external_force_count} external force context(s).")
        if influence_count:
            parts.append(f"Detected {influence_count} influence context(s).")
        if external_repetition_count:
            parts.append(f"Detected {external_repetition_count} external repetition context(s).")
        if event_sequence_count:
            parts.append(f"Detected {event_sequence_count} event sequence context(s).")
        if social_learning_count:
            parts.append(f"Detected {social_learning_count} social learning context(s).")
        if provenance_count:
            parts.append(f"Detected {provenance_count} provenance context(s).")
        if activation_count:
            parts.append(f"Detected {activation_count} activation context(s).")
        if dendro_motor_count:
            parts.append(f"Detected {dendro_motor_count} dendro-motor context(s).")
        if music_count:
            parts.append(f"Detected {music_count} music context(s).")
        if intent_count:
            parts.append(f"Detected {intent_count} intent context(s).")
        if action_count:
            parts.append(f"Detected {action_count} action context(s).")
        if word_action_count:
            parts.append(f"Detected {word_action_count} word-action context(s).")
        if social_word_count:
            parts.append(f"Detected {social_word_count} social word context(s).")
        if music_belief_action_count:
            parts.append(
                f"Detected {music_belief_action_count} music-belief-action context(s)."
            )
        if belief_count:
            parts.append(f"Detected {belief_count} belief context(s).")
        if virtue_count:
            parts.append(f"Detected {virtue_count} virtue context(s).")
        if outer_thinking_count:
            parts.append(f"Detected {outer_thinking_count} outer thinking context(s).")
        if inner_thinking_count:
            parts.append(f"Detected {inner_thinking_count} inner thinking context(s).")
        return " ".join(parts)

    def _speech_segments(
        self,
        category_buckets: list[HabitCategoryBucket],
        origin_buckets: list[HabitOriginBucket],
        dendrite_buckets: list[HabitDendriteBucket],
        timeframe_buckets: list[HabitTimeframeBucket],
        onset_buckets: list[HabitOnsetBucket],
        frequency_buckets: list[HabitFrequencyBucket],
        neural_firing_buckets: list[HabitNeuralFiringBucket],
        muscle_memory_buckets: list[HabitMuscleMemoryBucket],
        pattern_buckets: list[HabitPatternBucket],
        structure_buckets: list[HabitStructureBucket],
        external_force_buckets: list[HabitExternalForceBucket],
        influence_buckets: list[HabitInfluenceBucket],
        external_repetition_buckets: list[HabitExternalRepetitionBucket],
        event_sequence_buckets: list[HabitEventSequenceBucket],
        social_learning_buckets: list[HabitSocialLearningBucket],
        provenance_buckets: list[HabitProvenanceBucket],
        activation_buckets: list[HabitActivationBucket],
        dendro_motor_buckets: list[HabitDendroMotorBucket],
        music_buckets: list[HabitMusicBucket],
        intent_buckets: list[HabitIntentBucket],
        action_buckets: list[HabitActionBucket],
        word_action_buckets: list[HabitWordActionBucket],
        social_word_buckets: list[HabitSocialWordBucket],
        music_belief_action_buckets: list[HabitMusicBeliefActionBucket],
        belief_buckets: list[HabitBeliefBucket],
        virtue_buckets: list[HabitVirtueBucket],
        outer_thinking_buckets: list[HabitOuterThinkingBucket],
        inner_thinking_buckets: list[HabitInnerThinkingBucket],
        sequences: list[HabitSequence],
        habits: list[HabitEntry],
        max_category_segments: int,
        max_sequence_segments: int,
        max_origin_segments: int,
        max_dendrite_segments: int,
        max_timeframe_segments: int,
        max_onset_segments: int,
        max_frequency_segments: int,
        max_neural_firing_segments: int,
        max_muscle_memory_segments: int,
        max_pattern_segments: int,
        max_structure_segments: int,
        max_external_force_segments: int,
        max_influence_segments: int,
        max_external_repetition_segments: int,
        max_event_sequence_segments: int,
        max_social_learning_segments: int,
        max_provenance_segments: int,
        max_activation_segments: int,
        max_dendro_motor_segments: int,
        max_music_segments: int,
        max_intent_segments: int,
        max_action_segments: int,
        max_word_action_segments: int,
        max_social_word_segments: int,
        max_music_belief_action_segments: int,
        max_belief_segments: int,
        max_virtue_segments: int,
        max_outer_thinking_segments: int,
        max_inner_thinking_segments: int,
        max_repetition_segments: int,
        max_habit_segments: int,
        max_speech_segments: int,
    ) -> tuple[list[HabitSpeechSegment], bool]:
        segments: list[HabitSpeechSegment] = []

        if max_category_segments and category_buckets:
            for bucket in category_buckets[:max_category_segments]:
                cue = f"Review {bucket.category} habits: {bucket.habit_indices}."
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="category",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_sequence_segments and sequences:
            for sequence in sequences[:max_sequence_segments]:
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="sequence",
                        cue=f"Walk through habit sequence {sequence.index}: {sequence.habit_indices}.",
                        habit_indices=sequence.habit_indices,
                        categories=[],
                        topics=sequence.sequence_markers[:4],
                    )
                )

        if max_origin_segments and origin_buckets:
            for bucket in origin_buckets[:max_origin_segments]:
                cue = (
                    f"Review {bucket.category} origin habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="origin",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_dendrite_segments and dendrite_buckets:
            for bucket in dendrite_buckets[:max_dendrite_segments]:
                cue = (
                    f"Review {bucket.category} dendrite habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="dendrite",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_timeframe_segments and timeframe_buckets:
            for bucket in timeframe_buckets[:max_timeframe_segments]:
                cue = (
                    f"Review {bucket.category} timeframe habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="timeframe",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_onset_segments and onset_buckets:
            for bucket in onset_buckets[:max_onset_segments]:
                cue = (
                    f"Review {bucket.category} onset habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="onset",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_frequency_segments and frequency_buckets:
            for bucket in frequency_buckets[:max_frequency_segments]:
                cue = (
                    f"Review {bucket.category} frequency habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="frequency",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_neural_firing_segments and neural_firing_buckets:
            for bucket in neural_firing_buckets[:max_neural_firing_segments]:
                cue = (
                    f"Review {bucket.category} neural firing habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="neural_firing",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_muscle_memory_segments and muscle_memory_buckets:
            for bucket in muscle_memory_buckets[:max_muscle_memory_segments]:
                cue = (
                    f"Review {bucket.category} muscle memory habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="muscle_memory",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_pattern_segments and pattern_buckets:
            for bucket in pattern_buckets[:max_pattern_segments]:
                cue = (
                    f"Review {bucket.category} habit patterns: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="pattern",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_structure_segments and structure_buckets:
            for bucket in structure_buckets[:max_structure_segments]:
                cue = (
                    f"Review {bucket.category} habit structures: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="structure",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_external_force_segments and external_force_buckets:
            for bucket in external_force_buckets[:max_external_force_segments]:
                cue = (
                    f"Review {bucket.category} external force habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="external_force",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_influence_segments and influence_buckets:
            for bucket in influence_buckets[:max_influence_segments]:
                cue = (
                    f"Review {bucket.category} influence habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="influence",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_external_repetition_segments and external_repetition_buckets:
            for bucket in external_repetition_buckets[:max_external_repetition_segments]:
                cue = (
                    f"Review {bucket.category} external repetition habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="external_repetition",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_event_sequence_segments and event_sequence_buckets:
            for bucket in event_sequence_buckets[:max_event_sequence_segments]:
                cue = (
                    f"Review {bucket.category} event sequence habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="event_sequence",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_social_learning_segments and social_learning_buckets:
            for bucket in social_learning_buckets[:max_social_learning_segments]:
                cue = (
                    f"Review {bucket.category} social learning habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="social_learning",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_provenance_segments and provenance_buckets:
            for bucket in provenance_buckets[:max_provenance_segments]:
                cue = (
                    f"Review {bucket.category} provenance habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="provenance",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_activation_segments and activation_buckets:
            for bucket in activation_buckets[:max_activation_segments]:
                cue = (
                    f"Review {bucket.category} activation habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="activation",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_dendro_motor_segments and dendro_motor_buckets:
            for bucket in dendro_motor_buckets[:max_dendro_motor_segments]:
                cue = (
                    f"Review {bucket.category} dendro-motor habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="dendro_motor",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_music_segments and music_buckets:
            for bucket in music_buckets[:max_music_segments]:
                cue = f"Review {bucket.category} music habits: {bucket.habit_indices}."
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="music",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_intent_segments and intent_buckets:
            for bucket in intent_buckets[:max_intent_segments]:
                cue = f"Review {bucket.category} intent habits: {bucket.habit_indices}."
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="intent",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_action_segments and action_buckets:
            for bucket in action_buckets[:max_action_segments]:
                cue = f"Review {bucket.category} action habits: {bucket.habit_indices}."
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="action",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_word_action_segments and word_action_buckets:
            for bucket in word_action_buckets[:max_word_action_segments]:
                cue = (
                    f"Review {bucket.category} word-action habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="word_action",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_social_word_segments and social_word_buckets:
            for bucket in social_word_buckets[:max_social_word_segments]:
                cue = (
                    f"Review {bucket.category} social word habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="social_word",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_music_belief_action_segments and music_belief_action_buckets:
            for bucket in music_belief_action_buckets[:max_music_belief_action_segments]:
                cue = (
                    "Review "
                    f"{bucket.category} music-belief-action habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="music_belief_action",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_belief_segments and belief_buckets:
            for bucket in belief_buckets[:max_belief_segments]:
                cue = f"Review {bucket.category} belief habits: {bucket.habit_indices}."
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="belief",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_virtue_segments and virtue_buckets:
            for bucket in virtue_buckets[:max_virtue_segments]:
                cue = f"Review {bucket.category} virtue habits: {bucket.habit_indices}."
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="virtue",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_outer_thinking_segments and outer_thinking_buckets:
            for bucket in outer_thinking_buckets[:max_outer_thinking_segments]:
                cue = (
                    f"Review {bucket.category} outer thinking habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="outer_thinking",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_inner_thinking_segments and inner_thinking_buckets:
            for bucket in inner_thinking_buckets[:max_inner_thinking_segments]:
                cue = (
                    f"Review {bucket.category} inner thinking habits: {bucket.habit_indices}."
                )
                if bucket.top_terms:
                    cue = f"{cue} Key terms: {', '.join(bucket.top_terms[:5])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="inner_thinking",
                        cue=cue,
                        habit_indices=bucket.habit_indices,
                        categories=[bucket.category],
                        topics=bucket.top_terms[:5],
                    )
                )

        if max_repetition_segments:
            repeated = [habit for habit in habits if habit.repetition_like]
            repeated.sort(
                key=lambda habit: (-len(habit.repetition_markers), habit.index)
            )
            for habit in repeated[:max_repetition_segments]:
                cue = f"Highlight repetition in habit {habit.index}: {habit.preview}"
                if habit.repetition_markers:
                    cue = f"{cue} Markers: {', '.join(habit.repetition_markers[:4])}."
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="repetition",
                        cue=cue,
                        habit_indices=[habit.index],
                        categories=habit.categories,
                        topics=habit.repetition_markers[:4],
                    )
                )

        if max_habit_segments and habits:
            ranked = sorted(
                habits,
                key=lambda habit: (-len(habit.categories), -len(habit.keywords), habit.index),
            )
            for habit in ranked[:max_habit_segments]:
                segments.append(
                    HabitSpeechSegment(
                        index=len(segments) + 1,
                        kind="habit",
                        cue=f"Highlight habit {habit.index}: {habit.preview}",
                        habit_indices=[habit.index],
                        categories=habit.categories,
                        topics=habit.keywords[:5],
                    )
                )

        truncated = False
        if max_speech_segments and len(segments) > max_speech_segments:
            segments = segments[:max_speech_segments]
            truncated = True

        for idx, segment in enumerate(segments, start=1):
            segment.index = idx
        return segments, truncated

    def _speech_closing(self, args: ContextHabitCategorizationArgs) -> str:
        if not args.include_closing:
            return ""
        return "Close with the dominant habit categories and any key sequences."

    def _preview(self, text: str) -> str:
        limit = self.config.preview_chars
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextHabitCategorizationArgs):
            return ToolCallDisplay(summary="context_habit_categorization")
        utterance_count = len(event.args.utterances or [])
        return ToolCallDisplay(
            summary="context_habit_categorization",
            details={
                "path": event.args.path,
                "utterance_count": utterance_count,
                "include_sequences": event.args.include_sequences,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextHabitCategorizationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")
        return ToolResultDisplay(
            success=True,
            message=(
                f"Categorized {event.result.habit_count} habit(s) into "
                f"{event.result.category_count} category(ies)"
            ),
            warnings=warnings,
            details={
                "habit_count": event.result.habit_count,
                "category_count": event.result.category_count,
                "sequence_count": event.result.sequence_count,
                "repetition_count": event.result.repetition_like_count,
                "origin_count": event.result.origin_count,
                "dendrite_count": event.result.dendrite_like_count,
                "timeframe_count": event.result.timeframe_like_count,
                "onset_count": event.result.onset_like_count,
                "frequency_count": event.result.frequency_like_count,
                "neural_firing_count": event.result.neural_firing_like_count,
                "muscle_memory_count": event.result.muscle_memory_like_count,
                "pattern_count": event.result.pattern_like_count,
                "structure_count": event.result.structure_like_count,
                "external_force_count": event.result.external_force_like_count,
                "influence_count": event.result.influence_like_count,
                "external_repetition_count": event.result.external_repetition_like_count,
                "event_sequence_count": event.result.event_sequence_like_count,
                "social_learning_count": event.result.social_learning_like_count,
                "provenance_count": event.result.provenance_like_count,
                "activation_count": event.result.activation_like_count,
                "dendro_motor_count": event.result.dendro_motor_like_count,
                "music_count": event.result.music_like_count,
                "intent_count": event.result.intent_like_count,
                "action_count": event.result.action_like_count,
                "word_action_count": event.result.word_action_like_count,
                "social_word_count": event.result.social_word_like_count,
                "music_belief_action_count": event.result.music_belief_action_like_count,
                "belief_count": event.result.belief_like_count,
                "virtue_count": event.result.virtue_like_count,
                "outer_thinking_count": event.result.outer_thinking_like_count,
                "inner_thinking_count": event.result.inner_thinking_like_count,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Categorizing habits"
