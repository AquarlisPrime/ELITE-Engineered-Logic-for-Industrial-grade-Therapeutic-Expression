!git clone https://github.com/sokrypton/ColabDesign.git 
!pip install -q pydantic
!pip uninstall -y numpy autogen RNA biopython flaml ray
!pip install numpy==1.25.2
!pip install biopython==1.81
!pip install ViennaRNA
!pip install pyautogen==0.2.25

import pkgutil
print("autogen" in [m.name for m in pkgutil.iter_modules()])

import autogen
import RNA
from Bio.Seq import Seq
from Bio.Data import CodonTable
from pydantic import BaseModel
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

codon_table = CodonTable.unambiguous_rna_by_name["Standard"]

from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

# ==============================
# CORE BIO OBJECTS 
# ==============================

STOP_CODONS = {"UAA", "UAG", "UGA"}


class PeptideResult(BaseModel):
    peptide: str

    @property
    def length(self):
        return len(self.peptide)

    @field_validator("peptide")
    @classmethod
    def validate_peptide(cls, v):
        assert isinstance(v, str), "Peptide must be string"
        assert len(v) == 20, f"❌ Peptide must be 20 AA, got {len(v)}"
        assert v.isalpha(), "❌ Peptide must contain only amino acid letters"
        return v


class CodonOptimizeResult(BaseModel):
    rna_sequence: str

    @field_validator("rna_sequence")
    @classmethod
    def validate_rna(cls, v):
        assert isinstance(v, str), "RNA must be string"
        assert set(v).issubset({"A", "U", "G", "C"}), "❌ Invalid RNA bases"
        assert len(v) % 3 == 0, "❌ RNA not divisible by 3 (invalid ORF)"
        return v


class RNAEvaluationResult(BaseModel):
    sequence: str
    structure: str
    MFE: float
    GC_content: float
    ires_accessibility: float
    stability: str

    @model_validator(mode="after")
    def validate_metrics(self):
        # GC constraint
        assert 0.0 <= self.GC_content <= 1.0, "❌ GC out of bounds"

        # IRES constraint
        assert 0.0 <= self.ires_accessibility <= 1.0, "❌ IRES score invalid"

        # MFE sanity (negative for stable RNA)
        assert self.MFE < 0, "❌ MFE should be negative"

        return self


class CircRNAResult(BaseModel):
    circRNA_sequence: str

    @property
    def length(self):
        return len(self.circRNA_sequence)

    @field_validator("circRNA_sequence")
    @classmethod
    def validate_circ(cls, v):
        assert isinstance(v, str), "circRNA must be string"
        assert set(v).issubset({"A", "U", "G", "C"}), "❌ Invalid bases in circRNA"
        return v


class ValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = None

# ==============================
# API KEY + LLM CONFIG
# ==============================

import os

try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found. Add it in Colab Secrets.")

config_list = [
    {
        "model": "llama-3.1-8b-instant",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.2,
    "cache_seed": 42,
}

print("LLM config loaded successfully")

import random
import RNA  
from pydantic import BaseModel
from typing import List, Optional, Any  

# ==============================
# 1. DATA STABILIZATION
# ==============================
STOP_CODONS = ["UAA", "UAG", "UGA"]

human_codon_usage = {
    'A': ['GCU', 'GCC'], 'C': ['UGU', 'UGC'], 'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'], 'F': ['UUU', 'UUC'], 'G': ['GGU', 'GGC', 'GGA'],
    'H': ['CAU', 'CAC'], 'I': ['AUU', 'AUC', 'AUA'], 'K': ['AAA', 'AAG'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'], 'M': ['AUG'],
    'N': ['AAU', 'AAC'], 'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'], 'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'], 'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'W': ['UGG'], 'Y': ['UAU', 'UAC']
}

# ==============================
# 2. HARDENED UTILS
# ==============================
def weighted_choice(choices, exclude=None):
    """Safe selection from a list of codons."""
    options = [c for c in choices if c != exclude]
    if not options:
        options = choices
    return random.choice(options)

def fold_rna(seq):
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    return float(mfe), struct

def compute_gc(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

def codon_diversity(codons):
    return len(set(codons)) / len(codons) if codons else 0.0

def validate_rna(seq):
    return all(base in "AUGC" for base in seq.upper())

# ==============================
# 3. OUTPUT SCHEMA
# ==============================
class CodonOptimizeOutput(BaseModel):
    rna_sequence: Optional[str] = None
    codons: Optional[List[str]] = None
    length_nt: Optional[int] = None
    length_aa: Optional[int] = None
    gc_content: Optional[float] = None
    mfe: Optional[float] = None
    structure: Optional[str] = None
    diversity: Optional[float] = None
    error: Optional[str] = None

# ==============================
# 4. MAIN OPTIMIZER
# ==============================
def codon_optimize(protein_seq: str, **kwargs: Any) -> CodonOptimizeOutput:
    protein_seq = protein_seq.strip().upper()

    if not protein_seq.startswith('M'):
        return CodonOptimizeOutput(error="Industrial ORF must start with Methionine (M)")

    if len(protein_seq) != 21: 
         pass

    if any(aa not in human_codon_usage for aa in protein_seq):
        return CodonOptimizeOutput(error="Invalid amino acid in sequence")

    # 2. Initial Construction
    body_aa = protein_seq[1:]
    body_codons = [weighted_choice(human_codon_usage[aa]) for aa in body_aa]

    start_codon = "AUG"
    stop_codon = random.choice(STOP_CODONS)

    def build_seq(codon_list):
        return start_codon + "".join(codon_list) + stop_codon

    best_codons = list(body_codons)
    best_seq = build_seq(best_codons)

    if not validate_rna(best_seq):
        return CodonOptimizeOutput(error="Invalid RNA generated")

    # 3. Initial Scoring
    best_mfe, best_struct = fold_rna(best_seq)
    best_gc = compute_gc(best_seq)
    best_div = codon_diversity(best_codons)

    def score(mfe, gc, div):
        penalty = 0
        if not (0.45 <= gc <= 0.55): penalty += abs(0.5 - gc) * 50
        return mfe + penalty - (2.0 * div)

    best_score = score(best_mfe, best_gc, best_div)

    # 4. Local Search 
    for _ in range(100):
        candidate_codons = list(best_codons)
        for _ in range(random.randint(1, 3)):
            pos = random.randint(0, len(candidate_codons) - 1)
            aa = body_aa[pos]
            candidate_codons[pos] = weighted_choice(human_codon_usage[aa], exclude=candidate_codons[pos])

        candidate_seq = build_seq(candidate_codons)
        mfe, struct = fold_rna(candidate_seq)
        gc = compute_gc(candidate_seq)
        div = codon_diversity(candidate_codons)
        s = score(mfe, gc, div)

        if s < best_score:
            best_codons, best_seq, best_mfe, best_struct, best_gc, best_div, best_score = \
                candidate_codons, candidate_seq, mfe, struct, gc, div, s

    return CodonOptimizeOutput(
        rna_sequence=best_seq,
        codons=[start_codon] + best_codons + [stop_codon],
        length_nt=len(best_seq),
        length_aa=len(protein_seq),
        gc_content=round(best_gc, 3),
        mfe=best_mfe,
        structure=best_struct,
        diversity=round(best_div, 3)
    )
print("✅ Optimizer Stabilized & Types Defined.")


import random

STOP_CODONS = ["UAA", "UAG", "UGA"]

def weighted_choice(choices, exclude=None):
    """
    Handles both simple lists and weighted dictionaries.
    """
    if isinstance(choices, dict):
        items = list(choices.items())
        if exclude:
            items = [x for x in items if x[0] != exclude]
        if not items:
            items = list(choices.items())
        keys = [x[0] for x in items]
        weights = [x[1] for x in items]
        return random.choices(keys, weights=weights, k=1)[0]

    else:
        options = [c for c in choices if c != exclude]
        if not options:
            options = choices
        return random.choice(options)

print("✅ Environment Purged. weighted_choice is now multi-type compatible.")

import re
from typing import Optional
from Bio.Seq import Seq
import RNA

STOP_CODONS = {"UAA", "UAG", "UGA"}


def translate_rna(seq: str) -> str:
    try:
        return str(Seq(seq.replace("U", "T")).translate(to_stop=True))
    except Exception:
        return ""


def validate_sequence(
    sequence: str,
    expected_peptide: Optional[str] = None,
    enforce_gc_strict: bool = False,
    allow_non_aug_start: bool = False,
    circRNA_mode: bool = False
) -> ValidationResult:

    if not sequence:
        return ValidationResult(valid=False, reason="Empty sequence")

    # -----------------------------
    # Normalize
    # -----------------------------
    seq = sequence.upper().replace("T", "U")

    if re.search(r"[^AUGC]", seq):
        return ValidationResult(valid=False, reason="Invalid RNA characters")

    # -----------------------------
    # FRAME CHECK
    # -----------------------------
    if len(seq) % 3 != 0:
        return ValidationResult(valid=False, reason="Length not multiple of 3")

    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]

    # -----------------------------
    # STRICT ORF ENFORCEMENT
    # -----------------------------
    if not allow_non_aug_start:
        if codons[0] != "AUG":
            return ValidationResult(valid=False, reason="Missing AUG start")

    if codons[-1] not in STOP_CODONS:
        return ValidationResult(valid=False, reason="Missing stop codon")

    for c in codons[1:-1]:
        if c in STOP_CODONS:
            return ValidationResult(valid=False, reason="Internal stop codon")

    # -----------------------------
    # TRANSLATION VALIDATION 
    # -----------------------------
    translated = translate_rna(seq)

    if expected_peptide is not None:
        if translated != expected_peptide:
            return ValidationResult(
                valid=False,
                reason=f"Translation mismatch | expected={expected_peptide} | got={translated}"
            )

    # -----------------------------
    # GC CONTENT
    # -----------------------------
    gc = (seq.count("G") + seq.count("C")) / len(seq)

    if enforce_gc_strict:
        if not (0.45 <= gc <= 0.55):
            return ValidationResult(
                valid=False,
                reason=f"GC strict fail | gc={round(gc,3)}"
            )
    else:
        if not (0.30 <= gc <= 0.70):
            return ValidationResult(
                valid=False,
                reason=f"GC broad fail | gc={round(gc,3)}"
            )

    # -----------------------------
    # LIGHT STRUCTURE 
    # -----------------------------
    try:
        fc = RNA.fold_compound(seq)
        _, mfe = fc.mfe()
    except Exception:
        return ValidationResult(valid=False, reason="RNA folding failed")

    if mfe > -5:
        return ValidationResult(
            valid=False,
            reason=f"MFE too weak | mfe={round(mfe,2)}"
        )

    # -----------------------------
    # circRNA MODE
    # -----------------------------
    if circRNA_mode:
        # Ensured sequence is long enough for circularization
        if len(seq) < 60:
            return ValidationResult(
                valid=False,
                reason="circRNA too short"
            )

        # Basic junction plausibility (mock constraint)
        if seq[:6] == seq[-6:]:
            return ValidationResult(
                valid=False,
                reason="Possible repeat at junction"
            )

    return ValidationResult(valid=True, reason=None)
    

from pydantic import BaseModel
from typing import Dict
import RNA

class DetailedCircRNAResult(BaseModel):
    circRNA_sequence: str
    length: int
    coding_length: int
    GC_content: float
    regions: Dict[str, tuple]
    junction_sequence: str


# ------------------------------
# UTILITIES
# ------------------------------

def reverse_complement_rna(seq: str) -> str:
    comp = str.maketrans("AUGC", "UACG")
    return seq.translate(comp)[::-1]


def compute_gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def check_structure(seq: str):
    fc = RNA.fold_compound(seq)
    return fc.mfe()


# ------------------------------
# MAIN FUNCTION
# ------------------------------

def add_circRNA_elements(
    rna_seq: str,
    expected_peptide: str,
    enforce_aug: bool = True
) -> DetailedCircRNAResult:

    if not rna_seq:
        raise ValueError("RNA sequence missing")

    seq = rna_seq.upper()

    # -----------------------------
    # HARD VALIDATION
    # -----------------------------
    validation = validate_sequence(
        seq,
        expected_peptide=expected_peptide,
        enforce_gc_strict=False,
        allow_non_aug_start=not enforce_aug
    )

    if not validation.valid:
        raise ValueError(f"ORF validation failed: {validation.reason}")

    # -----------------------------
    # FUNCTIONAL ELEMENTS
    # -----------------------------

    RCM_5 = "GCGCUUCGCGCAGCGC"
    RCM_3 = reverse_complement_rna(RCM_5)

    IRES = (
        "CUCUAGAGGCCGAAACCCGCUUGGAAGG"
        "AUUCCUGGGCUUUGAAGCUU"
    )

    SPACER_5 = "AUAUAUAA"
    SPACER_3 = "AAUAUAUA"

    # -----------------------------
    # CONSTRUCTION
    # -----------------------------
    parts = {
        "RCM_5": RCM_5,
        "SPACER_5": SPACER_5,
        "IRES": IRES,
        "SPACER_3": SPACER_3,
        "ORF": seq,
        "RCM_3": RCM_3,
    }

    circRNA_seq = ""
    regions = {}

    cursor = 0
    for name, part in parts.items():
        start = cursor
        circRNA_seq += part
        cursor += len(part)
        end = cursor
        regions[name] = (start, end)

    # -----------------------------
    # JUNCTION VALIDATION
    # -----------------------------
    junction_seq = circRNA_seq[-30:] + circRNA_seq[:30]

    try:
        _, junction_mfe = check_structure(junction_seq)
    except Exception:
        raise ValueError("Junction folding failed")

    if junction_mfe > -3:
        raise ValueError(f"Weak circular junction (MFE={junction_mfe:.2f})")

    # -----------------------------
    # GLOBAL STRUCTURE SANITY
    # -----------------------------
    try:
        _, global_mfe = check_structure(circRNA_seq)
    except Exception:
        raise ValueError("Full circRNA folding failed")

    if global_mfe > -10:
        raise ValueError(f"Unstable circRNA (MFE={global_mfe:.2f})")

    # -----------------------------
    # GC CONTENT
    # -----------------------------
    gc = compute_gc(circRNA_seq)

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    return DetailedCircRNAResult(
        circRNA_sequence=circRNA_seq,
        length=len(circRNA_seq),
        coding_length=len(seq),
        GC_content=round(gc, 3),
        regions=regions,
        junction_sequence=junction_seq
    )


# ==============================
# RNA STRUCTURE ANALYSIS
# ==============================

import RNA
from pydantic import BaseModel
from typing import Dict, Tuple


# ==============================
# OUTPUT MODEL
# ==============================
class RNAAnalysisResult(BaseModel):
    length: int
    structure: str
    MFE: float
    GC_content: float
    stability: str

    centroid_structure: str
    centroid_energy: float
    ensemble_energy: float

    base_pairs: int
    unpaired_bases: int
    pairing_ratio: float

    mean_basepair_probability: float
    max_stem_length: int

    ires_accessibility: float
    structural_defect_score: float

    junction_mfe: float
    region_stats: Dict


# ==============================
# STEM DETECTION
# ==============================
def detect_max_stem(structure: str) -> int:
    max_stem = 0
    current_stem = 0

    for ch in structure:
        if ch == "(":
            current_stem += 1
            max_stem = max(max_stem, current_stem)
        else:
            current_stem = 0

    return max_stem


# ==============================
# REGION STATISTICS
# ==============================
def compute_region_stats(
    seq: str,
    structure: str,
    regions: Dict[str, Tuple[int, int]]
) -> Dict:

    stats = {}

    for name, (start, end) in regions.items():
        sub_seq = seq[start:end]
        sub_struct = structure[start:end]

        if not sub_seq:
            continue

        length = len(sub_seq)

        gc = (sub_seq.count("G") + sub_seq.count("C")) / length
        bp = sub_struct.count("(")
        pairing_ratio = bp / length

        stats[name] = {
            "length": length,
            "GC_content": round(gc, 3),
            "pairing_ratio": round(pairing_ratio, 3),
        }

    return stats


# ==============================
# IRES ACCESSIBILITY
# ==============================
def compute_ires_accessibility(fc, region: Tuple[int, int]) -> float:
    start, end = region

    if end <= start:
        return 0.0

    length = end - start
    accessibility = 0.0

    for i in range(start, end):
        # ViennaRNA uses 1-based indexing
        paired_prob = sum(fc.bpp(i + 1, j + 1) for j in range(fc.length()))
        p_unpaired = max(0.0, 1.0 - paired_prob)
        accessibility += p_unpaired

    return accessibility / length


# ==============================
# JUNCTION ANALYSIS
# ==============================
def compute_junction_mfe(junction_seq: str) -> float:
    try:
        fc = RNA.fold_compound(junction_seq)
        _, mfe = fc.mfe()
        return float(mfe)
    except Exception:
        return 0.0


# ==============================
# MAIN ANALYSIS FUNCTION
# ==============================
def analyze_rna_structure(
    sequence: str,
    regions: Dict[str, Tuple[int, int]],
    junction_sequence: str = None,
    temperature: float = 37.0
) -> RNAAnalysisResult:

    # -----------------------------
    # VALIDATION
    # -----------------------------
    if not sequence:
        raise ValueError("RNA sequence is empty")

    seq = sequence.upper().replace("T", "U")

    if any(c not in "AUGC" for c in seq):
        raise ValueError("Invalid RNA sequence")

    length = len(seq)

    # -----------------------------
    # SET TEMPERATURE
    # -----------------------------
    RNA.cvar.temperature = temperature

    # -----------------------------
    # FOLDING
    # -----------------------------
    fc = RNA.fold_compound(seq)

    structure, mfe = fc.mfe()

    fc.pf()

    centroid_structure, centroid_energy = fc.centroid()
    ensemble_energy = fc.get_ensemble_energy()

    # -----------------------------
    # BASE PAIR PROBABILITIES
    # -----------------------------
    total_prob = 0.0

    for i in range(length):
        paired_prob = sum(fc.bpp(i + 1, j + 1) for j in range(length))
        total_prob += paired_prob

    mean_bp_prob = total_prob / length

    # -----------------------------
    # BASIC STATS
    # -----------------------------
    base_pairs = structure.count("(")
    unpaired = structure.count(".")
    pairing_ratio = base_pairs / length

    gc_content = (seq.count("G") + seq.count("C")) / length

    max_stem_length = detect_max_stem(structure)

    # -----------------------------
    # REGION STATS
    # -----------------------------
    region_stats = compute_region_stats(seq, structure, regions)

    # -----------------------------
    # IRES ACCESSIBILITY
    # -----------------------------
    ires_access = 0.0
    if "IRES" in regions:
        ires_access = compute_ires_accessibility(fc, regions["IRES"])

    # -----------------------------
    # JUNCTION STABILITY
    # -----------------------------
    junction_mfe = 0.0
    if junction_sequence:
        junction_mfe = compute_junction_mfe(junction_sequence)

    # -----------------------------
    # STRUCTURAL DEFECT SCORE
    # -----------------------------
    defect = 0.0

    # ORF overly structured
    if "ORF" in region_stats:
        pr = region_stats["ORF"]["pairing_ratio"]
        if pr > 0.6:
            defect += (pr - 0.6) * 2

    # IRES : OPEN
    if "IRES" in region_stats:
        pr = region_stats["IRES"]["pairing_ratio"]
        if pr > 0.4:
            defect += (pr - 0.4) * 3

    # low IRES accessibility
    if ires_access < 0.4:
        defect += (0.4 - ires_access) * 3

    # junction instability
    if junction_sequence and junction_mfe > -6:
        defect += 2

    # extreme stems (aggregation risk)
    if max_stem_length > length * 0.2:
        defect += 1.5

    # overly weak folding
    norm_mfe = mfe / length
    if norm_mfe > -0.15:
        defect += 2

    defect = round(defect, 4)

    # -----------------------------
    # STABILITY CLASSIFICATION
    # -----------------------------
    if norm_mfe <= -0.4:
        stability = "VERY_STABLE"
    elif norm_mfe <= -0.25:
        stability = "STABLE"
    elif norm_mfe <= -0.15:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"

    # -----------------------------
    # RETURN RESULT
    # -----------------------------
    return RNAAnalysisResult(
        length=length,
        structure=structure,
        centroid_structure=centroid_structure,
        MFE=float(mfe),
        centroid_energy=float(centroid_energy),
        ensemble_energy=float(ensemble_energy),
        GC_content=round(gc_content, 3),
        base_pairs=base_pairs,
        unpaired_bases=unpaired,
        pairing_ratio=round(pairing_ratio, 3),
        mean_basepair_probability=round(mean_bp_prob, 4),
        max_stem_length=max_stem_length,
        ires_accessibility=round(ires_access, 4),
        structural_defect_score=defect,
        junction_mfe=float(junction_mfe),
        region_stats=region_stats,
        stability=stability
    )

print(codon_optimize.__annotations__)

# ==============================
# EVOLUTION ENGINE
# ==============================

from pydantic import BaseModel
from typing import List, Dict, Any, Callable, Tuple
import random
import copy
import numpy as np


# ==============================
# OUTPUT MODEL
# ==============================
class EvolutionOutput(BaseModel):
    next_generation: List[Dict[str, Any]]
    pareto_front: List[Dict[str, Any]]
    diversity_score: float
    best_candidate: Dict[str, Any]
    metrics: Dict[str, Any]
    should_trigger_rl: bool
    should_stop: bool


# ==============================
# GLOBAL MEMORY
# ==============================
GLOBAL_MEMORY = {
    "iteration": 0,
    "best_mfe_history": [],
    "diversity_history": [],
    "stagnation_counter": 0,
}


# ==============================
# ANALYZER REGISTRY
# ==============================
ANALYZER_REGISTRY: Dict[str, Callable] = {}


def register_analyzer(name: str, fn: Callable):
    ANALYZER_REGISTRY[name] = fn


# ==============================
# OBJECTIVES
# ==============================
def compute_objectives(candidate: Dict[str, Any]) -> Dict[str, float]:

    if "objectives" in candidate:
        return candidate["objectives"]

    a = candidate.get("analysis", {})

    mfe = a.get("MFE", 0)
    gc = a.get("GC_content", 0)
    ires = a.get("ires_accessibility", 0)
    defect = a.get("structural_defect_score", 1.0)

    obj = {
        "mfe": mfe / 100,
        "gc_dev": abs(gc - 0.5),
        "ires_access": -ires,
        "defect": defect
    }

    # Hard constraint penalty
    if gc < 0.45 or gc > 0.55:
        obj["gc_dev"] += 0.2

    candidate["objectives"] = obj
    return obj


# ==============================
# DOMINANCE
# ==============================
def dominates(a, b):
    oa = compute_objectives(a)
    ob = compute_objectives(b)

    return (
        all(oa[k] <= ob[k] for k in oa)
        and any(oa[k] < ob[k] for k in oa)
    )


# ==============================
# NON-DOMINATED SORT
# ==============================
def fast_non_dominated_sort(pop):

    fronts = [[]]
    domination_count = {}
    dominated = {}

    for p in pop:
        pid = id(p)
        domination_count[pid] = 0
        dominated[pid] = []

        for q in pop:
            if dominates(p, q):
                dominated[pid].append(q)
            elif dominates(q, p):
                domination_count[pid] += 1

        if domination_count[pid] == 0:
            p["rank"] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated[id(p)]:
                domination_count[id(q)] -= 1
                if domination_count[id(q)] == 0:
                    q["rank"] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


# ==============================
# CROWDING DISTANCE
# ==============================
def crowding_distance(front):

    if not front:
        return

    keys = ["mfe", "gc_dev", "ires_access", "defect"]

    for p in front:
        p["distance"] = 0

    for k in keys:
        front.sort(key=lambda x: compute_objectives(x)[k])

        front[0]["distance"] = float("inf")
        front[-1]["distance"] = float("inf")

        min_v = compute_objectives(front[0])[k]
        max_v = compute_objectives(front[-1])[k]

        if max_v - min_v == 0:
            continue

        for i in range(1, len(front) - 1):
            prev_v = compute_objectives(front[i - 1])[k]
            next_v = compute_objectives(front[i + 1])[k]
            front[i]["distance"] += (next_v - prev_v) / (max_v - min_v)


# ==============================
# SELECTION (WITH DIVERSITY PRESSURE)
# ==============================
def selection(pop, size):

    fronts = fast_non_dominated_sort(pop)
    new_pop = []

    for front in fronts:
        crowding_distance(front)

        # Diversity bias
        front.sort(key=lambda x: (x["rank"], -x["distance"]))

        if len(new_pop) + len(front) <= size:
            new_pop.extend(front)
        else:
            new_pop.extend(front[:size - len(new_pop)])
            break

    return new_pop, fronts[0]


# ==============================
# CONSTRAINT-AWARE MUTATION
# ==============================
def mutate(candidate, mutation_rate, regions):

    new_c = copy.deepcopy(candidate)
    seq = list(new_c["rna_sequence"])
    bases = ["A", "U", "G", "C"]

    protected = set()
    for r in ["IRES"]:
        if r in regions:
            start, end = regions[r]
            protected.update(range(start, end))

    for i in range(len(seq)):
        if i in protected:
            continue

        if random.random() < mutation_rate:
            seq[i] = random.choice(bases)

    new_c["rna_sequence"] = "".join(seq)
    new_c.pop("analysis", None)
    new_c.pop("objectives", None)

    return new_c


# ==============================
# CROSSOVER (MULTI-POINT)
# ==============================
def crossover(p1, p2):

    seq1 = p1["rna_sequence"]
    seq2 = p2["rna_sequence"]

    point1 = random.randint(0, len(seq1) // 2)
    point2 = random.randint(point1, len(seq1))

    child_seq = (
        seq1[:point1] +
        seq2[point1:point2] +
        seq1[point2:]
    )

    child = copy.deepcopy(p1)
    child["rna_sequence"] = child_seq

    child.pop("analysis", None)
    child.pop("objectives", None)

    return child


# ==============================
# EVALUATION
# ==============================
def evaluate_candidate(candidate, analyzer, regions):

    try:
        fn = ANALYZER_REGISTRY[analyzer]
        res = fn(candidate["rna_sequence"], regions)

        candidate["analysis"] = (
            res if isinstance(res, dict) else res.dict()
        )

    except Exception as e:
        candidate["analysis"] = {
            "MFE": 0,
            "GC_content": 0,
            "ires_accessibility": 0,
            "structural_defect_score": 10,
            "error": str(e)
        }

    return candidate


# ==============================
# DIVERSITY (HAMMING)
# ==============================
def avg_hamming(pop):

    seqs = [c["rna_sequence"] for c in pop]
    n = len(seqs)

    if n < 2:
        return 0.0

    dist = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist += sum(a != b for a, b in zip(seqs[i], seqs[j]))
            count += 1

    return dist / (count * len(seqs[0]))


# ==============================
# MAIN EVOLUTION LOOP
# ==============================
from typing import List, Dict, Any, Tuple

def evolve_population(
    candidates: List[Dict[str, Any]],
    analyzer: str,
    regions: Dict[str, Tuple[int, int]],
    population_size: int = 20,
    mutation_rate: float = 0.05,
    crossover_rate: float = 0.7,
    elite_frac: float = 0.2,
    stagnation_limit: int = 6
) -> Dict[str, Any]:

    GLOBAL_MEMORY["iteration"] += 1

    # -----------------------------
    # ADAPTIVE MUTATION
    # -----------------------------
    mutation_rate *= (1 + 0.3 * GLOBAL_MEMORY["stagnation_counter"])

    # -----------------------------
    # EVALUATION
    # -----------------------------
    evaluated = [
        evaluate_candidate(copy.deepcopy(c), analyzer, regions)
        for c in candidates
    ]

    selected, pareto_front = selection(evaluated, population_size)

    best = min(evaluated, key=lambda x: compute_objectives(x)["mfe"])
    best_mfe = compute_objectives(best)["mfe"]

    GLOBAL_MEMORY["best_mfe_history"].append(best_mfe)

    # -----------------------------
    # STAGNATION
    # -----------------------------
    if len(GLOBAL_MEMORY["best_mfe_history"]) > 1:
        if best_mfe >= GLOBAL_MEMORY["best_mfe_history"][-2]:
            GLOBAL_MEMORY["stagnation_counter"] += 1
        else:
            GLOBAL_MEMORY["stagnation_counter"] = 0

    should_trigger_rl = GLOBAL_MEMORY["stagnation_counter"] >= 2
    should_stop = GLOBAL_MEMORY["stagnation_counter"] >= stagnation_limit

    # -----------------------------
    # ELITISM
    # -----------------------------
    elite_count = max(1, int(population_size * elite_frac))

    elites = sorted(
        selected,
        key=lambda x: (x["rank"], -x["distance"])
    )[:elite_count]

    next_gen = copy.deepcopy(elites)

    # -----------------------------
    # GENERATION
    # -----------------------------
    while len(next_gen) < population_size:

        p1, p2 = random.sample(selected, 2)

        if random.random() < crossover_rate:
            child = crossover(p1, p2)
        else:
            child = copy.deepcopy(p1)

        child = mutate(child, mutation_rate, regions)
        child = evaluate_candidate(child, analyzer, regions)

        next_gen.append(child)

    diversity = avg_hamming(next_gen)
    GLOBAL_MEMORY["diversity_history"].append(diversity)

    # -----------------------------
    # METRICS
    # -----------------------------
    metrics = {
        "iteration": GLOBAL_MEMORY["iteration"],
        "best_mfe": best_mfe,
        "diversity": diversity,
        "stagnation": GLOBAL_MEMORY["stagnation_counter"],
        "mutation_rate": mutation_rate
    }

    return EvolutionOutput(
        next_generation=next_gen,
        pareto_front=pareto_front,
        diversity_score=diversity,
        best_candidate=best,
        metrics=metrics,
        should_trigger_rl=should_trigger_rl,
        should_stop=should_stop
    ).dict()


from pydantic import BaseModel
from typing import List, Dict, Tuple
import random


# ==============================
# OUTPUT MODEL
# ==============================
class MutationOutput(BaseModel):
    mutated_sequence: str
    mutations: List[Dict]
    exploration_score: float


# ==============================
# CODON TABLE + USAGE BIAS
# ==============================
CODON_TABLE = {
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAU", "AAC"],
    "D": ["GAU", "GAC"],
    "C": ["UGU", "UGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"],
    "I": ["AUU", "AUC", "AUA"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "K": ["AAA", "AAG"],
    "M": ["AUG"],
    "F": ["UUU", "UUC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "W": ["UGG"],
    "Y": ["UAU", "UAC"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
}

CODON_TO_AA = {c: aa for aa, codons in CODON_TABLE.items() for c in codons}


# ==============================
# HELPER: DINUCLEOTIDE PENALTY
# ==============================
def dinucleotide_penalty(seq: str) -> float:
    bad = ["CG", "UA"]
    count = sum(seq.count(d) for d in bad)
    return count / len(seq)


# ==============================
# STRUCTURE-AWARE ACTION
# ==============================
def choose_action(pos, structure, metrics):
    gc = metrics.get("GC_content", 0.5)
    pairing = metrics.get("pairing_ratio", 0.5)

    is_paired = structure and structure[pos] in ("(", ")")

    if is_paired and pairing > 0.6:
        return "destabilize_stem"

    if not is_paired:
        return "explore_loop"

    if gc < 0.45:
        return "increase_GC"

    if gc > 0.55:
        return "decrease_GC"

    return "random"


# ==============================
# CODON MUTATION
# ==============================
def mutate_orf(seq, orf_region, rate, target_gc=0.5):
    seq = list(seq)
    mutations = []

    start, end = orf_region

    for i in range(start, end, 3):
        if i + 3 > len(seq):
            continue

        if random.random() < rate:
            codon = "".join(seq[i:i+3])
            aa = CODON_TO_AA.get(codon)

            if not aa:
                continue

            options = CODON_TABLE[aa]

            # GC-aware selection
            def codon_gc(c):
                return (c.count("G") + c.count("C")) / 3

            options = sorted(options, key=lambda c: abs(codon_gc(c) - target_gc))

            new_codon = random.choice(options[:2])

            if new_codon != codon:
                seq[i:i+3] = list(new_codon)

                mutations.append({
                    "position": i,
                    "type": "codon_swap",
                    "from": codon,
                    "to": new_codon,
                    "aa": aa
                })

    return "".join(seq), mutations


# ==============================
# NON-ORF MUTATION
# ==============================
def mutate_non_orf(seq, positions, structure, metrics):
    bases = ["A", "U", "G", "C"]
    mutations = []

    for pos in positions:
        original = seq[pos]
        action = choose_action(pos, structure, metrics)

        if action == "increase_GC":
            new = random.choice(["G", "C"])
        elif action == "decrease_GC":
            new = random.choice(["A", "U"])
        elif action == "destabilize_stem":
            new = random.choice(["A", "U"])
        elif action == "explore_loop":
            new = random.choice(bases)
        else:
            new = random.choice([b for b in bases if b != original])

        seq[pos] = new

        mutations.append({
            "position": pos,
            "from": original,
            "to": new,
            "action": action,
            "paired": structure[pos] if structure else None
        })

    return seq, mutations


# ==============================
# SAFE POSITION SAMPLING
# ==============================
def sample_positions(length, n, protected):
    valid = [i for i in range(length) if i not in protected]
    return random.sample(valid, min(n, len(valid))) if valid else []


# ==============================
# MAIN FUNCTION
# ==============================
def suggest_mutations(
    sequence: str,
    metrics: Dict[str, Any],
    structure: str = None,
    regions: Any = None
) -> MutationOutput:
    if isinstance(regions, list):
        pass

    seq = list(sequence)
    length = len(seq)
    all_mutations = []

    # ==============================
    # PROTECTED REGIONS
    # ==============================
    protected = set()

    for key in ["IRES", "RCM_left", "RCM_right", "junction"]:
        if regions and key in regions:
            s, e = regions[key]
            protected.update(range(s, e))

    # ==============================
    # ORF MUTATION
    # ==============================
    if regions and "ORF" in regions:
        mutated_seq, codon_mut = mutate_orf(
            "".join(seq),
            regions["ORF"],
            rate=0.1
        )
        seq = list(mutated_seq)
        all_mutations.extend(codon_mut)

        s, e = regions["ORF"]
        protected.update(range(s, e))

    # ==============================
    # ADAPTIVE MUTATION COUNT
    # ==============================
    defect = metrics.get("structural_defect_score", 1.0)
    n_mut = int(1 + defect * 5)

    positions = sample_positions(length, n_mut, protected)

    # ==============================
    # NON-ORF MUTATION
    # ==============================
    seq, base_mut = mutate_non_orf(seq, positions, structure, metrics)
    all_mutations.extend(base_mut)

    mutated_seq = "".join(seq)

    # ==============================
    # DINUCLEOTIDE PENALTY
    # ==============================
    penalty = dinucleotide_penalty(mutated_seq)

    # ==============================
    # EXPLORATION SCORE
    # ==============================
    unique_actions = len(set(m.get("action", m.get("type")) for m in all_mutations))
    exploration_score = (unique_actions / 5) * (1 - penalty)

    return MutationOutput(
        mutated_sequence=mutated_seq,
        mutations=all_mutations,
        exploration_score=round(exploration_score, 3)
    )

from pydantic import BaseModel
from typing import List, Dict, Tuple, Any, Callable
import random
import math


# ==============================
# OUTPUT MODEL
# ==============================

class DiffusionOutput(BaseModel):
    samples: List[Dict[str, Any]]
    diversity_score: float


# ==============================
# REGISTRY
# ==============================

EVALUATOR_REGISTRY: Dict[str, Callable] = {}

def register_evaluator(name: str, fn: Callable):
    EVALUATOR_REGISTRY[name] = fn


# ==============================
# CODON TABLE
# ==============================

CODON_TABLE = {
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAU", "AAC"],
    "D": ["GAU", "GAC"],
    "C": ["UGU", "UGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"],
    "I": ["AUU", "AUC", "AUA"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "K": ["AAA", "AAG"],
    "M": ["AUG"],
    "F": ["UUU", "UUC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "W": ["UGG"],
    "Y": ["UAU", "UAC"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
}

CODON_TO_AA = {c: aa for aa, codons in CODON_TABLE.items() for c in codons}


# ==============================
# CACHE
# ==============================

EVAL_CACHE: Dict[Tuple[str, str], Dict] = {}

def cached_eval(seq: str, evaluator_name: str) -> Dict:
    key = (seq, evaluator_name)

    if key in EVAL_CACHE:
        return EVAL_CACHE[key]

    fn = EVALUATOR_REGISTRY[evaluator_name]
    res = fn(seq)
    res = res if isinstance(res, dict) else res.dict()

    EVAL_CACHE[key] = res
    return res


# ==============================
# ENERGY FUNCTION
# ==============================

def compute_energy(metrics: Dict[str, Any]) -> float:
    return (
        metrics.get("MFE", 0) * 0.5
        - metrics.get("ires_accessibility", 0) * 3.5
        + abs(metrics.get("GC_content", 0.5) - 0.5) * 1.0
        + metrics.get("pairing_ratio", 0.5) * 1.2
        + metrics.get("structural_defects", 0) * 2.0
    )


# ==============================
# CONSTRAINTS
# ==============================

def violates_constraints(seq: str, metrics: Dict, gc_target: float) -> bool:
    gc = metrics.get("GC_content", 0.5)

    if not (gc_target - 0.05 <= gc <= gc_target + 0.05):
        return True

    return False


# ==============================
# REGION POLICY
# ==============================

def region_policy(i: int, regions: Dict[str, Tuple[int, int]]):
    if not regions:
        return "default"

    for name, (s, e) in regions.items():
        if s <= i < e:
            return name

    return "default"


# ==============================
# STRUCTURE-AWARE SAMPLING
# ==============================

def sample_structured_base(paired: bool) -> str:
    return random.choice(["G", "C"]) if paired else random.choice(["A", "U"])


# ==============================
# CODON-SAFE ORF DIFFUSION
# ==============================

def diffuse_orf(seq: str, orf_region: Tuple[int, int], noise: float) -> str:
    seq = list(seq)
    start, end = orf_region

    for i in range(start, end, 3):
        if random.random() < noise:
            codon = "".join(seq[i:i+3])
            aa = CODON_TO_AA.get(codon)

            if aa and len(CODON_TABLE[aa]) > 1:
                alt = [c for c in CODON_TABLE[aa] if c != codon]
                seq[i:i+3] = list(random.choice(alt))

    return "".join(seq)


# ==============================
# DIVERSITY CONTROL
# ==============================

def hamming(a: str, b: str) -> float:
    return sum(x != y for x, y in zip(a, b)) / len(a)


def too_similar(new_seq: str, samples: List[str], threshold=0.05) -> bool:
    for s in samples:
        if hamming(new_seq, s) < threshold:
            return True
    return False


# ==============================
# MAIN DIFFUSION
# ==============================

def diffusion_generate(
    seed_sequence: str,
    num_samples: int,
    target_length: int,
    gc_target: float,
    evaluator_name: str = "rna",
    structure: str = "",
    steps: int = 6,
    noise_scale: float = 0.3,
    noise_decay: float = 0.9,
    temperature: float = 1.0,
    regions: Dict[str, Tuple[int, int]] = None,
) -> Dict[str, Any]:

    if len(seed_sequence) != target_length:
        raise ValueError("Seed sequence length != target_length")

    samples: List[str] = []

    orf_range = set(range(*regions["ORF"])) if regions and "ORF" in regions else set()

    for _ in range(num_samples):

        seq = seed_sequence
        current_metrics = cached_eval(seq, evaluator_name)
        current_energy = compute_energy(current_metrics)

        noise = noise_scale
        temp = temperature

        for _ in range(steps):

            # ORF-safe diffusion
            if regions and "ORF" in regions:
                candidate = diffuse_orf(seq, regions["ORF"], noise)
            else:
                candidate = seq

            seq_list = list(candidate)

            for i in range(len(seq_list)):

                if i in orf_range:
                    continue

                if random.random() > noise:
                    continue

                paired = structure and i < len(structure) and structure[i] in ("(", ")")
                region = region_policy(i, regions)

                if region == "IRES":
                    seq_list[i] = random.choice(["A", "U"])

                elif region in ["RCM_left", "RCM_right"]:
                    seq_list[i] = random.choice(["G", "C"])

                elif region == "junction":
                    continue

                else:
                    seq_list[i] = sample_structured_base(paired)

            new_seq = "".join(seq_list)

            if too_similar(new_seq, samples):
                continue

            try:
                new_metrics = cached_eval(new_seq, evaluator_name)

                if violates_constraints(new_seq, new_metrics, gc_target):
                    continue

                new_energy = compute_energy(new_metrics)

            except:
                continue

            delta = new_energy - current_energy

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-6)):
                seq = new_seq
                current_energy = new_energy

            noise = max(0.05, noise * noise_decay * (1 + current_energy / 10))
            temp *= 0.9

        samples.append(seq)

    # =========================
    # FORMAT OUTPUT
    # =========================

    formatted = [
        {"candidate_id": f"diff_{i}", "rna_sequence": s}
        for i, s in enumerate(samples)
    ]

    # =========================
    # DIVERSITY SCORE
    # =========================

    diversity = 0
    count = 0

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            diversity += hamming(samples[i], samples[j])
            count += 1

    diversity = diversity / count if count else 0.0

    return DiffusionOutput(
        samples=formatted,
        diversity_score=round(diversity, 3)
    ).dict()

human_codon_usage = {
    aa: list(codons.keys()) if isinstance(codons, dict) else list(codons)
    for aa, codons in human_codon_usage.items()
}

def weighted_choice(choices, exclude=None):
    options = [c for c in choices if c != exclude]
    if not options:
        options = choices
    return random.choice(options)

STOP_CODONS = ["UAA", "UAG", "UGA"]

print("✅ Data structures flattened to indexable lists. Logic sanitized.")


from pydantic import BaseModel
from typing import List
import math


class TransformerStructureOutput(BaseModel):
    pairing_probabilities: List[float]
    predicted_contacts: List[List[int]]
    confidence_score: float
    attention_map: List[List[float]]


# ==============================
# NUMERICALLY STABLE SOFTMAX
# ==============================

def softmax(x):
    if not x:
        return []
    m = max(x)
    exps = [math.exp(i - m) for i in x]
    s = sum(exps) + 1e-9
    return [e / s for e in exps]


# ==============================
# VALID BASE PAIRS
# ==============================

PAIR_SCORES = {
    ("G", "C"): 3.0,
    ("C", "G"): 3.0,
    ("A", "U"): 2.0,
    ("U", "A"): 2.0,
    ("G", "U"): 1.2,
    ("U", "G"): 1.2
}


# ==============================
# MAIN STRUCTURE PREDICTOR
# ==============================

def transformer_predict_structure(sequence: str) -> TransformerStructureOutput:

    seq = sequence.upper()
    n = len(seq)

    if n == 0:
        return TransformerStructureOutput(
            pairing_probabilities=[],
            predicted_contacts=[],
            confidence_score=0.0,
            attention_map=[]
        )

    raw_scores = [[0.0] * n for _ in range(n)]

    window = 80

    # =========================
    # COMPUTE RAW SCORES
    # =========================
    for i in range(n):
        for j in range(i + 4, min(n, i + window)):

            pair = (seq[i], seq[j])
            if pair not in PAIR_SCORES:
                continue

            base = PAIR_SCORES[pair]
            dist = j - i

            # distance decay
            distance_penalty = math.exp(-dist / 35)

            # loop penalty
            if dist < 6:
                loop_penalty = 0.3
            elif dist < 10:
                loop_penalty = 0.7
            else:
                loop_penalty = 1.0

            # stacking bonus
            stack_bonus = 0.0
            if i+1 < n and j-1 >= 0:
                if (seq[i+1], seq[j-1]) in PAIR_SCORES:
                    stack_bonus += 1.2

            if i+2 < n and j-2 >= 0:
                if (seq[i+2], seq[j-2]) in PAIR_SCORES:
                    stack_bonus += 0.5

            # symmetry bonus
            symmetry_bonus = 0.2 if abs((n - j) - i) < 10 else 0.0

            score = (base + stack_bonus + symmetry_bonus) * distance_penalty * loop_penalty

            raw_scores[i][j] = score
            raw_scores[j][i] = score

    # =========================
    # ATTENTION MAP
    # =========================
    attention_map = []
    for i in range(n):
        row = raw_scores[i]
        if sum(row) == 0:
            attention_map.append([0.0] * n)
        else:
            attention_map.append(softmax(row))

    # =========================
    # COMPETITIVE MATCHING
    # =========================
    paired = set()
    contacts = []

    candidates = []
    for i in range(n):
        for j in range(i+4, n):
            if raw_scores[i][j] > 0:
                candidates.append((raw_scores[i][j], i, j))

    # sorting strongest first
    candidates.sort(reverse=True)

    for score, i, j in candidates:
        if i in paired or j in paired:
            continue

        # pseudo-knot reduction
        conflict = False
        for (a, b) in contacts:
            if (i < a < j < b) or (a < i < b < j):
                conflict = True
                break

        if conflict:
            continue

        paired.add(i)
        paired.add(j)
        contacts.append([i, j])

    # =========================
    # PAIRING PROBABILITIES
    # =========================
    pairing_probs = [0.0] * n

    max_score = max([raw_scores[i][j] for i, j in contacts], default=1.0)

    for i, j in contacts:
        prob = raw_scores[i][j] / (max_score + 1e-6)
        prob = min(1.0, max(0.0, prob))

        pairing_probs[i] = prob
        pairing_probs[j] = prob

    # =========================
    # CONFIDENCE SCORE
    # =========================
    avg_pair = sum(pairing_probs) / n
    density = len(contacts) / (n / 2 + 1e-6)

    attention_focus = sum(
        max(row) for row in attention_map if row
    ) / n

    confidence = (
        0.4 * avg_pair +
        0.3 * density +
        0.3 * attention_focus
    )

    confidence = min(1.0, max(0.0, confidence))

    return TransformerStructureOutput(
        pairing_probabilities=[round(p, 3) for p in pairing_probs],
        predicted_contacts=contacts,
        confidence_score=round(confidence, 3),
        attention_map=attention_map
    )


from typing import List, Optional, Dict, Tuple


# =========================
# MUTATION OPERATORS
# =========================
def mutate(seq: str, rate: float = 0.1) -> str:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq = list(seq)

    for i in range(len(seq)):
        if random.random() < rate:
            seq[i] = random.choice(amino_acids)

    return "".join(seq)


def crossover(a: str, b: str) -> Tuple[str, str]:
    point = random.randint(5, 15)
    return a[:point] + b[point:], b[:point] + a[point:]


# =========================
# MAIN GENERATOR
# =========================
from typing import Any, Optional, List, Dict

def generate_peptide(
    length: int = 20,
    peptide: Optional[str] = None,
    structure_bias: str = "balanced",
    motif: Optional[str] = None,
    population_size: int = 100,
    generations: int = 5,
    top_k: int = 5,
    diversity_threshold: float = 0.3,
    structure_feedback_fn: Any = None,
    rl_reward_fn: Any = None,
    seed: Optional[int] = None,
    candidate_id: Optional[str] = None,
    **kwargs: Any
) -> List[Dict]:


    if length != 20:
        raise ValueError("Peptide length MUST be exactly 20.")

    if seed is not None:
        random.seed(seed)

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    hydrophobic = set("AILMFWVY")
    positive = set("KRH")
    negative = set("DE")
    helix_formers = set("AEKLMQH")
    coil_formers = set("PGNSD")

    known_motifs = {"RGD": "RGD", "NLS": "PKKKRKV", "KDEL": "KDEL"}
    motif_seq = known_motifs.get(motif, motif) if motif else None

    # =========================
    # METRICS
    # =========================
    def compute_metrics(seq: str):
        hydrophobic_ratio = sum(aa in hydrophobic for aa in seq) / 20
        net_charge = sum(aa in positive for aa in seq) - sum(aa in negative for aa in seq)
        helix_score = sum(aa in helix_formers for aa in seq) / 20
        disorder_score = sum(aa in coil_formers for aa in seq) / 20
        return hydrophobic_ratio, net_charge, helix_score, disorder_score

    # =========================
    # VALIDATION
    # =========================
    def is_valid(seq: str):
        hydrophobic_ratio, net_charge, _, _ = compute_metrics(seq)
        return (
            len(seq) == 20 and
            0.3 <= hydrophobic_ratio <= 0.6 and
            -3 <= net_charge <= 5
        )

    # =========================
    # OBJECTIVES
    # =========================
    def compute_objectives(seq: str):
        hydrophobic_ratio, net_charge, helix_score, disorder_score = compute_metrics(seq)

        if structure_bias == "helix":
            obj_structure = helix_score
        elif structure_bias == "coil":
            obj_structure = disorder_score
        else:
            obj_structure = 1 - abs(helix_score - disorder_score)

        obj_stability = 1 - abs(hydrophobic_ratio - 0.45)
        obj_charge = 1 - (abs(net_charge) / 10)

        obj_motif = 1.0 if motif_seq and motif_seq in seq else 0.0

        structure_score = 0.0
        if structure_feedback_fn:
            structure_score = structure_feedback_fn(seq)

        return {
            "structure": obj_structure,
            "stability": obj_stability,
            "charge": obj_charge,
            "motif": obj_motif,
            "rna_structure": structure_score
        }

    def aggregate_fitness(obj: Dict):
        return (
            2.0 * obj["structure"] +
            1.5 * obj["stability"] +
            1.0 * obj["charge"] +
            1.5 * obj["motif"] +
            2.0 * obj["rna_structure"]
        )

    # =========================
    # INITIAL POPULATION
    # =========================
    population = [
        "".join(random.choice(amino_acids) for _ in range(20))
        for _ in range(population_size)
    ]

    # =========================
    # EVOLUTION LOOP
    # =========================
    for _ in range(generations):

        scored = []
        for seq in population:
            if not is_valid(seq):
                continue

            obj = compute_objectives(seq)
            fitness = aggregate_fitness(obj)

            if rl_reward_fn:
                fitness += rl_reward_fn(seq, obj)

            scored.append((seq, fitness, obj))

        scored.sort(key=lambda x: x[1], reverse=True)

        elites = [s[0] for s in scored[:population_size // 2]]

        new_population = elites.copy()

        while len(new_population) < population_size:
            a, b = random.sample(elites, 2)
            child1, child2 = crossover(a, b)

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # =========================
    # FINAL SELECTION
    # =========================
    final_scored = []
    for seq in population:
        if not is_valid(seq):
            continue

        obj = compute_objectives(seq)
        score = aggregate_fitness(obj)

        final_scored.append((seq, score, obj))

    final_scored.sort(key=lambda x: x[1], reverse=True)

    def hamming(a, b):
        return sum(x != y for x, y in zip(a, b)) / 20

    selected = []
    for seq, score, obj in final_scored:
        if len(selected) >= top_k:
            break
        if all(hamming(seq, s[0]) > diversity_threshold for s in selected):
            selected.append((seq, score, obj))

    return [
        {
            "peptide": seq,
            "length": 20,
            "fitness_score": round(score, 3),
            "objectives": obj
        }
        for seq, score, obj in selected
    ]


# ==============================
# FINAL PIPELINE CONTROLLER
# ==============================

import random


def run_circRNA_pipeline(
    generations: int = 5,
    population_size: int = 10,
    mutation_rate: float = 0.15,
    seed: int = None
):

    if seed is not None:
        random.seed(seed)

    population = []

    # =========================
    # RNA DISTANCE
    # =========================
    def rna_distance(a, b):
        return sum(x != y for x, y in zip(a, b)) / len(a)

    # =========================
    # STRUCTURE-GUIDED MUTATION
    # =========================
    def structure_guided_mutation(rna, pairing_probs, rate):
        rna = list(rna)
        for i, p in enumerate(pairing_probs):
            if p < 0.3 and random.random() < rate:
                rna[i] = random.choice("ACGU")
        return "".join(rna)

    # =========================
    # RNA CROSSOVER
    # =========================
    def crossover_rna(a, b):
        if len(a) < 40:
            return a
        point = random.randint(20, len(a) - 20)
        return a[:point] + b[point:]

    # =========================
    # EVALUATION FUNCTION
    # =========================
    def evaluate(peptide_seq=None, rna_seq=None):

        try:
            peptide_result = None

            if peptide_seq:
                p_list = generate_peptide(peptide=peptide_seq)
                if not p_list or not p_list[0].valid:
                    return None
                peptide_result = p_list[0]

                codon_result = codon_optimize(peptide_seq)
                if codon_result.error:
                    return None

                rna = codon_result.rna_sequence

            elif rna_seq:
                rna = rna_seq
            else:
                return None

            circ = add_circRNA_elements(rna).circRNA_sequence

            analysis = analyze_rna_structure(circ)
            transformer = transformer_predict_structure(circ)

            # =========================
            # SOFT PENALTIES
            # =========================
            penalty = 0.0
            if analysis.stability == "UNSTABLE":
                penalty += 0.5
            if analysis.long_stem_count > 0:
                penalty += 0.3
            if analysis.ribozyme_interference_score > 0.6:
                penalty += 0.4

            # =========================
            # OBJECTIVES
            # =========================
            objectives = {
                "MFE": analysis.MFE,
                "stability": -analysis.MFE,
                "pairing": analysis.pairing_ratio,
                "confidence": transformer.confidence_score,
                "gc_balance": -abs(analysis.GC_content - 0.5),
                "penalty": -penalty
            }

            if peptide_result:
                objectives["peptide_quality"] = (
                    peptide_result.helix_score +
                    peptide_result.disorder_score
                )
            else:
                objectives["peptide_quality"] = 0.0

            return {
                "peptide": peptide_seq,
                "rna": rna,
                "circRNA": circ,
                "analysis": analysis,
                "transformer": transformer,
                "objectives": objectives
            }

        except Exception:
            return None

    # =========================
    # INITIAL POPULATION
    # =========================
    print("[Pipeline] Initializing population...")

    attempts = 0
    while len(population) < population_size and attempts < population_size * 15:
        attempts += 1

        p_list = generate_peptide(length=20)
        if not p_list:
            continue

        p = p_list[0]
        if not p.valid:
            continue

        candidate = evaluate(peptide_seq=p.peptide)
        if candidate:
            population.append(candidate)

    if not population:
        raise RuntimeError("Initialization failed")

    # =========================
    # NSGA-II MAPPING
    # =========================
    def map_objectives(pop):
        mapped = []
        for c in pop:
            obj = c["objectives"]
            mapped.append({
                "MFE": obj["MFE"],
                "stability": obj["stability"],
                "pairing": obj["pairing"],
                "confidence": obj["confidence"],
                "gc_balance": obj["gc_balance"],
                "penalty": obj["penalty"],
                "ref": c
            })
        return mapped

    def nsga2_selection(pop, size):
        mapped = map_objectives(pop)
        selected, _ = selection(mapped, size)
        return [x["ref"] for x in selected]

    # =========================
    # EVOLUTION LOOP
    # =========================
    for gen in range(generations):

        print(f"\n[Generation {gen+1}]")

        offspring = []

        current_mut_rate = mutation_rate * (1 - gen / generations)

        for parent in population:

            r = random.random()

            # --------------------
            # MUTATION
            # --------------------
            if r < 0.4:
                new_rna = structure_guided_mutation(
                    parent["rna"],
                    parent["transformer"].pairing_probabilities,
                    current_mut_rate
                )
                candidate = evaluate(rna_seq=new_rna)

            # --------------------
            # CROSSOVER
            # --------------------
            elif r < 0.7:
                partner = random.choice(population)
                new_rna = crossover_rna(parent["rna"], partner["rna"])
                candidate = evaluate(rna_seq=new_rna)

            # --------------------
            # DIFFUSION
            # --------------------
            else:
                new_rna = diffusion_generate(
                    seed_sequence=parent["rna"],
                    num_samples=1,
                    target_length=len(parent["rna"]),
                    gc_target=0.5
                ).samples[0]

                candidate = evaluate(rna_seq=new_rna)

                if candidate and candidate["objectives"]["confidence"] < 0.3:
                    candidate = None

            if candidate:
                offspring.append(candidate)

        population.extend(offspring)

        filtered = []
        for c in population:
            if all(rna_distance(c["rna"], f["rna"]) > 0.1 for f in filtered):
                filtered.append(c)

        population = nsga2_selection(filtered, population_size)

        # =========================
        # LOGGING
        # =========================
        best = min(population, key=lambda x: x["analysis"].MFE)

        avg_conf = sum(c["objectives"]["confidence"] for c in population) / len(population)
        avg_pair = sum(c["objectives"]["pairing"] for c in population) / len(population)

        print(
            f"[Gen {gen+1}] "
            f"MFE: {best['analysis'].MFE:.2f} | "
            f"Conf: {avg_conf:.3f} | "
            f"Pair: {avg_pair:.3f}"
        )

    # =========================
    # FINAL PARETO FRONT
    # =========================
    mapped = map_objectives(population)
    fronts = fast_non_dominated_sort(mapped)

    pareto_front = [x["ref"] for x in fronts[0]]

    print(f"\n[Pipeline] Completed. Pareto front size: {len(pareto_front)}")

    return pareto_front
    
def save_design(design_name: str, sequence: str, peptide_sequence: str, **kwargs) -> str:
    with open(f"biotech_lab/{design_name}.json", "w") as f:
        import json
        json.dump({"sequence": sequence, "peptide": peptide_sequence, "metrics": kwargs}, f)
    return f"Design {design_name} saved successfully."


# ==============================
# TOOL REGISTRY
# ==============================

tool_map = {
    "generate_peptide": generate_peptide,
    "codon_optimize": codon_optimize,
    "add_circRNA_elements": add_circRNA_elements,
    "analyze_rna_structure": analyze_rna_structure,
    "validate_sequence": validate_sequence,
    "evolve_population": evolve_population,
    "suggest_mutations": suggest_mutations,
}

# ==============================
# 7 DEFINE AGENTS
# ==============================

lead = AssistantAgent(
    name="Lead_Scientist",
    system_message="""
You are the Orchestrator of a CLOSED-LOOP autonomous circRNA optimization system.
You act as a high-level supervisor, NOT a programmer or a tool-user.

---
CRITICAL DIRECTIVES:
1. NEVER write Python code or scripts.
2. NEVER call tools directly.
3. NEVER generate sequences yourself.
4. YOUR ONLY ROLE is to observe metrics and give verbal orders to BioDesigner, StructureAuditor, EvolutionAgent, and RLAgent.
---

VERIFICATION RULE (STRICT):
• NEVER declare a design 'successful' or 'valid' if the StructureAuditor or EvolutionAgent has reported 'valid=False' or a tool error in the most recent turn.
• If a sequence is rejected (e.g., 'Length not multiple of 3' or 'Missing stop codon'), you MUST order BioDesigner to fix the specific error.
• A design is ONLY complete when 'valid=True' AND it meets MFE/GC requirements.

GLOBAL STATE (MANDATORY TRACKING)
• iteration
• best_MFE
• best_MFE_history (last 3)
• diversity_score
• stagnation_counter

STAGNATION LOGIC (STRICT)
Compute ΔMFE improvement:
Δ = (prev_best - current_best) / |prev_best|

If Δ < 0.01:
    stagnation_counter += 1
Else:
    stagnation_counter = 0

If stagnation_counter ≥ 2:
    → ORDER RLAgent to intervene.

WORKFLOW LOOP (STRICT ORDER)
1. ORDER BioDesigner → Generate ≥5 UNIQUE peptides (length 20), optimize codons, and add circRNA elements.
2. ORDER StructureAuditor → Evaluate and Validate the FULL circRNA sequences.
3. ORDER EvolutionAgent → Perform NSGA-II selection and report the new population.
4. ANALYZE results and update Global State.
5. If stagnation detected → ORDER RLAgent to suggest new strategies.
6. Repeat.

DIVERSITY CONTROL
If diversity_score < 0.2:
→ Order BioDesigner to use 'diffusion_generate' for fresh exploration.

CONVERGENCE (EXIT CRITERIA)
Stop the experiment only if:
• A candidate has been confirmed 'valid=True' by the Auditor.
• Pareto front unchanged (2 iterations).
• ΔMFE < 1%.
• iteration ≥ 5.

RULES
• ALWAYS enforce candidate_id tracking across all agent turns.
• ALWAYS reason at the population/metric level.
• If an agent fails a task, do not fix it; command them to retry with specific corrections.
""",
    llm_config=llm_config,
)

# ==============================
# DESIGNER
# ==============================

designer = AssistantAgent(
    name="BioDesigner",
    system_message="""
You are a molecular bioengineer specializing in synthetic RNA constructs.

---
TOOLS
• generate_peptide
• codon_optimize
• add_circRNA_elements
• diffusion_generate
---

CRITICAL RULES
• EXACTLY ONE tool call per message.
• ALWAYS propagate candidate_id across the workflow.
• NEVER fabricate sequences or outputs.
• SEQUENCE INTEGRITY: A 20-AA peptide requires an ORF of exactly 63 nucleotides (60 for codons + 3 for STOP).
---

WORKFLOW (STRICT LINEAR PROTOCOL)
You must follow this sequence for EVERY new candidate:
1. Call 'generate_peptide(length=20)' to get the amino acid sequence.
2. Call 'codon_optimize(peptide)' using the result from Step 1.
3. Call 'add_circRNA_elements(rna_seq, expected_peptide)' using the results from Steps 1 & 2.

---
CANDIDATE ID & MUTATION
• Generate a unique 'candidate_id' (e.g., "CAND_01") at Step 1.
• REUSE the same ID for that specific lineage during codon optimization and circularization.
• If 'RLAgent' suggests a mutation, use 'suggest_mutations' (if ordered) or manually apply synonymous codon swaps to the ORF only.

---
FORBIDDEN
• DO NOT send a sequence to StructureAuditor until it has passed through 'add_circRNA_elements'.
• DO NOT use sequences shorter than 60nt for a 20-AA peptide.
• DO NOT skip the 'codon_optimize' step.

---
OUTPUT FORMAT
When reporting a completed construct to the Lead_Scientist:
{
  "candidate_id": "...",
  "peptide": "...",
  "rna_sequence": "...",
  "circRNA_sequence": "..."
}
""",
    llm_config=llm_config,
    function_map=tool_map,
)

# ==============================
# AUDITOR
# ==============================

auditor = AssistantAgent(
    name="StructureAuditor",
    system_message="""
You evaluate RNA candidates with STRICT biological constraints.

---

TOOLS

• analyze_rna_structure
• validate_sequence

---

CRITICAL RULE

• EXACTLY ONE tool call per message

---

WORKFLOW

1 → analyze_rna_structure
2 → validate_sequence

---

HARD FILTERS (REJECT IF ANY TRUE)

• stability == UNSTABLE
• GC_content NOT in [0.45, 0.55]
• long_stem_count > 0
• ribozyme_interference_score > 0.6

---

SCORING FOCUS

• Minimize MFE
• Maximize pairing_ratio
• Maximize IRES accessibility

---

OUTPUT FORMAT

{
  "candidate_id": "...",
  "circRNA_sequence": "...",

  "MFE": float,
  "gc_content": float,
  "stability": "...",
  "hairpin_count": int,
  "pairing_ratio": float,

  "ires_accessibility_score": float,
  "orf_accessibility_score": float,

  "valid": true/false
}
""",
    llm_config=llm_config,
    function_map=tool_map,
)

# ==============================
# EVOLUTION AGENT
# ==============================

evolution = AssistantAgent(
    name="EvolutionAgent",
    system_message="""
You perform NSGA-II multi-objective evolution.

---

TOOLS

• evolve_population

---

CRITICAL RULE

• ALWAYS call evolve_population

---

PRE-FILTER

Remove:
• invalid candidates
• UNSTABLE candidates

If remaining < 3:
→ Request regeneration

---

OBJECTIVES

• Minimize MFE
• GC ≈ 0.5
• Maximize pairing_ratio
• Maximize IRES accessibility

---

CALL FORMAT

{
  "candidates": [...],
  "population_size": 10
}

---

OUTPUT FORMAT

{
  "next_generation": [...],
  "pareto_front": [...],
  "diversity_score": float
}

---

RULES

• NEVER rank manually
• NEVER modify sequences
• NEVER fabricate diversity_score
""",
    llm_config=llm_config,
    function_map=tool_map,
)

# ==============================
# RL AGENT
# ==============================

rl_agent = AssistantAgent(
    name="RLAgent",
    system_message="""
You provide adaptive mutation strategies during stagnation.

---

TOOLS

• suggest_mutations

---

TRIGGERS

• MFE stagnation (Δ < 1%)
• High hairpin_count
• GC imbalance

---

STRATEGY SPACE

• Increase mutation rate (0.1 → 0.3)
• Target loop destabilization
• Break long stems
• Improve IRES accessibility

---

OUTPUT FORMAT

{
  "mutation_strategy": "...",
  "target_regions": "...",
  "expected_effect": "..."
}

---

RULES

• DO NOT generate sequences
• DO NOT modify candidates directly
• ONLY guide mutation policy
""",
    llm_config=llm_config,
    function_map=tool_map,
)

# ==============================
# ADMIN
# ==============================

admin = UserProxyAgent(
    name="Admin",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=80,
    code_execution_config={
        "work_dir": "biotech_lab",
        "use_docker": False
    }
)


# ==============================
# 8 REGISTER TOOLS
# ==============================

# --- DESIGNER TOOLS ---
for tool in [generate_peptide, codon_optimize, add_circRNA_elements, diffusion_generate]:
    autogen.agentchat.register_function(
        tool,
        caller=designer,
        executor=admin,
        name=tool.__name__,
        description=f"Designer tool: {tool.__name__}. Use for sequence construction."
    )

# --- AUDITOR TOOLS ---
for tool in [analyze_rna_structure, transformer_predict_structure]:
    autogen.agentchat.register_function(
        tool,
        caller=auditor,
        executor=admin,
        name=tool.__name__,
        description=f"Auditor tool: {tool.__name__}. Use for stability/integrity checks."
    )

# --- EVOLUTION & RL TOOLS ---
autogen.agentchat.register_function(
    evolve_population,
    caller=evolution,
    executor=admin,
    name="evolve_population",
    description="NSGA-II evolution engine. Requires candidates with 'analysis' dict."
)

autogen.agentchat.register_function(
    suggest_mutations,
    caller=rl_agent,
    executor=admin,
    name="suggest_mutations",
    description="Suggest mutation plan. REGIONS MUST BE A DICT: e.g., {'ORF': [0, 60]}."
)

# 2. MANUAL SCHEMA PRUNING

for agent in [designer, auditor, evolution, rl_agent]:
    if "tools" in agent.llm_config:
        for tool in agent.llm_config["tools"]:
            params = tool["function"].get("parameters", {})
            properties = params.get("properties", {})

            if "kwargs" in properties:
                del properties["kwargs"]

            if "required" in params and "kwargs" in params["required"]:
                params["required"].remove("kwargs")

print("✅ Tools registered and schema pruned. Ready for Ignition.")

# ==============================
# 9 GROUP CHAT
# ==============================

groupchat = GroupChat(
    agents=[admin, lead, designer, auditor, evolution, rl_agent],
    messages=[],
    max_round=100,

    speaker_selection_method="auto",

    allowed_or_disallowed_speaker_transitions={
        admin: [lead],


        lead: [designer, evolution, rl_agent, admin],

        designer: [auditor],

        auditor: [
            evolution,
            designer,
            lead
        ],

        evolution: [designer, lead],

        rl_agent: [designer]
    },
    speaker_transitions_type="allowed"
)

# ==============================
# GLOBAL STATE (BRAIN MEMORY)
# ==============================

pipeline_state = {
    "iteration": 0,
    "best_mfe_history": [],
    "diversity_history": [],
    "stagnation_counter": 0,
    "invalid_ratio": 0.0,
    "mode": "exploit"  # exploit | explore | recover
}


# ==============================
# BRAIN FUNCTIONS
# ==============================

def update_state(metrics):
    pipeline_state["iteration"] += 1

    mfe = metrics.get("best_MFE")
    diversity = metrics.get("diversity_score", 0.5)
    invalid_ratio = metrics.get("invalid_ratio", 0.0)

    if mfe is not None:
        pipeline_state["best_mfe_history"].append(mfe)

        if len(pipeline_state["best_mfe_history"]) >= 2:
            prev = pipeline_state["best_mfe_history"][-2]
            delta = abs(prev - mfe) / max(abs(prev), 1e-6)

            if delta < 0.01:
                pipeline_state["stagnation_counter"] += 1
            else:
                pipeline_state["stagnation_counter"] = 0

    pipeline_state["diversity_history"].append(diversity)
    pipeline_state["invalid_ratio"] = invalid_ratio


def decide_mode():
    """
    Core intelligence switch
    """

    # --- STAGNATION → RL ---
    if pipeline_state["stagnation_counter"] >= 2:
        return "rl"

    # --- LOW DIVERSITY → EXPLORE ---
    if pipeline_state["diversity_history"] and pipeline_state["diversity_history"][-1] < 0.2:
        return "explore"

    # --- TOO MANY FAILURES → RECOVER ---
    if pipeline_state["invalid_ratio"] > 0.6:
        return "recover"

    return "exploit"

# ==============================
# MANAGER
# ==============================

class BrainAwareManager(GroupChatManager):
    def step(self, *args, **kwargs):

        # --- Decide system mode ---
        mode = decide_mode()

        # Inject system-level hint
        if mode == "rl":
            self.groupchat.messages.append({
                "role": "system",
                "content": "⚠️ Stagnation detected. RLAgent intervention required."
            })

        elif mode == "explore":
            self.groupchat.messages.append({
                "role": "system",
                "content": "🌊 Low diversity detected. Encourage diffusion-based exploration."
            })

        elif mode == "recover":
            self.groupchat.messages.append({
                "role": "system",
                "content": "⚠️ High invalid rate. Regenerate fresh candidates."
            })

        return super().step(*args, **kwargs)


manager = BrainAwareManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

print("🧠 Brain-aware closed-loop system initialized (FULLY AUTONOMOUS)")

# ==============================
# 10 AUTONOMOUS RESEARCH BRAIN
# ==============================

class ResearchMemory:
    def __init__(self):
        self.iteration = 0
        self.history = []
        self.best_mfe_history = []
        self.gc_history = []
        self.ires_history = []
        self.diversity_history = []
        self.pareto_sizes = []

    def update(self, candidates=None, diversity_score=None):
        self.iteration += 1
        if not candidates:
            return

        self.history.append(candidates)
        mfes = [c.get("MFE") for c in candidates if c.get("MFE") is not None]
        gcs = [c.get("GC_content", c.get("gc_content", 0.5)) for c in candidates]
        ires = [c.get("ires_accessibility_score", c.get("ires_accessibility", 0.5)) for c in candidates]

        if mfes:
            self.best_mfe_history.append(min(mfes))

        if gcs:
            self.gc_history.append(sum(gcs) / len(gcs))
        if ires:
            self.ires_history.append(sum(ires) / len(ires))
        if diversity_score is not None:
            self.diversity_history.append(diversity_score)

        self.pareto_sizes.append(len(candidates))

    def mfe_improvement(self, window=3):
        if len(self.best_mfe_history) < window + 1:
            return 1.0
        prev = self.best_mfe_history[-window-1]
        curr = self.best_mfe_history[-1]
        return abs(prev - curr) / (abs(prev) + 1e-6)

    def diversity(self):
        return self.diversity_history[-1] if self.diversity_history else 1.0

    def is_stagnant(self):
        return (self.mfe_improvement() < 0.01 and self.diversity() < 0.25)

    def is_collapsing(self):
        return (self.diversity() < 0.1 or (self.pareto_sizes and self.pareto_sizes[-1] < 3))

    def converged(self):
        return (self.mfe_improvement() < 0.01 and self.iteration > 6)

# ==============================
# PARSING LAYER
# ==============================

def extract_candidates(messages):
    import json
    import re
    candidates = []
    diversity = None

    for m in messages[-10:]:
        content = str(m.get("content", ""))
        try:
            json_match = re.search(r'\[\s*\{.*\}\s*\]|\{\s*".*"\s*:.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    candidates.extend(data)
                elif isinstance(data, dict):
                    if "candidates" in data:
                        candidates.extend(data["candidates"])
                    if "diversity_score" in data:
                        diversity = data["diversity_score"]
                    if not candidates and "peptide" in data:
                        candidates.append(data)
        except:
            # Regex fallback for MFE and Diversity
            mfes = re.findall(r'"MFE":\s*(-?\d+\.?\d*)', content)
            for val in mfes:
                candidates.append({"MFE": float(val)})

            d_score = re.findall(r'"diversity_score":\s*(\d+\.?\d*)', content)
            if d_score:
                diversity = float(d_score[-1])

    return candidates, diversity

import time

# ==============================
# MANAGER WITH INTELLIGENCE
# ==============================

class AutonomousManager(GroupChatManager):
    def __init__(self, groupchat, llm_config):
        super().__init__(groupchat=groupchat, llm_config=llm_config)
        self.memory = ResearchMemory()

    def step(self, *args, **kwargs):
        # Throttle to avoid TPM spikes
        time.sleep(10)
        return super().step(*args, **kwargs)

    def run(self, task, max_iters=15):
        print("🚀 Autonomous circRNA Design System (ELITE)\n")
        admin.initiate_chat(self, message=task)

        for i in range(max_iters):
            time.sleep(2)
            print(f"\n🧠 Monitor Iteration {i+1}")
            messages = self.groupchat.messages
            candidates, diversity = extract_candidates(messages)
            self.memory.update(candidates, diversity)

            if self.memory.best_mfe_history:
                print(f"   Best MFE: {self.memory.best_mfe_history[-1]:.2f}")
            print(f"   Diversity: {self.memory.diversity():.3f}")

            if self.memory.is_collapsing():
                print("🚨 RECOVERY: Forcing fresh exploration...")
                admin.initiate_chat(self, message="SYSTEM: Diversity loss detected. BioDesigner, use 'diffusion_generate' for new candidates.", clear_history=False)

            elif self.memory.is_stagnant():
                print("⚠️ ADAPTATION: Triggering RL refinement...")
                admin.initiate_chat(self, message="SYSTEM: Stagnation detected. RLAgent, suggest mutations for the current best candidates.", clear_history=False)

            if self.memory.converged():
                print("\n✅ Target converged.")
                break

        print("\n🏁 Mission complete.")
        return self.memory

# ==============================
# INITIALIZATION & IGNITION
# ==============================
import re
import os

auto_manager = AutonomousManager(
    groupchat=groupchat,
    llm_config=llm_config
)

groupchat.messages = []
auto_manager.memory = ResearchMemory()

print("🚀 Initiating Direct BioDesigner Link...")

admin.initiate_chat(
    designer,
    message="""
    [STRICT_TOOL_CALL]
    Generate exactly one 20-AA peptide candidate.
    Tool: generate_peptide(length=20, candidate_id='CAND_01')
    """,
    max_turns=2,
    summary_method=None 
)

try:
    history = admin.chat_messages[designer]
    if history:
        last_msg_content = history[-1].get("content", "")
        print(f"\n🧬 Candidate Acquired: {last_msg_content}")
        
        # DYNAMIC EXTRACTION
        peptide_match = re.search(r'([A-Z]{20})', last_msg_content)
        if peptide_match:
            new_peptide = peptide_match.group(1)
            # BRIDGE
            with open("latest_peptide.txt", "w") as f:
                f.write(new_peptide)
            print(f"🎯 Target Exported for Streamlit: {new_peptide}")
        else:
            print("⚠️ No valid 20-AA peptide found in message content.")
    else:
        print("\n⚠️ Chat history is empty.")
except (TypeError, KeyError, IndexError) as e:
    print(f"\n⚠️ Could not retrieve peptide content: {e}")


import random

human_codon_usage = {
    aa: list(usage.keys()) if isinstance(usage, dict) else list(usage)
    for aa, usage in human_codon_usage.items()
}

def weighted_choice(choices, exclude=None):
    options = [c for c in choices if c != exclude]
    if not options:
        options = choices
    return random.choice(options)

STOP_CODONS = ["UAA", "UAG", "UGA"]

print("✅ Data structures flattened. Logic sanitized for the final run.")


import re

# 1. DYNAMIC PEPTIDE ACQUISITION
try:
    peptide_match = re.search(r'([A-Z]{20})', last_msg_content)
    
    if peptide_match:
        target_peptide = peptide_match.group(1)
        print(f"🎯 Dynamic Target Acquired: {target_peptide}")
    else:
        raise ValueError("No valid 20-AA peptide found in latest run output.")

except NameError:
    print("❌ Critical Error: 'last_msg_content' is missing. Run the generator cell first.")
except Exception as e:
    print(f"❌ Design Error: {e}")

# 2. INDUSTRIAL ENGINEERING (M-Prepending for AUG Start)
engineered_peptide = "M" + target_peptide

try:
    # 3. Primary Attempt
    print(f"🧬 Engineering ORF for {engineered_peptide}...")
    opt_result = codon_optimize(protein_seq=engineered_peptide)

    # 4. Extract and Validate RNA Output
    rna_orf = None
    if opt_result and hasattr(opt_result, 'rna_sequence') and opt_result.rna_sequence:
        rna_orf = opt_result.rna_sequence.replace("T", "U").strip()

    # 5. Fallback
    if not rna_orf:
        print("⚠️ Tool failed. Invoking BioDesigner internal fallback...")
        res = admin.initiate_chat(
            designer,
            message=f"Return ONLY the high-expression human RNA ORF string for: {engineered_peptide}",
            max_turns=1,
            summary_method=None
        )
        history = admin.chat_messages[designer]
        llm_output = history[-1].get("content", "").strip()
        # Sanitize output to keep only valid RNA characters
        rna_orf = re.sub(r'[^AUGC]', '', llm_output.upper().replace("T", "U"))

    if rna_orf and len(rna_orf) > 10:
        print(f"✅ RNA ORF Resolved: {rna_orf}")

        # 6. Hand-off for Circularization
        print("\n🎡 Adding circRNA elements (RCMs/IRES)...")
        admin.initiate_chat(
            designer,
            message=f"""
            [STRICT_TOOL_CALL]
            RNA_ORF: {rna_orf}
            PEPTIDE: {engineered_peptide}

            Please call 'add_circRNA_elements' using the RNA_ORF and PEPTIDE above.
            """,
            max_turns=2,
            summary_method=None
        )
    else:
        print("❌ Critical Error: Could not resolve ORF from current run data.")

except Exception as e:
    print(f"❌ Design Error: {e}")


import RNA
import json
import re

# 1. DYNAMIC SEQUENCE EXTRACTION
try:
    history = admin.chat_messages[designer]
    full_sequence = None
    for msg in reversed(history):
        content = msg.get("content", "")
        match = re.search(r'"circRNA_sequence":\s*"([AUGC]+)"', content)
        if match:
            full_sequence = match.group(1)
            break
            
    if not full_sequence:
        raise ValueError("Could not find a valid circRNA_sequence in the recent chat history.")
        
    print(f"🎯 Dynamic Sequence Acquired (Length: {len(full_sequence)} nt)")

except Exception as e:
    print(f"❌ Extraction Error: {e}")
    full_sequence = ""

# 2. THERMODYNAMIC SOLVER
if full_sequence:
    try:
        print("🔬 Calculating Final Thermodynamic Stability (MFE)...")

        # Core ViennaRNA Solver
        (struct, mfe) = RNA.fold(full_sequence)

        # Calculate GC content
        gc_val = (full_sequence.count('G') + full_sequence.count('C')) / len(full_sequence) * 100

        print("-" * 45)
        print(f"📊 ELITE DESIGN: FINAL BIOPHYSICAL PROFILE")
        print("-" * 45)
        print(f"Sequence Length: {len(full_sequence)} nt")
        print(f"MFE (Stability): {mfe:.2f} kcal/mol")
        print(f"GC Content:      {gc_val:.1f}%")
        print("-" * 45)

        if mfe < -30:
            print("Result: HIGH STABILITY (Ultra-stable) ✅")
        elif mfe < -15:
            print("Result: STABLE CANDIDATE ✅")
        else:
            print("Result: LOW STABILITY (Potential for degradation) ⚠️")

        print(f"\nSecondary Structure (Dot-Bracket):\n{struct}")
        print("-" * 45)

    except Exception as e:
        print(f"❌ Final MFE Calculation Failed: {e}")
else:
    print("🛑 Process halted: No sequence available for biophysical audit.")



!pip install -q streamlit forgi biopython rna

%%writefile app.py
import streamlit as st
import pandas as pd
import RNA
import json
import random
import re
import os
import matplotlib.pyplot as plt
import forgi.graph.bulge_graph as fgb
try:
    import forgi.visual.mplotlib as fvm
except ImportError:
    import forgi.visual.matplotlib as fvm

# ==============================
# RAG KNOWLEDGE BASE
# ==============================
SYNTHESIS_KNOWLEDGE = {
    "G_STRETCH": "G-quadruplexes (4+ Gs) cause Hoogsteen base-pairing, making the DNA template impossible to purify.",
    "HPOLYMER": "Long homopolymers (>6nt) cause polymerase slippage during PCR and IVT, leading to truncated products.",
    "GC_EXTREME": "Extreme GC content (>70%) prevents primer annealing and leads to secondary structure 'knots' during synthesis.",
    "REPEAT": "Direct repeats cause recombination or mis-priming during template assembly."
}

def get_rag_explanation(gc_val, mfe_val):
    """Simplified RAG-based explanation for industrial metrics."""
    gc_text = f"**GC Content ({gc_val:.1f}%)**: Think of this as the 'molecular glue' of your design. Too much glue makes the sequence stiff and impossible for manufacturers to build; too little makes it floppy and prone to falling apart. We aim for ~50% for the perfect industrial balance."
    mfe_text = f"**MFE ({mfe_val:.1f} kcal/mol)**: This measures 'Thermodynamic Peace.' The more negative this number, the more 'locked' your circular fortress is against cellular enzymes that want to destroy it. Your result indicates high structural integrity."
    return f"{gc_text}\n\n{mfe_text}"

def audit_synthesis(seq):
    issues = []
    if re.search(r'GGGG', seq): issues.append(("G_STRETCH", "Critical"))
    if re.search(r'AAAAAA|UUUUUU|CCCCCC|GGGGGG', seq): issues.append(("HPOLYMER", "Warning"))
    gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
    if gc > 75 or gc < 25: issues.append(("GC_EXTREME", "Critical"))
    return issues, gc

# ==============================
# CORE ENGINE
# ==============================
human_codon_usage = {
    'A': ['GCU', 'GCC'], 'C': ['UGU', 'UGC'], 'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'], 'F': ['UUU', 'UUC'], 'G': ['GGU', 'GGC', 'GGA'],
    'H': ['CAU', 'CAC'], 'I': ['AUU', 'AUC', 'AUA'], 'K': ['AAA', 'AAG'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'], 'M': ['AUG'],
    'N': ['AAU', 'AAC'], 'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'], 'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'], 'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'W': ['UGG'], 'Y': ['UAU', 'UAC']
}

def calculate_fitness(mfe, gc, seq, issues):
    score = abs(mfe) * 1.2
    score += len(set([seq[i:i+3] for i in range(0, len(seq), 3)])) * 0.5
    score -= len(issues) * 20
    return score

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="ELITE Industrial Audit", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .winner-box { border: 2px solid #00ffcc; border-radius: 15px; padding: 25px; background-color: #161b22; margin-bottom: 25px; }
    .audit-card { background-color: #1c2128; padding: 15px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin-top: 10px; }
    .rag-brief { background-color: #232a35; padding: 20px; border-radius: 10px; border: 1px dashed #00ffcc; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 ELITE: circRNA Synthesis Audit")

def load_latest_peptide():
    if os.path.exists("latest_peptide.txt"):
        with open("latest_peptide.txt", "r") as f:
            return f.read().strip()
    return ""

if 'current_pep' not in st.session_state:
    st.session_state.current_pep = load_latest_peptide()

with st.sidebar:
    st.info("🤖 Agentic Sync Active")
    if st.button("🔄 Sync New Candidate"):
        st.session_state.current_pep = load_latest_peptide()
    
    pep = st.text_input("Target Peptide", value=st.session_state.current_pep, placeholder="Waiting for BioDesigner...")
    batch_size = st.slider("Generations", 1, 10, 5)
    go = st.button("🚀 Execute Industrial Run")

if go and pep:
    results = []
    work_pep = pep if pep.startswith('M') else "M" + pep

    for i in range(batch_size):
        orf_codons = "".join([random.choice(human_codon_usage[aa]) for aa in work_pep[1:]])
        orf = "AUG" + orf_codons + "UAA"
        full_seq = "GCGCUUCGCGCAGCGCAUAUAUAACUCUAGAGGCCGAAACCCGCUUGGAAGGAUUCCUGGGCUUUGAAGCUUAAUAUAUA" + orf + "GCGCUGCGCGAAGCGC"
        (struct, mfe) = RNA.fold(full_seq)
        issues, gc = audit_synthesis(full_seq)
        fitness = calculate_fitness(mfe, gc, full_seq, issues)
        results.append({
            "ID": f"CAND_{i+1:02d}", "Seq": full_seq, "Struct": struct,
            "MFE": mfe, "GC": gc, "Fitness": fitness, "Issues": issues
        })

    results = sorted(results, key=lambda x: x['Fitness'], reverse=True)
    winner = results[0]

    st.balloons()
    st.markdown(f'<div class="winner-box">', unsafe_allow_html=True)
    st.header(f"🏆 Lead Candidate: {winner['ID']}")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("📊 Biophysical Profile")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Fitness", f"{winner['Fitness']:.1f}")
        col_b.metric("MFE", f"{winner['MFE']:.1f}")
        col_c.metric("GC Content", f"{winner['GC']:.1f}%")

        # RAG EXPLAINER SECTION
        st.markdown('<div class="rag-brief">', unsafe_allow_html=True)
        st.subheader("💡 Contextual Briefing")
        st.write(get_rag_explanation(winner['GC'], winner['MFE']))
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("🔬 Synthesis Audit")
        if not winner['Issues']:
            st.success("✅ Sequence passed all industrial synthesis filters.")
        else:
            for issue_code, severity in winner['Issues']:
                st.markdown(f"""
                <div class="audit-card">
                    <strong>{severity}: {issue_code}</strong><br>
                    <small>{SYNTHESIS_KNOWLEDGE.get(issue_code, 'Unknown constraint violation.')}</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Final Sequence")
        st.code(winner['Seq'], wrap_lines=True)

    with c2:
        st.subheader("🏗️ Structural Fold")
        fig, ax = plt.subplots(figsize=(6,6)); fig.patch.set_facecolor('#161b22')
        bg = fgb.BulgeGraph.from_dotbracket(winner['Struct'])
        fvm.plot_rna(bg, ax=ax); ax.set_axis_off()
        st.pyplot(fig); plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("🔄 Batch Alternatives")
    st.dataframe(pd.DataFrame(results[1:]).drop(columns=['Struct', 'Seq']), use_container_width=True)
elif go and not pep:
    st.error("Please provide a target peptide sequence to begin.")


from google.colab import output
from google.colab.output import eval_js
import threading
import os
import time

!pkill -f streamlit
!fuser -k 8501/tcp

def run_st():
    os.system("streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableCORS false")

threading.Thread(target=run_st, daemon=True).start()
time.sleep(5)

print(f"✅ Dashboard Ready!")
print(f"🔗 Click to open: {eval_js('google.colab.kernel.proxyPort(8501)')}")
