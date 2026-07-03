"""
Zoetis Veterinary Product Classifier — Streamlit UI
Colours: Purple #4B286D · Teal #00A591 · White
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, json, os, time, math, io, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv, set_key
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zoetis Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (Zoetis brand)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,400;0,600;0,700;0,800;1,800&display=swap');
  :root {
    --orange:  #E8590C;
    --orange2: #FF6B1A;
    --dark:    #1A1A1A;
    --grey:    #555555;
    --light:   #F9F9F9;
    --white:   #FFFFFF;
    --border:  #E0E0E0;
    --teal:    #00A591;
  }

  html, body, [class*="css"], .stMarkdown, p, li, label {
    font-family: 'Nunito', sans-serif !important;
    color: var(--dark);
  }
  /* Apply Nunito to spans but NOT Material Symbols icon spans */
  span:not(.material-symbols-rounded):not([class*="material-symbol"]):not([data-testid*="Icon"]) {
    font-family: 'Nunito', sans-serif !important;
    color: var(--dark);
  }
  /* Preserve Material Symbols font for expander icons */
  .material-symbols-rounded,
  [class*="material-symbol"],
  [data-testid="stExpanderToggleIcon"] span {
    font-family: 'Material Symbols Rounded' !important;
    color: inherit;
  }

  /* ── Main background ── */
  .stApp { background: var(--white) !important; }
  [data-testid="stAppViewContainer"] > .main { background: var(--white) !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--dark) !important;
    border-right: 3px solid var(--orange);
  }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #F0F0F0 !important; font-family: 'Nunito', sans-serif !important; }

  [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
  [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div { color: #111 !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] [role="option"] { color: #111 !important; }
  [data-testid="stSidebar"] hr { border-color: #444 !important; }

  /* ── Headings ── */
  h1, h2, h3 { font-family: 'Nunito', sans-serif !important; color: var(--dark) !important; font-weight: 800 !important; }
  h2 { border-bottom: 3px solid var(--orange); padding-bottom: 6px; display: inline-block; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--light); border-radius: 10px;
    display: flex; width: 100%;
    border: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    flex: 1; text-align: center; justify-content: center;
    font-size: 0.9rem; font-weight: 700;
    font-family: 'Nunito', sans-serif !important;
    color: var(--grey) !important;
  }
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--orange) !important; color: white !important;
    border-radius: 8px;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: white; border-left: 5px solid var(--orange);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(232,89,12,.10);
  }
  .metric-card h4 { margin: 0 0 4px; color: var(--grey); font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; }
  .metric-card p  { margin: 0; font-size: 1.5rem; font-weight: 800; color: var(--orange); }

  /* ── Review row ── */
  .review-row {
    background: #FFF4EE; border-left: 4px solid var(--orange);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
  }

  /* ── Source badges ── */
  .badge-regex    { background:#00A591; color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }
  .badge-master   { background:#333;    color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }
  .badge-svm      { background:#E8590C; color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }
  .badge-claude   { background:#E84E1B; color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }
  .badge-gpt      { background:#10A37F; color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }
  .badge-gemini   { background:#4285F4; color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }
  .badge-llama    { background:#7C3AED; color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }
  .badge-unknown  { background:#999;    color:white; padding:2px 9px; border-radius:12px; font-size:.75rem; font-weight:700; }

  /* ── Expanders ── */
  div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
  }

  /* ── Buttons ── */
  .stButton > button {
    background: var(--orange) !important; color: white !important;
    border: none; border-radius: 8px; font-weight: 700;
    font-family: 'Nunito', sans-serif !important;
    padding: 0.45rem 1.2rem;
    transition: background .2s;
  }
  .stButton > button:hover { background: #C44A08 !important; }
  .stDownloadButton > button {
    background: var(--dark) !important; color: #FFFFFF !important;
    border: none; border-radius: 8px; font-weight: 700;
    font-family: 'Nunito', sans-serif !important;
  }
  .stDownloadButton > button *,
  .stDownloadButton > button p,
  .stDownloadButton > button span { color: #FFFFFF !important; }
  .stDownloadButton > button:hover { background: var(--orange) !important; }
  .stDownloadButton > button:hover,
  .stDownloadButton > button:hover * { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
HIGH = "HIGH"; MEDIUM = "MEDIUM"; LOW = "LOW"
ENV_FILE = Path(__file__).parent / ".env"
DEFAULT_CONF_THRESHOLD = 0.80
REVIEW_CONF_THRESHOLD  = 0.80

# ── Bundled reference files (shipped in the repo) ──
# These are loaded automatically at startup so the user only uploads the INPUT file.
_APP_DIR         = Path(__file__).parent
BUNDLED_MASTER   = _APP_DIR / "Veterinary_Product_Master_56_COMPREHENSIVE.xlsx"
BUNDLED_PRODUCTS = _APP_DIR / "Product_List.xlsx"
BUNDLED_TRAINING = _APP_DIR / "Training_Data.xlsx"

# AI provider definitions — all disabled by default
AI_PROVIDERS = {
    'Llama': {
        'models':   ['llama3.2', 'llama3.1', 'llama3', 'llama2'],
        'env_key':  'LLAMA_ENDPOINT',
        'label':    'LLAMA',
        'key_hint': 'http://localhost:11434/v1',
        'needs_key': False,  # Ollama = no key needed
    },
    'Gemini': {
        'models':   ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-1.5-flash'],
        'env_key':  'GEMINI_API_KEY',
        'label':    'GEMINI',
        'key_hint': 'AIza...',
        'needs_key': True,
    },
    'GPT': {
        'models':   ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-4.5-nano'],
        'env_key':  'OPENAI_API_KEY',
        'label':    'GPT',
        'key_hint': 'sk-...',
        'needs_key': True,
    },
    'Claude': {
        'models':   ['claude-haiku-4-5-20251001', 'claude-sonnet-4-6',
                     'claude-opus-4-8', 'claude-fable-5'],
        'env_key':  'ANTHROPIC_API_KEY',
        'label':    'CLAUDE',
        'key_hint': 'sk-ant-...',
        'needs_key': True,
    },
}

PRODUCT_ALIASES = {
    'NEXGUARD':'NEXGARD','NEXGAARD':'NEXGARD','PRAZITAL':'PRAZITEL',
    'MILBIMAX':'MILBEMAX','MILBERMAX':'MILBEMAX','ELIMINAL':'ELIMINALL',
    'POPENTAL':'POPANTEL','PESTIGONE':'PESTIGON','REVOLUBRUN':'REVOLUTION',
    'PANACURE':'PANACUR','CREDELLIO':'CREDELIO','MILBE MAX':'MILBEMAX',
    'PRAC TIC':'PRAC-TIC','NEX SPECTRA':'NEXGARD','NEXGAARD SPECTRA':'NEXGARD',
    'NEXGAURD SPECTRA':'NEXGARD','NEXGARD SPECTRA':'NEXGARD','FIPNIL':'FIPRONIL',
    'VITRECTO':'BRAVECTO','HEARTGARD PLUS':'HEARTGARD','HEARTGARD30':'HEARTGARD',
    'HEARTGARD 30':'HEARTGARD','K9ADV10':'ADVANTIX','K9ADV':'ADVANTIX',
}
# Generic/placeholder values that should never be treated as valid MASTER results
INVALID_MASTER_VALUES = {
    'ALL WEIGHTS','ALL WEIGHT','ALL SIZES','ALL SIZE','ALL','VARIES',
    'VARIABLE','N/A','NA','ANY','ALL WEIGHTS AND SIZES','VARIOUS',
}
# Category/description/drug-class words that AIs return instead of a real brand name
INVALID_PRODUCT_WORDS = {
    # Generic descriptions
    'WORMER','WORM','WORMING','DEWORMER','DEWORM','DEWORMING',
    'FLEA','FLEA TREATMENT','TICK','FLEA AND TICK','PARASITE',
    'TREATMENT','PRODUCT','MEDICINE','MEDICATION','SUPPLEMENT',
    'SPOT ON','SPOT-ON','TABLET','TABLETS','CAPSULE','SPRAY',
    'VETERINARY','VET','GENERIC','UNKNOWN','NOT_PRODUCT',
    'CAT WORMER','DOG WORMER','WORMER TREATMENT','PARASITE TREATMENT',
    # Drug classes — never valid brand names
    'ANTHELMINTIC','ANTIPARASITIC','ECTOPARASITICIDE','ENDECTOCIDE',
    'FLEA_TREATMENT','HEARTWORM_PREVENTATIVE','HEARTWORM',
    'ANTIBIOTIC','ANTIFUNGAL','ANALGESIC','NSAID','SEDATIVE',
    'VACCINE','IMMUNOGLOBULIN','PARASITICIDAL','INSECTICIDE',
    'ACARICIDE','ANTIPROTOZOAL','CORTICOSTEROID','STEROID',
}
SPECIES_ORDER  = ['DOG','CAT','RABBIT','HORSE','BIRD','CATTLE','SHEEP']
SPECIES_KWORDS = [
    ('DOG',    [r'\bdogs?\b',r'\bchiens?\b',r'\bcanine\b',r'\bpupp(?:y|ies)\b',
                r'\bchiot\b',r'\bk9\b',r'\bk-9\b',r'\bhund\b',r'\bperro\b']),
    ('CAT',    [r'\bcats?\b',r'\bchats?\b',r'\bfeline\b',r'\bfelin\b',
                r'\bkittens?\b',r'\bchaton\b',r'\bfelino\b',r'\bgatto\b']),
    ('RABBIT', [r'\brabbits?\b',r'\blapins?\b',r'\bbunny\b',r'\bbunnies\b']),
    ('HORSE',  [r'\bhorses?\b',r'\bcheval\b',r'\bequine\b',r'\bpony\b',r'\bfoal\b']),
    ('BIRD',   [r'\bbirds?\b',r'\boiseaux?\b',r'\bparrot\b',r'\bbudgie\b',
                r'\bpigeon\b',r'\bdove\b',r'\bcanary\b',r'\bcockatiel\b',
                r'\bpoultry\b',r'\bchicken\b',r'\bfinch\b']),
    ('CATTLE', [r'\bcattle\b',r'\bbovine\b',r'\bcows?\b',r'\bvache\b']),
    ('SHEEP',  [r'\bsheep\b',r'\bmouton\b',r'\blamb\b']),
]
SIZE_KWORDS = [
    ('XL',    [r'\bx-?l\b',r'\bextra\s*large\b',r'\bex\s*lge\b',r'\bxxl\b']),
    ('LARGE', [r'\blarge\b',r'\blge\b',r'\blgr?\b']),
    ('MEDIUM',[r'\bmedium\b',r'\bmed\b',r'\bmdm\b']),
    ('SMALL', [r'\bsmall\b',r'\bsml\b',r'\bsm\b']),
]
WT_RANGE_RE  = re.compile(r'(\d+(?:\.\d+)?)\s*(?:k(?:g|gs))?\s*[-]\s*(\d+(?:\.\d+)?)\s*k(?:g|gs)', re.I)
WT_UPTO_RE   = re.compile(r'(?:up\s*to|less\s*than|<)\s*(\d+(?:\.\d+)?)\s*k(?:g|gs)', re.I)
LBS_RANGE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[-]\s*(\d+(?:\.\d+)?)\s*lbs?', re.I)
PACK_RE = re.compile(
    r'(?:pack\s*of\s*(\d+))|(?:pk\s*(\d+))|(?:(\d+)\s*pk\b)|\[pk(\d+)\]'
    r'|(?:\[(\d+)\])|(?:bte\s*(?:de\s*)?(\d+))|(?:(\d+)\s*bte\b)'
    r'|(?:x\s*(\d+)\b)|(?:(\d+)\s*x\b)'
    r'|(?:(\d+)\s*(?:caps?|tablets?|tabs?|capsules?|doses?|chews?|comp)\b(?!\s*(?:mg|mcg|ml|g\b)))'
    r'|(?:(\d+)\s*pack\b)|(?:pack\s*(\d+)\b)', re.I)
PIP_RE = re.compile(r'\b(\d{1,2})\s*(?:pip+(?:ettes?|et)?|amp(?:oules?)?)\b', re.I)

# Token cost estimate (claude-haiku as baseline)
HAIKU_INPUT_COST_PER_1K  = 0.00025
HAIKU_OUTPUT_COST_PER_1K = 0.00125
AVG_TOKENS_PER_NOTE_IN   = 180
AVG_TOKENS_PER_NOTE_OUT  = 120
AVG_CONTEXT_WINDOW       = 200_000

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "result_df":        None,
        "review_df":        None,
        "overrides":        {},
        "run_complete":     False,
        "master_lookup":    {},
        "master_token_lkp": {},
        "master_products":  [],
        "svm_models":       {},
        "svm_trained":      False,
        "config_saved":     False,
        "train_df":         None,
        "api_keys":         {},   # {provider: key_or_endpoint}
        "system_prompt":      "",
        # chatbot review
        "chatbot_sessions":   {},   # {row_idx: [{"role":..,"content":..,"_hidden":bool}]}
        "corrections_log":    [],   # list of correction dicts (also appended to disk)
        "active_chat_row":    None,
        "reviewer_name":      "",
        "chatbot_sys_prompt": "",
        "chatbot_model":      "claude-sonnet-4-6",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Load FINAL_SYSTEM_PROMPT.txt once at startup
    if not st.session_state["system_prompt"]:
        _sp_path = os.path.join(os.path.dirname(__file__), "FINAL_SYSTEM_PROMPT.txt")
        try:
            with open(_sp_path, "r", encoding="utf-8") as _f:
                st.session_state["system_prompt"] = _f.read()
        except FileNotFoundError:
            pass  # falls back to inline string if file missing

_init_state()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _secret(key: str) -> str:
    """Read a key from Streamlit Cloud secrets. Safe when no secrets file exists (local runs)."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return ""

def _load_env():
    """Resolve provider keys: Streamlit secrets first (cloud), then .env (local)."""
    load_dotenv(ENV_FILE)
    out = {}
    for p, cfg in AI_PROVIDERS.items():
        out[p] = _secret(cfg['env_key']) or os.getenv(cfg['env_key'], '')
    return out

def _save_env(provider: str, value: str):
    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), AI_PROVIDERS[provider]['env_key'], value)

def _most_common(series):
    s = series.dropna().astype(str).str.strip()
    s = s[~s.str.upper().isin(['', 'NAN', 'NONE'])]
    return s.mode().iloc[0] if len(s) > 0 else None

def _all_species(series):
    vals = set()
    for v in series.dropna().astype(str).str.upper().str.strip():
        for sp in SPECIES_ORDER:
            if sp in v: vals.add(sp)
    if not vals: return None
    return '/'.join(s for s in SPECIES_ORDER if s in vals)

def _norm_species(found):
    return '/'.join(s for s in SPECIES_ORDER if s in found)

def _valid(v):
    return v is not None and str(v).strip().upper() not in ('','NONE','UNKNOWN','NAN','NOT_PRODUCT')

def _valid_master(v):
    """Like _valid but also rejects generic placeholder strings from the master DB."""
    return _valid(v) and str(v).strip().upper() not in INVALID_MASTER_VALUES

def _clean_val(v):
    """Extract a clean value from ALL_ formatted strings (e.g. 'REGEX: CAT | SVM: CAT(0.9)' → 'CAT')."""
    if not v or (isinstance(v, float) and pd.isna(v)): return None
    s = str(v).strip()
    m = re.search(r'(?:REGEX|MASTER|SVM|CLAUDE|GPT\w*|GEMINI|LLAMA):\s*([A-Z0-9\-/\.]+)', s)
    if m: return m.group(1)
    return s

def all_cell(regex_val, master_val, svm_val=None, svm_conf=None, ai_results=None):
    """REGEX: val | MASTER: val | SVM: val(conf) | CLAUDE: val(conf) | GPT: val(conf) | ..."""
    parts = []
    rv = str(regex_val).strip().upper() if _valid(regex_val) else None
    mv = str(master_val).strip().upper() if _valid_master(master_val) else None
    if rv: parts.append(f'REGEX: {str(regex_val).strip()}')
    if mv and mv != rv: parts.append(f'MASTER: {str(master_val).strip()}')
    if _valid(svm_val):
        c = f'({round(float(svm_conf),2)})' if svm_conf is not None else ''
        parts.append(f'SVM: {str(svm_val).strip()}{c}')
    if ai_results:
        for label, (val, conf) in ai_results.items():
            if _valid(val):
                c = f'({round(float(conf),2)})' if conf is not None else ''
                parts.append(f'{label}: {str(val).strip()}{c}')
    return ' | '.join(parts) if parts else 'UNKNOWN'

# ─────────────────────────────────────────────
# MASTER DB LOADER
# ─────────────────────────────────────────────
def load_master(master_bytes: bytes, product_list_bytes: bytes = None):
    df = pd.read_excel(io.BytesIO(master_bytes))
    df['Product_Family'] = df['Product_Name'].astype(str).str.upper().str.strip()
    lookup = {}
    for family, grp in df.groupby('Product_Family'):
        lookup[family.upper()] = {
            'species':          _all_species(grp['Target_Species']),
            'weight':           _most_common(grp['Weight_Range']),
            'formulation':      _most_common(grp['Formulation']),
            'indication':       _most_common(grp['Primary_Indication']),
            'strength':         _most_common(grp['Active_Ingredient']),
            'pack_size':        _most_common(grp['Pack_Info']),
            'pack_unit':        _most_common(grp['Pack_Unit']),
            'route':            _most_common(grp['Route_of_Administration']),
            'dosing_frequency': _most_common(grp['Dosing_Frequency']),
        }
    empty = dict(species=None, weight=None, formulation=None, indication=None,
                 strength=None, pack_size=None, pack_unit=None, route=None, dosing_frequency=None)
    if product_list_bytes:
        try:
            pl = pd.read_excel(io.BytesIO(product_list_bytes))
            for pf in pl['Product_Family'].dropna().astype(str).str.upper().str.strip():
                if pf and pf != 'UNKNOWN' and pf not in lookup:
                    lookup[pf] = empty.copy()
        except Exception:
            pass
    token_lkp = {}
    for family in lookup:
        tok = family.split()[0]
        if tok not in token_lkp:
            token_lkp[tok] = lookup[family]
    products = sorted(lookup.keys(), key=len, reverse=True)
    return lookup, token_lkp, products

_MASTER_KEYS_CACHE = {}   # id(lookup) → keys sorted by length desc (computed once, not per row)

def get_master_info(pf, lookup, token_lkp):
    empty = dict(species=None, weight=None, formulation=None, indication=None,
                 strength=None, pack_size=None, pack_unit=None, route=None, dosing_frequency=None)
    if not pf: return empty
    pf = str(pf).upper().strip()
    if pf in lookup: return lookup[pf]
    cache_key = id(lookup)
    sorted_keys = _MASTER_KEYS_CACHE.get(cache_key)
    if sorted_keys is None or len(sorted_keys) != len(lookup):
        sorted_keys = sorted(lookup.keys(), key=len, reverse=True)
        _MASTER_KEYS_CACHE[cache_key] = sorted_keys
    for key in sorted_keys:
        if pf.startswith(key) or key.startswith(pf): return lookup[key]
    tok = pf.split()[0]
    if tok in token_lkp: return token_lkp[tok]
    return empty

# ─────────────────────────────────────────────
# PRODUCT LOOKUP + REGEX EXTRACTORS
# ─────────────────────────────────────────────
_ALIASES_BY_LEN = sorted(PRODUCT_ALIASES, key=len, reverse=True)   # sorted once at import

def lookup_product(note, master_products):
    u = note.upper().strip()
    for alias in _ALIASES_BY_LEN:
        if alias in u: return PRODUCT_ALIASES[alias], HIGH
    for product in master_products:
        if product in u: return product, HIGH
    return None, LOW

def extract_species(note):
    lo = note.lower(); found = set()
    for s in re.findall(r'\b(dog|cat|rabbit|horse|bird|cattle|sheep)\b', lo): found.add(s.upper())
    for sp, pats in SPECIES_KWORDS:
        if any(re.search(p, lo) for p in pats): found.add(sp)
    return (_norm_species(found), HIGH) if found else (None, LOW)

def extract_size(note):
    lo = note.lower()
    for sz, pats in SIZE_KWORDS:
        if any(re.search(p, lo) for p in pats): return sz, HIGH
    return None, LOW

def extract_weight(note):
    m = WT_RANGE_RE.search(note)
    if m: return f'{m.group(1)}-{m.group(2)}KG', HIGH
    m = LBS_RANGE_RE.search(note)
    if m:
        return f'{round(float(m.group(1))/2.205,1)}-{round(float(m.group(2))/2.205,1)}KG', HIGH
    m = WT_UPTO_RE.search(note)
    if m: return f'LESS THAN {m.group(1)}KG', HIGH
    return None, LOW

def extract_pack_size(note):
    m = PACK_RE.search(note)
    if m:
        val = next((g for g in m.groups() if g is not None), None)
        if val: return str(int(val)), HIGH
    return None, LOW

def extract_pipettes(note):
    m = PIP_RE.search(note)
    if m: return str(m.group(1)), HIGH
    return None, LOW

# ─────────────────────────────────────────────
# PRIORITY LOGIC
# ─────────────────────────────────────────────
def determine_best(regex_val, regex_conf, master_val, svm_val, svm_conf,
                   ai_results=None, use_svm=True,
                   fallback_threshold=0.0, provider_order=None):
    """Priority: REGEX → MASTER → SVM/AI.

    If fallback_threshold > 0 and provider_order is given, AI providers are tried
    in order; the first one meeting the confidence threshold wins (cascade mode).
    Otherwise falls back to majority vote across SVM + all AI providers.
    """
    if _valid(regex_val) and regex_conf == HIGH:
        return str(regex_val), 1.0, 'REGEX'
    if _valid_master(master_val):
        return str(master_val), 0.95, 'MASTER'

    # ── AI fallback cascade ──
    if ai_results and fallback_threshold > 0 and provider_order:
        for src in provider_order:
            if src in ai_results:
                val, conf = ai_results[src]
                if _valid(val) and float(conf) >= fallback_threshold:
                    return str(val), round(float(conf), 2), src
        # No provider met the threshold — pick highest-confidence AI result if any
        best_ai = max(
            ((s, v, c) for s, (v, c) in ai_results.items() if _valid(v)),
            key=lambda x: x[2], default=None
        )
        if best_ai:
            return best_ai[1], round(best_ai[2], 2), best_ai[0]
        # Fall through to SVM below

    # ── SVM + majority vote (no cascade, or cascade found nothing) ──
    preds = []
    if use_svm and _valid(svm_val):
        preds.append(('SVM', str(svm_val), float(svm_conf or 0)))
    if ai_results and not (fallback_threshold > 0 and provider_order):
        for src, (val, conf) in ai_results.items():
            if _valid(val):
                preds.append((src, str(val), float(conf or 0)))
    if not preds:
        return 'UNKNOWN', 0.0, 'UNKNOWN'
    val_votes = Counter(v for _, v, _ in preds)
    top_val   = val_votes.most_common(1)[0][0]
    top_preds = [(s, v, c) for s, v, c in preds if v == top_val]
    src, val, conf = max(top_preds, key=lambda x: x[2])
    return val, round(conf, 2), src

# ─────────────────────────────────────────────
# SVM
# ─────────────────────────────────────────────
class SVMExtractor:
    def __init__(self, name):
        self.name = name
        self.pipe = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=8000)),
            ('clf',   CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=2))
        ])
        self.trained = False

    def train(self, texts, labels):
        pairs = [(str(t), str(l).strip().upper())
                 for t, l in zip(texts, labels)
                 if not pd.isna(l) and str(l).strip().upper() not in ('','NAN','UNKNOWN','NONE')]
        cnt = Counter(l for _, l in pairs)
        pairs = [(t, l) for t, l in pairs if cnt[l] >= 2]
        if len(pairs) < 20 or len(set(l for _, l in pairs)) < 2: return
        X, y = zip(*pairs)
        self.pipe.fit(X, y)
        self.trained = True

    def predict(self, text):
        if not self.trained: return None, 0.0
        proba = self.pipe.predict_proba([str(text)])[0]
        idx = int(np.argmax(proba))
        return self.pipe.classes_[idx], round(float(proba[idx]), 2)

    def predict_batch(self, texts):
        """Vectorised prediction — one TF-IDF transform for all rows instead of one per row."""
        if not self.trained: return [(None, 0.0)] * len(texts)
        probas = self.pipe.predict_proba([str(t) for t in texts])
        idxs   = np.argmax(probas, axis=1)
        classes = self.pipe.classes_
        return [(classes[j], round(float(probas[k, j]), 2)) for k, j in enumerate(idxs)]

def train_svms(train_bytes: bytes, labelled_bytes: bytes):
    train_df = pd.read_excel(io.BytesIO(train_bytes))
    try:
        combined = pd.concat([train_df, pd.read_excel(io.BytesIO(labelled_bytes))], ignore_index=True)
    except Exception:
        combined = train_df
    models = {}
    # Prefer BEST_ columns (clean values); fall back to plain column names
    col_map = [
        ('species',   ['BEST_Species',   'Species']),
        ('size',      ['BEST_Size',      'Size']),
        ('weight',    ['BEST_Weight',    'Weight Band']),
        ('pack_size', ['BEST_Pack_Size', 'Pack Size']),
    ]
    for attr, candidates in col_map:
        m = SVMExtractor(attr)
        col = next((c for c in candidates if c in combined.columns), None)
        if col:
            labels = combined[col].apply(_clean_val).tolist()
            m.train(combined['Notes'].tolist(), labels)
        models[attr] = m
    models['pipettes'] = SVMExtractor('pipettes')
    return models

@st.cache_resource(show_spinner="Loading built-in reference data & training models…")
def load_bundled_reference():
    """Load the repo-bundled Master DB + Product List + Training Data and train SVMs.

    Cached across sessions (@st.cache_resource) so the heavy work runs once for the
    whole server. Returns None if the bundled files are missing (falls back to manual upload).
    """
    if not BUNDLED_MASTER.exists() or not BUNDLED_TRAINING.exists():
        return None
    master_bytes       = BUNDLED_MASTER.read_bytes()
    product_list_bytes = BUNDLED_PRODUCTS.read_bytes() if BUNDLED_PRODUCTS.exists() else None
    train_bytes        = BUNDLED_TRAINING.read_bytes()
    lookup, token_lkp, master_products = load_master(master_bytes, product_list_bytes)
    svm_models = train_svms(train_bytes, train_bytes)
    train_df   = pd.read_excel(io.BytesIO(train_bytes))
    return lookup, token_lkp, master_products, svm_models, train_df

# ─────────────────────────────────────────────
# AI CALLERS  (Claude / GPT / Gemini / Llama)
# ─────────────────────────────────────────────
_FEW_SHOT_CACHE     = {}   # id(train_df) → examples string (static per run; don't rebuild per batch)
_KNOWN_BRANDS_CACHE = {}   # id(master_products) → joined brand string

def _build_few_shot(train_df, n=2):
    if train_df is None: return ""
    ck = (id(train_df), n)
    if ck in _FEW_SHOT_CACHE: return _FEW_SHOT_CACHE[ck]
    # Prefer BEST_ columns for clean values; fall back to plain names
    def _gcol(row, *candidates):
        for c in candidates:
            v = row.get(c)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return _clean_val(str(v)) or "UNKNOWN"
        return "UNKNOWN"
    sp_col = 'BEST_Species' if 'BEST_Species' in train_df.columns else 'Species'
    examples = []
    for _, row in train_df.dropna(subset=[sp_col]).head(n).iterrows():
        pf  = _gcol(row, 'BEST_Product', 'Product Family', 'Product_Family')
        sp  = _gcol(row, 'BEST_Species',  'Species')
        sz  = _gcol(row, 'BEST_Size',     'Size')
        wt  = _gcol(row, 'BEST_Weight',   'Weight Band')
        ps  = _gcol(row, 'BEST_Pack_Size','Pack Size')
        examples.append(
            f'"{row["Notes"]}" → product:{pf}, species:{sp}, '
            f'size:{sz}, weight:{wt}, pack_size:{ps}, pipettes:UNKNOWN'
        )
    out = "\n".join(examples)
    _FEW_SHOT_CACHE[ck] = out
    return out

def _build_prompt(notes_batch, train_df, master_products, open_mode=False):
    """
    open_mode=True  → no KNOWN BRANDS constraint; AI uses full veterinary knowledge.
                      Used when REGEX+MASTER found nothing — same behaviour as pasting
                      the note directly into Claude.
    open_mode=False → KNOWN BRANDS list included to help confirm partial matches.
    """
    examples = _build_few_shot(train_df)
    notes_text = "\n".join(f'{i+1}. "{n}"' for i, n in enumerate(notes_batch))

    if open_mode:
        brand_section = (
            "Use your full veterinary pharmacology knowledge to identify the product. "
            "The product may be ANY approved veterinary medicine — do not limit yourself "
            "to a specific brand list. Common abbreviations: PHP=product prefix, "
            "PK/BTE=pack, PIP=pipette, SML=small, LGE=large, MED=medium, COMP=tablet, "
            "chien/perro=dog, chat/gato=cat."
        )
    else:
        ck = id(master_products)
        known = _KNOWN_BRANDS_CACHE.get(ck)
        if known is None:
            known = ", ".join(sorted(set(master_products) | set(PRODUCT_ALIASES.values())))
            _KNOWN_BRANDS_CACHE[ck] = known
        brand_section = (
            f"KNOWN BRANDS: {known}\n\n"
            "ABBREVIATIONS: PK/BTE=pack, PIP=pipette, SML=small, LGE=large, MED=medium, COMP=tablet. "
            "chien/perro=dog. chat/gato=cat."
        )

    return f"""Extract veterinary product attributes from messy retail sales notes.
Return ONLY valid JSON — no markdown.

IMPORTANT:
- Prefer accuracy; light inference is allowed for product knowledge.
- If you infer (not explicit in text) use lower confidence: 0.40-0.70.
- Use UNKNOWN only when there is no reliable clue.
- For services/procedures use product_family = NOT_PRODUCT.

{brand_section}

EXAMPLES:
{examples}

NOTES:
{notes_text}

Return JSON:
{{"items":[{{"product_family":"string","product_family_conf":0.0,
"species":"DOG|CAT|RABBIT|HORSE|BIRD|CATTLE|SHEEP|UNKNOWN","species_conf":0.0,
"size":"SMALL|MEDIUM|LARGE|XL|UNKNOWN","size_conf":0.0,
"weight":"string","weight_conf":0.0,"pack_size":"string","pack_size_conf":0.0,
"pipettes":"string","pipettes_conf":0.0,"reasoning":"string"}}]}}"""

def _parse_json(text):
    try:
        if not text: return None
        text = re.sub(r"```(?:json)?\s*", "", str(text)).strip("`").strip()
        obj = json.loads(text)
        if isinstance(obj, dict) and "items" in obj: return obj["items"]
        if isinstance(obj, list): return obj
        return None
    except Exception:
        return None

def call_claude(notes_batch, api_key, model, master_products, train_df, batch_num=0, open_mode=False):
    try:
        import anthropic as _anthropic
    except ImportError:
        st.warning("anthropic package not installed. Run: pip install anthropic")
        return None
    client = _anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(notes_batch, train_df, master_products, open_mode=open_mode)
    for attempt in range(4):
        try:
            sys_prompt = st.session_state.get("system_prompt") or "You are a veterinary data extraction assistant. Return ONLY valid JSON."
            resp = client.messages.create(
                model=model, max_tokens=4096,
                system=sys_prompt,
                messages=[{"role":"user","content":prompt}])
            result = _parse_json(resp.content[0].text if resp.content else "")
            if result: return result
        except Exception as e:
            errs = str(e).lower()
            wait = (30 + attempt * 30) if 'rate' in errs else (15 + attempt * 20) if '5' in errs[:3] else 2 ** attempt
            time.sleep(wait)
    return None

def call_gpt(notes_batch, api_key, model, master_products, train_df, batch_num=0, open_mode=False):
    try:
        import openai
    except ImportError:
        st.warning("openai package not installed. Run: pip install openai")
        return None
    client = openai.OpenAI(api_key=api_key)
    prompt = _build_prompt(notes_batch, train_df, master_products, open_mode=open_mode)
    for attempt in range(4):
        try:
            sys_prompt = st.session_state.get("system_prompt") or "You are a veterinary data extraction assistant. Return ONLY valid JSON."
            resp = client.chat.completions.create(
                model=model, max_tokens=4096,
                messages=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":prompt}
                ])
            result = _parse_json(resp.choices[0].message.content)
            if result: return result
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def call_gemini(notes_batch, api_key, model, master_products, train_df, batch_num=0, open_mode=False):
    try:
        import google.generativeai as genai
    except ImportError:
        st.warning("google-generativeai package not installed. Run: pip install google-generativeai")
        return None
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)
    prompt = _build_prompt(notes_batch, train_df, master_products, open_mode=open_mode)
    for attempt in range(4):
        try:
            resp = gmodel.generate_content(prompt)
            result = _parse_json(resp.text)
            if result: return result
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def call_llama(notes_batch, endpoint, model, master_products, train_df, batch_num=0, open_mode=False):
    """Calls Llama via OpenAI-compatible API (e.g. Ollama at localhost:11434)."""
    try:
        import openai
    except ImportError:
        st.warning("openai package not installed. Run: pip install openai")
        return None
    base_url = endpoint.rstrip('/') if endpoint else "http://localhost:11434/v1"
    if not base_url.endswith('/v1'): base_url += '/v1'
    client = openai.OpenAI(api_key="ollama", base_url=base_url)
    prompt = _build_prompt(notes_batch, train_df, master_products, open_mode=open_mode)
    for attempt in range(4):
        try:
            sys_prompt = st.session_state.get("system_prompt") or "You are a veterinary data extraction assistant. Return ONLY valid JSON."
            resp = client.chat.completions.create(
                model=model, max_tokens=4096,
                messages=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":prompt}
                ])
            result = _parse_json(resp.choices[0].message.content)
            if result: return result
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def call_ai(notes_batch, provider, key_or_endpoint, model, master_products, train_df,
            batch_num=0, open_mode=False):
    if provider == 'Claude':
        return call_claude(notes_batch, key_or_endpoint, model, master_products, train_df, batch_num, open_mode)
    elif provider == 'GPT':
        return call_gpt(notes_batch, key_or_endpoint, model, master_products, train_df, batch_num, open_mode)
    elif provider == 'Gemini':
        return call_gemini(notes_batch, key_or_endpoint, model, master_products, train_df, batch_num, open_mode)
    elif provider == 'Llama':
        return call_llama(notes_batch, key_or_endpoint, model, master_products, train_df, batch_num, open_mode)
    return None

# ─────────────────────────────────────────────
# WEB SEARCH FALLBACK
# ─────────────────────────────────────────────
def _web_search_snippets(note: str, n: int = 4) -> str:
    """Return top-n DuckDuckGo snippets for a veterinary product note. Returns '' on any failure."""
    try:
        from duckduckgo_search import DDGS
        query = f"veterinary product \"{note}\" species ingredients"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n))
        if not results:
            return ""
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results if r.get('body'))
    except ImportError:
        return ""
    except Exception:
        return ""

def _build_web_prompt(note: str, web_context: str, master_products) -> str:
    return f"""You are a veterinary data extraction assistant.
Using the web search results below, identify the SPECIFIC BRAND NAME of this veterinary product.

CRITICAL RULES:
- product_family MUST be a real brand/product name (e.g. PROFENDER, DRONSPOT, STRONGHOLD)
- NEVER return a category word like WORMER, FLEA, TREATMENT, SPOT-ON, PRODUCT, MEDICINE
- If you cannot find a specific brand name, return UNKNOWN
- Use the web context to identify the exact product

NOTE: "{note}"

WEB SEARCH CONTEXT:
{web_context}

Return JSON (single item, not array):
{{"product_family":"BRAND_NAME_ONLY","product_family_conf":0.0,
"species":"DOG|CAT|RABBIT|HORSE|BIRD|CATTLE|SHEEP|UNKNOWN","species_conf":0.0,
"size":"SMALL|MEDIUM|LARGE|XL|UNKNOWN","size_conf":0.0,
"weight":"string","weight_conf":0.0,"pack_size":"string","pack_size_conf":0.0,
"pipettes":"string","pipettes_conf":0.0,"reasoning":"string"}}"""

def call_ai_with_web_context(note: str, provider: str, key_or_endpoint: str, model: str,
                              master_products, web_context: str):
    """Single-note AI call augmented with web search context. Returns a single result dict or None."""
    prompt = _build_web_prompt(note, web_context, master_products)

    def _extract(raw_text):
        text = re.sub(r"```(?:json)?\s*", "", str(raw_text or "")).strip("`").strip()
        try:
            obj = json.loads(text)
            # Accept single dict or items list
            if isinstance(obj, list) and obj: return obj[0]
            if isinstance(obj, dict) and 'product_family' in obj: return obj
            if isinstance(obj, dict) and 'items' in obj and obj['items']: return obj['items'][0]
        except Exception:
            pass
        return None

    _sys = st.session_state.get("system_prompt") or "You are a veterinary data extraction assistant. Return ONLY valid JSON."
    for attempt in range(3):
        try:
            if provider == 'Claude':
                import anthropic as _anthropic
                client = _anthropic.Anthropic(api_key=key_or_endpoint)
                resp = client.messages.create(
                    model=model, max_tokens=1024,
                    system=_sys,
                    messages=[{"role":"user","content":prompt}])
                result = _extract(resp.content[0].text if resp.content else "")
            elif provider == 'GPT':
                import openai
                client = openai.OpenAI(api_key=key_or_endpoint)
                resp = client.chat.completions.create(
                    model=model, max_tokens=1024,
                    messages=[
                        {"role":"system","content":_sys},
                        {"role":"user","content":prompt}])
                result = _extract(resp.choices[0].message.content)
            elif provider == 'Gemini':
                import google.generativeai as genai
                genai.configure(api_key=key_or_endpoint)
                result = _extract(genai.GenerativeModel(model).generate_content(prompt).text)
            elif provider == 'Llama':
                import openai
                base_url = key_or_endpoint.rstrip('/') if key_or_endpoint else "http://localhost:11434/v1"
                if not base_url.endswith('/v1'): base_url += '/v1'
                client = openai.OpenAI(api_key="ollama", base_url=base_url)
                resp = client.chat.completions.create(
                    model=model, max_tokens=1024,
                    messages=[
                        {"role":"system","content":_sys},
                        {"role":"user","content":prompt}])
                result = _extract(resp.choices[0].message.content)
            else:
                return None
            if result: return result
        except Exception:
            time.sleep(2 ** attempt)
    return None

# ─────────────────────────────────────────────
# COST ESTIMATOR
# ─────────────────────────────────────────────
def estimate_cost(n_rows: int, batch_size: int = 10, provider: str = 'Claude'):
    n_batches  = math.ceil(n_rows / batch_size)
    in_tokens  = n_rows * AVG_TOKENS_PER_NOTE_IN
    out_tokens = n_rows * AVG_TOKENS_PER_NOTE_OUT
    cost = (in_tokens / 1000 * HAIKU_INPUT_COST_PER_1K +
            out_tokens / 1000 * HAIKU_OUTPUT_COST_PER_1K)
    secs = n_batches * 1.2
    note = "" if provider == 'Claude' else f"Estimate based on Claude Haiku pricing. {provider} pricing may differ."
    return {
        "n_rows": n_rows, "in_tokens": in_tokens, "out_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "cost_usd": round(cost, 4), "est_mins": round(secs / 60, 1),
        "pct_context": round((in_tokens / AVG_CONTEXT_WINDOW) * 100, 1),
        "note": note,
    }

# ─────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(input_df, lookup, token_lkp, master_products,
                 svm_models, train_df, use_svm,
                 enabled_providers, api_keys,
                 batch_size, progress_bar, status_text,
                 fallback_threshold=0.0, use_web_search=False):
    """
    enabled_providers:   {provider_name: model_string}  e.g. {'Claude': 'claude-haiku-...'}
    api_keys:            {provider_name: key_or_endpoint}
    fallback_threshold:  min confidence for a provider to "win" in cascade mode (0 = majority vote)
    use_web_search:      if True, rows still UNKNOWN after AI cascade get a DuckDuckGo re-try
    """
    notes = input_df['Notes'].astype(str).tolist()
    total = len(notes)

    # ── Pass 1: REGEX + MASTER ──
    status_text.text("Pass 1: REGEX + Master DB…")
    p1, needs_ai = [], []
    prog_step = max(1, total // 100)   # update progress ~100 times total, not per row
    for i, note in enumerate(notes):
        pf,  pfc = lookup_product(note, master_products)
        sp,  spc = extract_species(note)
        sz,  szc = extract_size(note)
        wt,  wtc = extract_weight(note)
        ps,  psc = extract_pack_size(note)
        pip, pic = extract_pipettes(note)
        mi = get_master_info(pf, lookup, token_lkp)
        p1.append(dict(note=note, pf=pf, pfc=pfc, sp=sp, spc=spc,
                       sz=sz, szc=szc, wt=wt, wtc=wtc,
                       ps=ps, psc=psc, pip=pip, pic=pic, mi=mi))
        if any(x is None for x in [pf, sp, sz, wt, ps]):
            needs_ai.append(i)
        if (i + 1) % prog_step == 0 or i + 1 == total:
            progress_bar.progress(int((i+1) / total * 40))

    # ── Pass 2: SVM (vectorised — one sklearn call per field for ALL rows) ──
    svm_cache = {}
    if use_svm:
        status_text.text("Pass 2: SVM prediction…")
        batch_preds = {
            'sp':  svm_models['species'].predict_batch(notes),
            'sz':  svm_models['size'].predict_batch(notes),
            'wt':  svm_models['weight'].predict_batch(notes),
            'ps':  svm_models['pack_size'].predict_batch(notes),
            'pip': svm_models['pipettes'].predict_batch(notes),
        }
        for i in range(total):
            svm_cache[i] = {k: batch_preds[k][i] for k in batch_preds}
        progress_bar.progress(60)

    # ── Pass 3: AI — all enabled providers run IN PARALLEL per batch ──
    # ai_caches: {label: {row_idx: result_dict}}
    ai_caches = {}
    active_providers = {
        p: m for p, m in enabled_providers.items()
        if api_keys.get(p) or not AI_PROVIDERS[p]['needs_key']
    }
    skipped = set(enabled_providers) - set(active_providers)
    if skipped:
        status_text.text(f"Skipping {', '.join(skipped)} — no API key.")

    if active_providers and needs_ai:
        n_batches = math.ceil(len(needs_ai) / batch_size)
        for lbl in active_providers:
            ai_caches[AI_PROVIDERS[lbl]['label']] = {}

        # Always use constrained prompt — AI needs KNOWN BRANDS context to avoid hallucination
        open_mode = False

        # Build every (batch, provider) task and run them ALL concurrently,
        # not batch-by-batch — wall-clock is no longer n_batches × latency.
        batches = []
        for b, start in enumerate(range(0, len(needs_ai), batch_size)):
            idx_batch = needs_ai[start:start+batch_size]
            batches.append((b, idx_batch, [notes[i] for i in idx_batch]))

        def _run_one(b, idx_batch, note_batch, provider, model):
            key = api_keys.get(provider, '')
            return provider, idx_batch, call_ai(note_batch, provider, key, model,
                                                master_products, train_df, b+1, open_mode)

        # ~8 in-flight requests per provider — high throughput without hammering rate limits
        max_workers = min(32, max(len(active_providers), len(active_providers) * 8))
        total_tasks = n_batches * len(active_providers)
        done_tasks  = 0
        status_text.text(
            f"Pass 3: {n_batches} batches × {len(active_providers)} providers "
            f"({total_tasks} calls, {max_workers} concurrent)…")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_one, b, ib, nb, p, m)
                       for (b, ib, nb) in batches
                       for p, m in active_providers.items()]
            for fut in as_completed(futures):
                provider, idx_batch, results = fut.result()
                label = AI_PROVIDERS[provider]['label']
                if results:
                    for j, idx in enumerate(idx_batch):
                        if j < len(results):
                            ai_caches[label][idx] = results[j]
                done_tasks += 1
                if done_tasks % max(1, total_tasks // 50) == 0 or done_tasks == total_tasks:
                    status_text.text(f"Pass 3: {done_tasks}/{total_tasks} AI calls done…")
                    progress_bar.progress(min(60 + int(done_tasks / total_tasks * 35), 95))

    # ── Build output ──
    # provider_order: labels in the order the user enabled them — used for cascade
    provider_order = [AI_PROVIDERS[p]['label'] for p in enabled_providers]
    use_cascade    = fallback_threshold > 0 and len(provider_order) > 0

    status_text.text("Building output…")
    rows = []
    for i, r in enumerate(p1):
        sv = svm_cache.get(i, {})
        def _sv(k): return sv.get(k, (None, 0.0))

        svm_sp, svm_spc = _sv('sp'); svm_sz, svm_szc = _sv('sz')
        svm_wt, svm_wtc = _sv('wt'); svm_ps, svm_psc = _sv('ps')

        # Collect results from all enabled AI providers per field
        def _ai_field(val_key, conf_key, is_product=False):
            out = {}
            for lbl, cache in ai_caches.items():
                res = cache.get(i, {})
                if not res: continue
                val  = str(res.get(val_key, '') or '').strip().upper() or None
                conf = round(float(res.get(conf_key, 0) or 0), 2)
                # Reject drug class / category words masquerading as product names
                if is_product and val and val in INVALID_PRODUCT_WORDS:
                    val = None
                if val: out[lbl] = (val, conf)
            return out

        ai_pf  = _ai_field('product_family', 'product_family_conf', is_product=True)
        ai_sp  = _ai_field('species',        'species_conf')
        ai_sz  = _ai_field('size',           'size_conf')
        ai_wt  = _ai_field('weight',         'weight_conf')
        ai_ps  = _ai_field('pack_size',      'pack_size_conf')
        ai_pip = _ai_field('pipettes',       'pipettes_conf')

        _cascade_kw = dict(
            fallback_threshold=fallback_threshold if use_cascade else 0.0,
            provider_order=provider_order if use_cascade else None,
        )

        mi = r['mi']
        bpf_v,  bpf_c,  bpf_s  = determine_best(r['pf'],  r['pfc'], None,            None,    None,    ai_pf,  use_svm, **_cascade_kw)
        bsp_v,  bsp_c,  bsp_s  = determine_best(r['sp'],  r['spc'], mi['species'],   svm_sp,  svm_spc, ai_sp,  use_svm, **_cascade_kw)
        bsz_v,  bsz_c,  bsz_s  = determine_best(r['sz'],  r['szc'], None,            svm_sz,  svm_szc, ai_sz,  use_svm, **_cascade_kw)
        bwt_v,  bwt_c,  bwt_s  = determine_best(r['wt'],  r['wtc'], mi['weight'],    svm_wt,  svm_wtc, ai_wt,  use_svm, **_cascade_kw)
        bps_v,  bps_c,  bps_s  = determine_best(r['ps'],  r['psc'], mi['pack_size'], svm_ps,  svm_psc, ai_ps,  use_svm, **_cascade_kw)
        bpip_v, bpip_c, bpip_s = determine_best(r['pip'], r['pic'], None,            None,    None,    ai_pip, use_svm, **_cascade_kw)

        reasoning = ' | '.join(
            f"{lbl}: {str(cache.get(i, {}).get('reasoning', '') or '').strip()}"
            for lbl, cache in ai_caches.items()
            if str(cache.get(i, {}).get('reasoning', '') or '').strip()
        ) or None

        rows.append({
            'Notes':          r['note'],
            'ALL_Product':    all_cell(r['pf'], None,            ai_results=ai_pf),
            'BEST_Product':   bpf_v,  'BEST_Product_Source':   bpf_s,  'BEST_Product_Conf':   bpf_c,
            'ALL_Species':    all_cell(r['sp'], mi['species'],   svm_sp, svm_spc, ai_results=ai_sp),
            'BEST_Species':   bsp_v,  'BEST_Species_Source':   bsp_s,  'BEST_Species_Conf':   bsp_c,
            'ALL_Size':       all_cell(r['sz'], None,            svm_sz, svm_szc, ai_results=ai_sz),
            'BEST_Size':      bsz_v,  'BEST_Size_Source':      bsz_s,  'BEST_Size_Conf':      bsz_c,
            'ALL_Weight':     all_cell(r['wt'], mi['weight'],    svm_wt, svm_wtc, ai_results=ai_wt),
            'BEST_Weight':    bwt_v,  'BEST_Weight_Source':    bwt_s,  'BEST_Weight_Conf':    bwt_c,
            'ALL_Pack_Size':  all_cell(r['ps'], mi['pack_size'], svm_ps, svm_psc, ai_results=ai_ps),
            'BEST_Pack_Size': bps_v,  'BEST_Pack_Size_Source': bps_s,  'BEST_Pack_Size_Conf': bps_c,
            'ALL_Pipettes':   all_cell(r['pip'], None,           ai_results=ai_pip),
            'BEST_Pipettes':  bpip_v, 'BEST_Pipettes_Source':  bpip_s, 'BEST_Pipettes_Conf':  bpip_c,
            'MASTER_Formulation':       mi.get('formulation'),
            'MASTER_Indication':        mi.get('indication'),
            'MASTER_Active_Ingredient': mi.get('strength'),
            'MASTER_Route':             mi.get('route'),
            'MASTER_Pack_Info':         mi.get('pack_size'),
            'MASTER_Dosing_Frequency':  mi.get('dosing_frequency'),
            'MATCH_TYPE':    'EXACT' if (r['pf'] and r['pfc']==HIGH) else ('ALIAS' if r['pf'] else 'NONE'),
            'AI_Providers':  ', '.join(enabled_providers.keys()) if enabled_providers else 'NONE',
            'AI_Reasoning':  reasoning,
        })

    # ── Pass 4: Web search for low-confidence / UNKNOWN product rows ──
    WEB_CONF_THRESHOLD = 0.70
    if use_web_search and enabled_providers:
        primary_provider = next(iter(enabled_providers))
        primary_model    = enabled_providers[primary_provider]
        primary_key      = api_keys.get(primary_provider, '')
        if primary_key or not AI_PROVIDERS[primary_provider]['needs_key']:
            unknown_idxs = [
                i for i, row in enumerate(rows)
                if row['BEST_Product'] == 'UNKNOWN'
                or float(row.get('BEST_Product_Conf') or 0) < WEB_CONF_THRESHOLD
            ]
            if unknown_idxs:
                status_text.text(
                    f"Pass 4: Web search for {len(unknown_idxs)} UNKNOWN rows "
                    f"→ {primary_provider}…")

                def _web_task(i):
                    note = rows[i]['Notes']
                    snippets = _web_search_snippets(note)
                    if not snippets:
                        return i, None
                    return i, call_ai_with_web_context(
                        note, primary_provider, primary_key, primary_model,
                        master_products, snippets)

                done_web = 0
                with ThreadPoolExecutor(max_workers=6) as wex:
                    web_futs = [wex.submit(_web_task, i) for i in unknown_idxs]
                    for wf in as_completed(web_futs):
                        i, web_result = wf.result()
                        if web_result:
                            pf_v  = str(web_result.get('product_family','') or '').strip().upper() or None
                            pf_c  = round(float(web_result.get('product_family_conf', 0) or 0), 2)
                            sp_v  = str(web_result.get('species','') or '').strip().upper() or None
                            sp_c  = round(float(web_result.get('species_conf', 0) or 0), 2)
                            sz_v  = str(web_result.get('size','') or '').strip().upper() or None
                            sz_c  = round(float(web_result.get('size_conf', 0) or 0), 2)
                            wt_v  = str(web_result.get('weight','') or '').strip().upper() or None
                            wt_c  = round(float(web_result.get('weight_conf', 0) or 0), 2)
                            ps_v  = str(web_result.get('pack_size','') or '').strip().upper() or None
                            ps_c  = round(float(web_result.get('pack_size_conf', 0) or 0), 2)
                            pip_v = str(web_result.get('pipettes','') or '').strip().upper() or None
                            pip_c = round(float(web_result.get('pipettes_conf', 0) or 0), 2)
                            web_src = f'WEB+{AI_PROVIDERS[primary_provider]["label"]}'
                            web_reasoning = str(web_result.get('reasoning','') or '').strip()
                            # Reject category/generic words masquerading as product names
                            if pf_v and pf_v.upper() in INVALID_PRODUCT_WORDS:
                                pf_v = None
                            # Overwrite BEST_Product only with a real brand name
                            if _valid(pf_v):
                                rows[i]['BEST_Product']        = pf_v
                                rows[i]['BEST_Product_Source'] = web_src
                                rows[i]['BEST_Product_Conf']   = pf_c
                                rows[i]['ALL_Product']         = (rows[i]['ALL_Product'] or 'UNKNOWN') + f' | WEB: {pf_v}({pf_c})'
                            for fld, v, c in [('Species',sp_v,sp_c),('Size',sz_v,sz_c),
                                              ('Weight',wt_v,wt_c),('Pack_Size',ps_v,ps_c),
                                              ('Pipettes',pip_v,pip_c)]:
                                if _valid(v) and rows[i].get(f'BEST_{fld}') == 'UNKNOWN':
                                    rows[i][f'BEST_{fld}']        = v
                                    rows[i][f'BEST_{fld}_Source'] = web_src
                                    rows[i][f'BEST_{fld}_Conf']   = c
                            if web_reasoning:
                                existing = rows[i].get('AI_Reasoning') or ''
                                rows[i]['AI_Reasoning'] = (existing + ' | ' if existing else '') + f'WEB: {web_reasoning}'
                        done_web += 1
                        progress_bar.progress(min(95 + int(done_web / len(unknown_idxs) * 5), 99))

    progress_bar.progress(100)
    status_text.text("Done.")
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# REVIEW QUEUE BUILDER
# ─────────────────────────────────────────────
def build_review_queue(df: pd.DataFrame, conf_threshold: float) -> pd.DataFrame:
    svm_src_cols = ['BEST_Species_Source','BEST_Size_Source',
                    'BEST_Weight_Source','BEST_Pack_Size_Source','BEST_Pipettes_Source']
    conf_cols    = ['BEST_Species_Conf','BEST_Size_Conf',
                    'BEST_Weight_Conf','BEST_Pack_Size_Conf','BEST_Pipettes_Conf']
    mask = pd.Series(False, index=df.index)
    for sc, cc in zip(svm_src_cols, conf_cols):
        if sc in df.columns and cc in df.columns:
            mask |= ((df[sc] == 'SVM') & (df[cc].fillna(0) < conf_threshold))
    return df[mask].copy()

# ─────────────────────────────────────────────
# CHATBOT REVIEW HELPERS
# ─────────────────────────────────────────────
def _chatbot_flagged_rows(result_df):
    """Return list of (row_idx, row_series, flag_reason) for chatbot review."""
    # Vectorised pre-filter — only materialise rows that are actually flagged
    prod_s  = result_df.get('BEST_Product', pd.Series('', index=result_df.index)).fillna('').astype(str)
    conf_s  = pd.to_numeric(result_df.get('BEST_Product_Conf', pd.Series(0, index=result_df.index)), errors='coerce').fillna(0)
    mtype_s = result_df.get('MATCH_TYPE', pd.Series('', index=result_df.index)).fillna('').astype(str)
    flagged_mask = (prod_s == 'UNKNOWN') | (conf_s < 0.85) | (mtype_s == 'NONE')
    out = []
    for idx in result_df.index[flagged_mask]:
        row   = result_df.loc[idx]
        prod  = prod_s.at[idx]
        conf  = float(conf_s.at[idx])
        mtype = mtype_s.at[idx]
        if prod == 'UNKNOWN':
            out.append((idx, row, 'UNKNOWN product'))
        elif conf < 0.85:
            out.append((idx, row, f'Low confidence ({conf*100:.0f}%)'))
        elif mtype == 'NONE':
            out.append((idx, row, 'No master match'))
    return out

def _build_chatbot_sys_prompt():
    """FINAL_SYSTEM_PROMPT.txt + conversational review mode instructions."""
    base = st.session_state.get("system_prompt") or \
        "You are a veterinary data extraction specialist. Extract product attributes from notes."
    addon = """

════════════════════════════════════════════════════════════
CHATBOT REVIEW MODE
════════════════════════════════════════════════════════════
You are assisting a human reviewer to verify and correct pipeline classifications.

When given a note:
1. Apply ALL extraction rules above carefully.
2. Explain your reasoning for each field (one line each).
3. Always display your proposed values in this exact format:
     Product:   [value]
     Species:   [value]
     Size:      [value]
     Weight:    [value]
     Pack Size: [value]
     Pipettes:  [value]
4. Ask: "Do you confirm this, or would you like to change anything?"

If the reviewer confirms → acknowledge and summarise in one sentence.
If the reviewer corrects a field → update your values and ask to confirm again.
If asked a question → answer using the product list and rules above.

When the system asks you to output JSON (you will see the exact format requested),
output ONLY raw JSON — no markdown fences, no explanation, nothing else.
════════════════════════════════════════════════════════════
"""
    return base + addon

def _call_chatbot_api(messages, api_key, max_tokens=600, model=None):
    """Call Claude for chatbot interaction. Model defaults to the one picked in the chatbot tab."""
    try:
        import anthropic as _anthropic
    except ImportError:
        return "Error: anthropic package not installed."
    if not api_key:
        return "Error: Claude API key not configured."
    if not st.session_state.get("chatbot_sys_prompt"):
        st.session_state.chatbot_sys_prompt = _build_chatbot_sys_prompt()
    model = model or st.session_state.get("chatbot_model") or "claude-sonnet-4-6"
    try:
        client = _anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=st.session_state.chatbot_sys_prompt,
            messages=messages,
        )
        return resp.content[0].text if resp.content else ""
    except Exception as e:
        return f"Error calling Claude: {e}"

def _determine_changed(chatbot_product, best_product, all_product_str):
    """Return NO / PARTIAL / YES depending on how chatbot result relates to pipeline output."""
    cp = (chatbot_product or '').strip().upper()
    bp = (best_product   or '').strip().upper()
    if cp == bp:
        return 'NO'
    if cp and cp in str(all_product_str or '').upper():
        return 'PARTIAL'
    return 'YES'

def _save_correction_to_disk(corr):
    """Append one correction dict to corrections_log.xlsx (create if missing)."""
    log_path = Path(__file__).parent / "corrections_log.xlsx"
    new_df = pd.DataFrame([corr])
    if log_path.exists():
        try:
            combined = pd.concat([pd.read_excel(log_path), new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_excel(log_path, index=False)

# ─────────────────────────────────────────────
# EXCEL EXPORT  (3 sheets)
# ─────────────────────────────────────────────
def to_excel_bytes(result_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    simplified = [
        'Notes',
        'BEST_Product',   'BEST_Product_Source',   'BEST_Product_Conf',
        'ALL_Species',    'BEST_Species',   'BEST_Species_Source',   'BEST_Species_Conf',
        'ALL_Size',       'BEST_Size',      'BEST_Size_Source',      'BEST_Size_Conf',
        'ALL_Weight',     'BEST_Weight',    'BEST_Weight_Source',    'BEST_Weight_Conf',
        'ALL_Pack_Size',  'BEST_Pack_Size', 'BEST_Pack_Size_Source', 'BEST_Pack_Size_Conf',
        'BEST_Pipettes',  'BEST_Pipettes_Source',
        'MASTER_Formulation','MASTER_Indication','MASTER_Active_Ingredient',
        'MASTER_Pack_Info','MASTER_Dosing_Frequency','MATCH_TYPE',
    ]
    src_rows = []
    for field in ['BEST_Product_Source','BEST_Species_Source','BEST_Size_Source',
                  'BEST_Weight_Source','BEST_Pack_Size_Source']:
        if field in result_df.columns:
            for src, cnt in result_df[field].value_counts().items():
                src_rows.append({'Field':field,'Source':src,'Count':cnt,
                                 'Pct':f'{cnt/len(result_df)*100:.1f}%'})
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Full Results', index=False)
        result_df[[c for c in simplified if c in result_df.columns]].to_excel(
            writer, sheet_name='Best Matches Only', index=False)
        pd.DataFrame(src_rows).to_excel(writer, sheet_name='Source Distribution', index=False)
    return buf.getvalue()

# ─────────────────────────────────────────────
# ═══════════  UI  ════════════════════════════
# ─────────────────────────────────────────────

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:18px 0 24px; border-bottom:1px solid #333; margin-bottom:12px'>
      <span style='
        font-family: Nunito, sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        font-style: italic;
        color: #E8590C;
        letter-spacing: -1px;
        line-height: 1;
      '>zoetis</span><br>
      <span style='
        font-size: .75rem;
        color: #aaa;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
      '>Veterinary Classifier</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Methods")
    st.checkbox("REGEX",     value=True, disabled=True, help="Always on — instant, no cost")
    st.checkbox("MASTER DB", value=True, disabled=True, help="Always on — no cost")
    use_svm = st.checkbox("SVM", value=True, help="Fast ML — no API cost")

    st.markdown("---")
    st.markdown("### AI Models")
    st.caption("Tick one or more · keys set in Config tab")
    enabled_providers = {}
    for _provider, _cfg in AI_PROVIDERS.items():
        _c1, _c2 = st.columns([1, 2])
        _on = _c1.checkbox(_provider, value=False, key=f"chk_{_provider}")
        if _on:
            _model = _c2.selectbox("", _cfg['models'],
                                   key=f"mdl_{_provider}",
                                   label_visibility="collapsed")
            enabled_providers[_provider] = _model
    use_ai = len(enabled_providers) > 0

    st.markdown("---")
    st.markdown("### Settings")
    conf_thresh = st.slider("Review threshold", 0.5, 1.0, REVIEW_CONF_THRESHOLD,
                            step=0.05, help="Rows below this confidence go to Review Queue")
    batch_size  = st.slider("AI batch size", 1, 25, 5, step=1)

    st.markdown("---")
    st.markdown("### AI Fallback Cascade")
    st.caption("If primary AI confidence < threshold → try next enabled AI → then UNKNOWN")
    use_cascade       = st.checkbox("Enable cascade", value=True,
                                    help="Try providers in order; stop at first confident result")
    fallback_threshold = st.slider("Cascade threshold", 0.30, 0.90, 0.60, step=0.05,
                                   disabled=not use_cascade,
                                   help="Min confidence for a provider to 'win'") if use_cascade else 0.0
    if use_cascade and len(enabled_providers) > 1:
        st.caption("Order: " + " → ".join(enabled_providers.keys()) + " → UNKNOWN")

    st.markdown("---")
    st.markdown("### 🌐 Web Search Fallback")
    use_web_search = st.checkbox(
        "Search web for low-confidence rows", value=True,
        help="If BEST_Product is UNKNOWN or confidence < 0.90, DuckDuckGo-searches the note "
             "and re-runs the primary AI with web context. Requires: pip install duckduckgo-search")
    if use_web_search:
        st.caption("Fires when product conf < 0.80 or UNKNOWN. Uses your primary (first ticked) AI.")

    st.markdown("---")
    if st.session_state.run_complete and st.session_state.result_df is not None:
        excel_bytes = to_excel_bytes(st.session_state.result_df)
        st.download_button("⬇ Download Results", excel_bytes,
                           "zoetis_output.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── Auto-load bundled reference files so the user only uploads the INPUT file ──
if not st.session_state.config_saved:
    _bundled = load_bundled_reference()
    if _bundled is not None:
        _lookup, _token_lkp, _products, _svms, _train_df = _bundled
        st.session_state.master_lookup    = _lookup
        st.session_state.master_token_lkp = _token_lkp
        st.session_state.master_products  = _products
        st.session_state.svm_models       = _svms
        st.session_state.svm_trained      = True
        st.session_state.train_df         = _train_df
        st.session_state.api_keys         = {p: _load_env().get(p, '') for p in AI_PROVIDERS}
        st.session_state.config_saved     = True

# ── Tabs ─────────────────────────────────────
tab_config, tab_run, tab_results, tab_review, tab_analysis, tab_chatbot = st.tabs([
    "Config", "Run", "Results", "Review Queue", "Analysis", "AI Review Chatbot"
])

# ════════════════════════════════════════════
# TAB 1 — CONFIG
# ════════════════════════════════════════════
with tab_config:
    st.markdown("## Configuration")
    st.markdown("Set your file paths and API keys here.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📁 Reference Files")
        if st.session_state.config_saved and st.session_state.master_products:
            st.success(
                f"✅ Built-in reference data loaded — "
                f"{len(st.session_state.master_products)} products, SVMs trained. "
                "Just upload your input file in the **Run** tab."
            )
        st.caption("The Master DB, Product List and Training Data are built into the app. "
                   "You only need the uploaders below if you want to **override** them.")
        with st.expander("Optional — override built-in files", expanded=not st.session_state.config_saved):
            uf_master       = st.file_uploader("Comprehensive Master DB (.xlsx)",
                type=["xlsx"], key="uf_master",
                help="Overrides the built-in Veterinary_Product_Master_56_COMPREHENSIVE.xlsx")
            uf_product_list = st.file_uploader("Product List (.xlsx)",
                type=["xlsx"], key="uf_product_list",
                help="Overrides the built-in Product_List.xlsx")
            uf_training     = st.file_uploader("Training Data (.xlsx)",
                type=["xlsx"], key="uf_training",
                help="Overrides the built-in training data (SVMs + few-shot examples)")
            uf_labelled     = st.file_uploader("Labelled Output — extra training (.xlsx)",
                type=["xlsx"], key="uf_labelled",
                help="Can be the same file as Training Data")

    with col2:
        st.markdown("### 🔑 API Keys / Endpoints")
        saved = _load_env()
        _from_secrets = [p for p in AI_PROVIDERS if _secret(AI_PROVIDERS[p]['env_key'])]
        if _from_secrets:
            st.success(f"🔒 Loaded from Streamlit Secrets: {', '.join(_from_secrets)}")
        st.caption("On Streamlit Cloud, set keys in **Settings → Secrets** (persists across "
                   "sessions). Locally they save to .env. Fields below pre-fill from either.")

        key_inputs = {}

        st.markdown("**Claude (Anthropic)**")
        key_inputs['Claude'] = st.text_input("Anthropic API Key",
            value=saved.get('Claude',''), type="password",
            key="key_claude", help="sk-ant-...")
        if st.button("Save Claude Key", key="save_claude"):
            _save_env('Claude', key_inputs['Claude']); st.success("Saved")

        st.markdown("**GPT (OpenAI)**")
        key_inputs['GPT'] = st.text_input("OpenAI API Key",
            value=saved.get('GPT',''), type="password",
            key="key_gpt", help="sk-...")
        if st.button("Save GPT Key", key="save_gpt"):
            _save_env('GPT', key_inputs['GPT']); st.success("Saved")

        st.markdown("**Gemini (Google)**")
        key_inputs['Gemini'] = st.text_input("Google AI API Key",
            value=saved.get('Gemini',''), type="password",
            key="key_gemini", help="AIza...")
        if st.button("Save Gemini Key", key="save_gemini"):
            _save_env('Gemini', key_inputs['Gemini']); st.success("Saved")

        st.markdown("**Llama (Ollama / compatible API)**")
        key_inputs['Llama'] = st.text_input("Endpoint URL",
            value=saved.get('Llama','http://localhost:11434'),
            key="key_llama", help="Ollama default: http://localhost:11434 — no API key needed")
        st.caption("Uses the OpenAI-compatible /v1/chat/completions endpoint.")
        if st.button("Save Llama Endpoint", key="save_llama"):
            _save_env('Llama', key_inputs['Llama']); st.success("Saved")

        st.markdown("### ℹ️ About")
        st.markdown("""
        **Priority:** `REGEX (1.0) → MASTER (0.95) → SVM → AI`

        **AI cascade** *(when enabled):*
        Try providers in sidebar order. First result ≥ cascade threshold wins.
        If none meet the threshold, the highest-confidence result is used.
        Disable cascade to use majority vote across all enabled providers.

        **Pass 3 — AI (parallel):** All enabled providers run simultaneously per batch.
        Best result by confidence wins. Open-mode prompt used when REGEX+MASTER found nothing.

        **Web search fallback** *(optional):*
        Rows with product conf < 0.90 → DuckDuckGo search → primary AI re-classifies with web context.
        Source tagged as `WEB+CLAUDE` / `WEB+GPT` etc. Requires `pip install duckduckgo-search`.

        **ALL_ cell format:**
        `REGEX: val | MASTER: val | SVM: val(conf) | CLAUDE/GPT/...: val(conf)`

        **Review Queue:** SVM-sourced rows below confidence threshold.

        **3-sheet Excel output:** Full Results · Best Matches Only · Source Distribution
        """)

    st.markdown("---")
    if st.button("✅ Save Config & Load Models"):
        # Uploads are optional — fall back to the built-in bundled files when absent.
        has_upload = bool(uf_master or uf_training or uf_product_list or uf_labelled)
        if not has_upload and not (BUNDLED_MASTER.exists() and BUNDLED_TRAINING.exists()):
            st.error("No built-in files found and nothing uploaded. "
                     "Upload a Master DB and Training Data.")
        else:
            with st.spinner("Loading master DB and training SVMs…"):
                try:
                    master_bytes       = uf_master.read()       if uf_master       else BUNDLED_MASTER.read_bytes()
                    product_list_bytes = uf_product_list.read() if uf_product_list else (
                        BUNDLED_PRODUCTS.read_bytes() if BUNDLED_PRODUCTS.exists() else None)
                    train_bytes        = uf_training.read()     if uf_training     else BUNDLED_TRAINING.read_bytes()
                    labelled_bytes     = uf_labelled.read()     if uf_labelled     else train_bytes

                    lookup, token_lkp, master_products = load_master(master_bytes, product_list_bytes)
                    st.session_state.master_lookup    = lookup
                    st.session_state.master_token_lkp = token_lkp
                    st.session_state.master_products  = master_products

                    svm_models = train_svms(train_bytes, labelled_bytes)
                    st.session_state.svm_models  = svm_models
                    st.session_state.svm_trained = True
                    st.session_state.config_saved = True

                    train_df = pd.read_excel(io.BytesIO(train_bytes))
                    st.session_state.train_df = train_df

                    st.session_state.api_keys = {
                        p: key_inputs.get(p, saved.get(p,'')) for p in AI_PROVIDERS
                    }

                    trained = [k for k, v in svm_models.items() if v.trained]
                    st.success(f"✓ Master DB loaded: {len(master_products)} products · "
                               f"SVMs trained: {', '.join(trained)}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ════════════════════════════════════════════
# TAB 2 — RUN
# ════════════════════════════════════════════
with tab_run:
    st.markdown("## Run Pipeline")

    if not st.session_state.config_saved:
        st.warning("⚠️ Go to **Config** tab first and click **Save Config & Load Models**.")
    else:
        uploaded = st.file_uploader("Upload input file (.xlsx)", type=["xlsx"],
                                    help="Must have a 'Notes' column")
        if uploaded:
            input_df = pd.read_excel(uploaded)
            n_rows   = len(input_df)

            if 'Notes' not in input_df.columns:
                st.error("Input file must have a 'Notes' column.")
            else:
                # ── Cost & time estimate per enabled provider ──
                st.markdown("### 📊 Run Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows to classify", f"{n_rows:,}")
                if enabled_providers:
                    total_cost = sum(estimate_cost(n_rows, batch_size, p)['cost_usd']
                                     for p in enabled_providers)
                    total_mins = sum(estimate_cost(n_rows, batch_size, p)['est_mins']
                                     for p in enabled_providers)
                    total_tok  = estimate_cost(n_rows, batch_size, 'Claude')['total_tokens'] * len(enabled_providers)
                    c2.metric("Est. tokens",  f"{total_tok:,}")
                    c3.metric("Est. cost",    f"${round(total_cost,4)}")
                    c4.metric("Est. time",    f"{round(total_mins,1)} min")
                    st.caption("ℹ️ Cost estimate based on Claude Haiku pricing as baseline.")
                    if enabled_providers:
                        rows_info = []
                        for p, m in enabled_providers.items():
                            e = estimate_cost(n_rows, batch_size, p)
                            rows_info.append({'Provider': p, 'Model': m,
                                              'Est. Cost': f"${e['cost_usd']}",
                                              'Est. Time': f"{e['est_mins']} min"})
                        st.table(pd.DataFrame(rows_info))
                else:
                    c2.metric("Est. tokens", "—")
                    c3.metric("Est. cost",   "Free")
                    c4.metric("Est. time",   "< 1 min")
                    st.success(f"✅ Ready · REGEX + MASTER + SVM only — free, under 1 min for {n_rows:,} rows")

                st.markdown("### Methods active")
                method_cols = st.columns(2 + len(enabled_providers))
                method_cols[0].markdown("🟢 **REGEX**")
                method_cols[1].markdown("🟢 **MASTER**")
                if use_svm:
                    method_cols[1].markdown("🟢 **SVM**")
                for ci, (p, m) in enumerate(enabled_providers.items()):
                    method_cols[2+ci].markdown(f"🟢 **{p}**\n\n_{m}_")

                st.markdown("---")
                if st.button("🚀 Run Classification", type="primary"):
                    prog = st.progress(0)
                    stxt = st.empty()
                    result = run_pipeline(
                        input_df,
                        st.session_state.master_lookup,
                        st.session_state.master_token_lkp,
                        st.session_state.master_products,
                        st.session_state.svm_models,
                        st.session_state.train_df,
                        use_svm,
                        enabled_providers,
                        st.session_state.api_keys,
                        batch_size, prog, stxt,
                        fallback_threshold=fallback_threshold if use_cascade else 0.0,
                        use_web_search=use_web_search,
                    )
                    st.session_state.result_df    = result
                    st.session_state.review_df    = build_review_queue(result, conf_thresh)
                    st.session_state.overrides    = {}
                    st.session_state.run_complete = True
                    st.success(f"✅ Done! {len(result):,} rows classified · "
                               f"{len(st.session_state.review_df)} rows need review.")

# ════════════════════════════════════════════
# TAB 3 — RESULTS
# ════════════════════════════════════════════
with tab_results:
    st.markdown("## Results")
    if not st.session_state.run_complete:
        st.info("Run the pipeline first (▶️ Run tab).")
    else:
        df = st.session_state.result_df
        st.markdown(f"**{len(df):,} rows classified**")

        with st.expander("🔍 Filter", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            flt_product = fc1.multiselect("Product", sorted(df['BEST_Product'].dropna().unique()))
            flt_species = fc2.multiselect("Species", sorted(df['BEST_Species'].dropna().unique()))
            flt_source  = fc3.multiselect("Best source",
                sorted(df['BEST_Species_Source'].dropna().unique()))

        view = df.copy()
        if flt_product: view = view[view['BEST_Product'].isin(flt_product)]
        if flt_species: view = view[view['BEST_Species'].isin(flt_species)]
        if flt_source:  view = view[view['BEST_Species_Source'].isin(flt_source)]

        show_cols = ['Notes','BEST_Product','BEST_Species','BEST_Species_Source',
                     'BEST_Size','BEST_Weight','BEST_Pack_Size',
                     'BEST_Species_Conf','BEST_Size_Conf',
                     'BEST_Weight_Conf','MATCH_TYPE']
        show_cols = [c for c in show_cols if c in view.columns]
        st.dataframe(view[show_cols], use_container_width=True, height=500)
        st.caption(f"Showing {len(view):,} of {len(df):,} rows")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            excel_bytes = to_excel_bytes(df)
            st.download_button("⬇ Download Full Excel (3 sheets)", excel_bytes,
                "zoetis_output.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col_d2:
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv_bytes, "zoetis_output.csv", "text/csv")

# ════════════════════════════════════════════
# TAB 4 — REVIEW QUEUE
# ════════════════════════════════════════════
with tab_review:
    st.markdown("## Review Queue")
    st.markdown("Rows where **source = SVM** and **confidence < threshold**. "
                "Review each row and override if needed.")

    if not st.session_state.run_complete:
        st.info("Run the pipeline first (▶️ Run tab).")
    else:
        rdf = st.session_state.review_df
        if len(rdf) == 0:
            st.success("✅ No rows need review — all fields resolved with REGEX or MASTER.")
        else:
            st.warning(f"**{len(rdf)} rows** flagged for review (SVM + conf < {conf_thresh})")

            FIELDS    = ['Species','Size','Weight','Pack_Size','Pipettes']
            BEST_COLS = {f: f'BEST_{f}'        for f in FIELDS}
            SRC_COLS  = {f: f'BEST_{f}_Source' for f in FIELDS}
            CONF_COLS = {f: f'BEST_{f}_Conf'   for f in FIELDS}
            ALL_COLS  = {f: f'ALL_{f}'         for f in FIELDS if f'ALL_{f}' in rdf.columns}

            overrides = st.session_state.overrides

            for idx, row in rdf.iterrows():
                note = str(row['Notes'])[:120]
                with st.expander(f"Row {idx+1} — {note}", expanded=False):
                    st.markdown(f"**Full note:** `{row['Notes']}`")
                    st.markdown(f"**Product:** {row.get('BEST_Product','—')} "
                                f"({row.get('BEST_Product_Source','—')})")

                    review_cols = st.columns(len(FIELDS))
                    for ci, field in enumerate(FIELDS):
                        bc = BEST_COLS[field]; sc = SRC_COLS[field]
                        cc = CONF_COLS[field]; ac = ALL_COLS.get(field)
                        with review_cols[ci]:
                            cur_val  = str(row.get(bc,'') or 'UNKNOWN')
                            cur_src  = str(row.get(sc,'') or '—')
                            cur_conf = float(row.get(cc, 0) or 0)
                            all_vals = str(row.get(ac,'') or '') if ac else ''

                            badge_cls = f'badge-{cur_src.lower()}' if cur_src.lower() in \
                                ('regex','master','svm','claude','gpt','gemini','llama') else 'badge-unknown'
                            st.markdown(f"**{field}**  "
                                        f"<span class='{badge_cls}'>{cur_src}</span>",
                                        unsafe_allow_html=True)
                            st.markdown(f"Current: `{cur_val}` ({cur_conf:.2f})")
                            if all_vals:
                                st.caption(all_vals)

                            key_id   = f"override_{idx}_{field}"
                            override = st.text_input(
                                "Override value", value="", key=key_id,
                                placeholder=f"e.g. {cur_val}",
                                label_visibility="collapsed")
                            if override.strip():
                                overrides[f"{idx}_{field}"] = override.strip().upper()

            st.session_state.overrides = overrides

            if st.button("✅ Apply Overrides & Export"):
                result_df = st.session_state.result_df.copy()
                applied = 0
                for key, val in overrides.items():
                    parts = key.split('_', 1)
                    if len(parts) == 2:
                        try:
                            row_idx = int(parts[0]); field = parts[1]
                            bc = f'BEST_{field}'
                            if bc in result_df.columns and row_idx in result_df.index:
                                result_df.at[row_idx, bc]                   = val
                                result_df.at[row_idx, f'BEST_{field}_Source'] = 'ANNOTATOR'
                                result_df.at[row_idx, f'BEST_{field}_Conf']   = 1.0
                                applied += 1
                        except Exception:
                            pass
                st.session_state.result_df = result_df
                excel_bytes = to_excel_bytes(result_df)
                st.success(f"✅ {applied} overrides applied.")
                st.download_button("⬇ Download Annotated Excel", excel_bytes,
                    "zoetis_annotated.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ════════════════════════════════════════════
# TAB 5 — ANALYSIS
# ════════════════════════════════════════════
with tab_analysis:
    st.markdown("## Analysis")
    if not st.session_state.run_complete:
        st.info("Run the pipeline first (▶️ Run tab).")
    else:
        df = st.session_state.result_df
        n  = len(df)

        # ── KPI cards ──
        st.markdown("### Coverage")
        kpi_cols = st.columns(6)
        for ci, f in enumerate(['Product','Species','Size','Weight','Pack_Size','Pipettes']):
            col_name = f'BEST_{f}'
            if col_name in df.columns:
                resolved = (df[col_name] != 'UNKNOWN').sum()
                pct = round(resolved / n * 100, 1)
                kpi_cols[ci].markdown(
                    f'<div class="metric-card"><h4>{f}</h4><p>{pct}%</p>'
                    f'<small>{resolved:,}/{n:,}</small></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Source distribution charts ──
        st.markdown("### Method contribution per field")
        src_fields = {
            'Product':   'BEST_Product_Source',
            'Species':   'BEST_Species_Source',
            'Size':      'BEST_Size_Source',
            'Weight':    'BEST_Weight_Source',
            'Pack Size': 'BEST_Pack_Size_Source',
        }
        colour_map = {
            'REGEX':'#00A591','MASTER':'#4B286D','SVM':'#F5A623',
            'CLAUDE':'#E84E1B','GPT':'#10A37F','GEMINI':'#4285F4',
            'LLAMA':'#7C3AED','UNKNOWN':'#CCCCCC','ANNOTATOR':'#2196F3',
        }
        chart_cols = st.columns(len(src_fields))
        for ci, (label, col) in enumerate(src_fields.items()):
            if col in df.columns:
                vc = df[col].value_counts().reset_index()
                vc.columns = ['Source','Count']
                vc['Pct'] = (vc['Count'] / n * 100).round(1)
                fig = px.pie(vc, values='Count', names='Source',
                             title=label, hole=0.45,
                             color='Source', color_discrete_map=colour_map)
                fig.update_traces(textinfo='percent', textfont_size=11)
                fig.update_layout(margin=dict(t=35,b=5,l=5,r=5),
                                  showlegend=True, height=250,
                                  legend=dict(font=dict(size=9)))
                chart_cols[ci].plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── UNKNOWN rate ──
        st.markdown("### UNKNOWN rate by field")
        unk_data = []
        for f in ['Product','Species','Size','Weight','Pack_Size','Pipettes']:
            col = f'BEST_{f}'
            if col in df.columns:
                unk = (df[col] == 'UNKNOWN').sum()
                unk_data.append({'Field':f,'Unknown':unk,'Resolved':n-unk})
        fig2 = px.bar(pd.DataFrame(unk_data), x='Field', y=['Resolved','Unknown'],
                      color_discrete_map={'Resolved':'#00A591','Unknown':'#E84E1B'},
                      title="Resolved vs UNKNOWN", barmode='stack',
                      labels={'value':'Rows','variable':'Status'})
        fig2.update_layout(height=320, margin=dict(t=40,b=20))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # ── Top products + species ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Top 15 Products")
            top_prod = df['BEST_Product'].value_counts().head(15).reset_index()
            top_prod.columns = ['Product','Count']
            fig3 = px.bar(top_prod, x='Count', y='Product', orientation='h',
                          color_discrete_sequence=['#4B286D'])
            fig3.update_layout(height=420, yaxis={'categoryorder':'total ascending'},
                               margin=dict(t=20,b=20))
            st.plotly_chart(fig3, use_container_width=True)
        with col_b:
            st.markdown("### Species breakdown")
            sp_vc = df['BEST_Species'].value_counts().reset_index()
            sp_vc.columns = ['Species','Count']
            fig4 = px.pie(sp_vc, values='Count', names='Species', hole=0.4,
                          color_discrete_sequence=['#4B286D','#00A591','#F5A623',
                                                   '#E84E1B','#9B59B6','#2ECC71','#E67E22'])
            fig4.update_layout(height=420, margin=dict(t=20,b=20))
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # ── Confidence distribution ──
        st.markdown("### Confidence distribution")
        conf_fields = [f for f in ['BEST_Species_Conf','BEST_Size_Conf',
                       'BEST_Weight_Conf','BEST_Pack_Size_Conf'] if f in df.columns]
        conf_melt = df[conf_fields].melt(var_name='Field', value_name='Confidence')
        conf_melt['Field'] = conf_melt['Field'].str.replace('BEST_','').str.replace('_Conf','')
        fig5 = px.box(conf_melt, x='Field', y='Confidence', color='Field',
                      color_discrete_sequence=['#00A591','#4B286D','#F5A623','#E84E1B'])
        fig5.update_layout(height=320, showlegend=False, margin=dict(t=20,b=20))
        fig5.add_hline(y=conf_thresh, line_dash='dash', line_color='red',
                       annotation_text=f'Review threshold ({conf_thresh})')
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("---")
        st.markdown("### Raw source distribution table")
        src_rows = []
        for label, col in src_fields.items():
            if col in df.columns:
                for src, cnt in df[col].value_counts().items():
                    src_rows.append({'Field':label,'Source':src,
                                     'Count':cnt,'Coverage':f'{cnt/n*100:.1f}%'})
        st.dataframe(pd.DataFrame(src_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# TAB 6 — AI CHATBOT REVIEW
# ════════════════════════════════════════════
with tab_chatbot:
    st.markdown("## AI Review Chatbot")
    st.caption("Each flagged record is re-classified independently by Claude using your full system prompt and reference documents.")

    if not st.session_state.run_complete:
        st.info("Run the pipeline first (Run tab).")
    else:
        # ── Resolve Claude API key ──
        _saved_keys = _load_env()
        _api_keys   = st.session_state.get("api_keys") or {}
        _claude_key = _api_keys.get("Claude") or _saved_keys.get("Claude") or ""

        if not _claude_key:
            st.warning("Claude API key required for the chatbot. Add it in the Config tab.")
        else:
            # ── Reviewer name + chatbot model ──
            _rc1, _rc2 = st.columns([2, 2])
            _reviewer = _rc1.text_input(
                "Reviewer name",
                value=st.session_state.get("reviewer_name",""),
                key="chatbot_reviewer_input",
                placeholder="Your name (logged with each correction)",
            )
            st.session_state.reviewer_name = _reviewer

            _claude_models = AI_PROVIDERS['Claude']['models']
            _cur_model     = st.session_state.get("chatbot_model", "claude-sonnet-4-6")
            _mdl_idx       = _claude_models.index(_cur_model) if _cur_model in _claude_models else 0
            _chosen_model  = _rc2.selectbox(
                "Chatbot model",
                _claude_models,
                index=_mdl_idx,
                key="chatbot_model_select",
                help="Model used to independently re-classify each flagged record.",
            )
            st.session_state.chatbot_model = _chosen_model

            # ── Build system prompt once per session ──
            if not st.session_state.get("chatbot_sys_prompt"):
                st.session_state.chatbot_sys_prompt = _build_chatbot_sys_prompt()

            # ── Gather flagged rows ──
            _df       = st.session_state.result_df
            _flagged  = _chatbot_flagged_rows(_df)
            _n_total  = len(_df)
            _n_flagged = len(_flagged)
            _pct      = round(_n_flagged / _n_total * 100, 1) if _n_total else 0

            _corrections = st.session_state.corrections_log
            _n_reviewed  = len(_corrections)
            _n_confirmed = sum(1 for c in _corrections if c.get("Changed") == "NO")
            _n_corrected = sum(1 for c in _corrections if c.get("Changed") in ("YES","PARTIAL"))

            # ── Summary ──
            _sc1, _sc2 = st.columns([3, 3])
            _sc1.warning(f"**{_n_flagged} records need review** out of {_n_total} ({_pct}%)")
            if _n_reviewed:
                _sc2.markdown(
                    f"✅ **{_n_reviewed}** reviewed &nbsp;|&nbsp; "
                    f"**{_n_confirmed}** confirmed same &nbsp;|&nbsp; **{_n_corrected}** corrected"
                )

            # ── Download corrections ──
            if _corrections:
                _cb = io.BytesIO()
                pd.DataFrame(_corrections).to_excel(_cb, index=False)
                _cb.seek(0)
                st.download_button(
                    "📥 Download Corrections", _cb.getvalue(), "corrections_log.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.markdown("---")

            if _n_flagged == 0:
                st.success("No records flagged for review.")
            else:
                _reviewed_set = {c.get("Original_Row") for c in _corrections}
                _active_row   = st.session_state.get("active_chat_row")

                # ── Queue table header ──
                _hcols = st.columns([1, 5, 2, 1, 3, 2])
                for _hc, _ht in zip(_hcols, ["Row","Note","Pipeline","Conf","Reason",""]):
                    _hc.markdown(f"**{_ht}**")
                st.markdown('<hr style="margin:2px 0 6px 0"/>', unsafe_allow_html=True)

                # ── Queue rows ──
                for _ridx, _row, _flag_reason in _flagged:
                    _already  = _ridx in _reviewed_set
                    _is_open  = (_ridx == _active_row)
                    _note_pre = str(_row.get("Notes",""))[:72]
                    _prod     = str(_row.get("BEST_Product","UNKNOWN"))
                    _conf_val = float(_row.get("BEST_Product_Conf",0) or 0)

                    _rc = st.columns([1, 5, 2, 1, 3, 2])
                    _rc[0].write(f"{_ridx+1}")
                    _rc[1].write(_note_pre)
                    _rc[2].write(_prod)
                    _rc[3].write(f"{_conf_val*100:.0f}%")
                    _rc[4].write(_flag_reason)

                    if _already:
                        _rc[5].markdown("✅ Done")
                    elif _is_open:
                        if _rc[5].button("Close", key=f"close_chat_{_ridx}"):
                            st.session_state.active_chat_row = None
                            st.rerun()
                    else:
                        if _rc[5].button("Review", key=f"rev_chat_{_ridx}"):
                            st.session_state.active_chat_row = _ridx
                            if _ridx not in st.session_state.chatbot_sessions:
                                st.session_state.chatbot_sessions[_ridx] = []
                            st.rerun()

                # ── Chat panel ──
                _active_row = st.session_state.get("active_chat_row")
                if _active_row is not None:
                    _flagged_idxs = {r[0] for r in _flagged}
                    if _active_row not in _flagged_idxs:
                        st.session_state.active_chat_row = None
                    else:
                        _rd          = _df.loc[_active_row]
                        _note        = str(_rd.get("Notes",""))
                        _best_prod   = str(_rd.get("BEST_Product","UNKNOWN"))
                        _best_conf   = float(_rd.get("BEST_Product_Conf",0) or 0)
                        _all_prod    = str(_rd.get("ALL_Product","") or "")
                        _ai_reason   = str(_rd.get("AI_Reasoning","") or "Not available")
                        _flag_rsn    = next((r[2] for r in _flagged if r[0]==_active_row),"")

                        st.markdown("---")
                        st.markdown(f"### Reviewing Row {_active_row+1}")
                        _ic1, _ic2, _ic3 = st.columns(3)
                        _ic1.markdown(f"**Note**\n\n`{_note}`")
                        _ic2.markdown(f"**Pipeline**\n\n{_best_prod} ({_best_conf*100:.0f}%)")
                        _ic3.markdown(f"**Flagged because**\n\n{_flag_rsn}")
                        st.markdown("---")

                        _chat_hist = st.session_state.chatbot_sessions.get(_active_row, [])

                        # ── Auto-fire first message if empty ──
                        if len(_chat_hist) == 0:
                            _first = (
                                f'Note to classify: "{_note}"\n\n'
                                f"Pipeline predicted: {_best_prod} ({_best_conf*100:.0f}% confidence)\n"
                                f"Pipeline reasoning: {_ai_reason}\n"
                                f"Reason flagged: {_flag_rsn}\n\n"
                                "Please read the reference documents in your system prompt carefully and "
                                "give your INDEPENDENT classification. Do NOT be influenced by the pipeline "
                                "prediction. Reason from the documents only.\n\n"
                                "Propose your classification for all 6 fields "
                                "(Product, Species, Size, Weight, Pack_Size, Pipettes) "
                                "and ask the reviewer to confirm."
                            )
                            with st.spinner("Analysing note..."):
                                _reply = _call_chatbot_api(
                                    [{"role":"user","content":_first}], _claude_key
                                )
                            _chat_hist = [
                                {"role":"user",      "content":_first,  "_hidden":True},
                                {"role":"assistant", "content":_reply},
                            ]
                            st.session_state.chatbot_sessions[_active_row] = _chat_hist
                            st.rerun()

                        # ── Render messages ──
                        for _msg in _chat_hist:
                            if _msg.get("_hidden"):
                                continue
                            if _msg["role"] == "assistant":
                                with st.chat_message("assistant", avatar="🤖"):
                                    st.markdown(_msg["content"])
                            else:
                                with st.chat_message("user"):
                                    st.markdown(_msg["content"])

                        # ── Confirm / Skip buttons ──
                        _bc1, _bc2, _ = st.columns([2, 2, 8])
                        _confirmed = _bc1.button("✅ Confirm", key=f"confirm_{_active_row}", type="primary")
                        _skipped   = _bc2.button("⏭ Skip",    key=f"skip_{_active_row}")

                        # ── Free-text input ──
                        _user_in = st.chat_input("Type a message or correction...", key=f"chat_in_{_active_row}")

                        # ── Handle CONFIRM ──
                        if _confirmed:
                            _api_msgs = [{"role":m["role"],"content":m["content"]} for m in _chat_hist]
                            _api_msgs.append({"role":"user","content":
                                'Output your final confirmed classification as JSON only '
                                '— no markdown fences, no other text:\n'
                                '{"Product":"...","Species":"...","Size":"...","Weight":"...",'
                                '"Pack_Size":"...","Pipettes":"...","Reasoning":"one sentence"}'
                            })
                            with st.spinner("Saving..."):
                                _json_raw = _call_chatbot_api(_api_msgs, _claude_key, max_tokens=250)
                            try:
                                _clean = re.sub(r'```(?:json)?|```','',_json_raw).strip()
                                _ext   = json.loads(_clean)
                            except Exception:
                                _ext = {
                                    "Product":_best_prod,"Species":"UNKNOWN","Size":"UNKNOWN",
                                    "Weight":"UNKNOWN","Pack_Size":"UNKNOWN","Pipettes":"UNKNOWN",
                                    "Reasoning":"Parse error — check manually",
                                }

                            _cp      = str(_ext.get("Product","") or "").strip().upper() or "UNKNOWN"
                            _changed = _determine_changed(_cp, _best_prod, _all_prod)

                            _corr = {
                                "Timestamp":           datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Reviewer":            _reviewer,
                                "Original_Row":        _active_row,
                                "Notes":               _note,
                                "Pipeline_Product":    _best_prod,
                                "Pipeline_Confidence": round(_best_conf, 3),
                                "Reviewed_Product":    _cp,
                                "Reviewed_Species":    str(_ext.get("Species","") or "UNKNOWN").upper(),
                                "Reviewed_Size":       str(_ext.get("Size","")    or "UNKNOWN").upper(),
                                "Reviewed_Weight":     str(_ext.get("Weight","")  or "UNKNOWN").upper(),
                                "Reviewed_Pack_Size":  str(_ext.get("Pack_Size","")or"UNKNOWN").upper(),
                                "Reviewed_Pipettes":   str(_ext.get("Pipettes","")or"UNKNOWN").upper(),
                                "Changed":             _changed,
                                "Chat_Reasoning":      str(_ext.get("Reasoning",""))[:300],
                            }
                            st.session_state.corrections_log.append(_corr)
                            _save_correction_to_disk(_corr)

                            # Apply confirmed values back to result_df
                            _rdf = st.session_state.result_df
                            for _fld, _ckey in [
                                ("Product","Reviewed_Product"), ("Species","Reviewed_Species"),
                                ("Size","Reviewed_Size"),       ("Weight","Reviewed_Weight"),
                                ("Pack_Size","Reviewed_Pack_Size"), ("Pipettes","Reviewed_Pipettes"),
                            ]:
                                _bc_col = f"BEST_{_fld}"
                                if _bc_col in _rdf.columns:
                                    _rdf.at[_active_row, _bc_col]               = _corr[_ckey]
                                    _rdf.at[_active_row, f"BEST_{_fld}_Source"] = "ANNOTATOR"
                                    _rdf.at[_active_row, f"BEST_{_fld}_Conf"]   = 1.0
                            st.session_state.result_df = _rdf
                            st.session_state.active_chat_row = None
                            st.success(f"Saved! Changed status: **{_changed}**")
                            st.rerun()

                        # ── Handle SKIP ──
                        if _skipped:
                            st.session_state.active_chat_row = None
                            st.rerun()

                        # ── Handle user message ──
                        if _user_in:
                            _api_msgs = [{"role":m["role"],"content":m["content"]} for m in _chat_hist]
                            _api_msgs.append({"role":"user","content":_user_in})
                            with st.spinner("Thinking..."):
                                _reply = _call_chatbot_api(_api_msgs, _claude_key)
                            _chat_hist.append({"role":"user",      "content":_user_in})
                            _chat_hist.append({"role":"assistant", "content":_reply})
                            st.session_state.chatbot_sessions[_active_row] = _chat_hist
                            st.rerun()
