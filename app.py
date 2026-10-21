import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import html
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
import faiss

from transformers.models.clip.processing_clip import CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "image_embeddings_full.npy")
IMAGE_IDS_PATH = os.path.join(BASE_DIR, "image_ids_full.npy")
METADATA_PATH = os.path.join(BASE_DIR, "polyvore_item_metadata.json")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

CATEGORIES = ["outerwear", "tops", "bottoms", "shoes", "bags"]

EXAMPLES = {
    "Weekend shopping": "A classy but comfortable outfit for shopping. Neutral and pastel colors, nothing too tight.",
    "Dinner date": "Elegant evening look for a dinner date, dark tones, a little edgy.",
    "Office casual": "Smart casual outfit for the office, navy and white, relaxed fit.",
    "Street style": "Oversized streetwear look in black and olive for a city walk.",
}

st.set_page_config(page_title="Fashion Stylist", page_icon="👗", layout="wide")


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:opsz,wght@6..96,500;6..96,600&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --ink: #1b1826;
  --muted: #6c687a;
  --paper: #f1eff5;
  --card: #ffffff;
  --line: #dcd8e6;
  --accent: #2f3fd8;
  --accent-soft: #e4e7fb;
}

html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 4rem; }

h1.hero {
  font-family: 'Bodoni Moda', serif; font-weight: 500;
  font-size: clamp(2.4rem, 5vw, 4rem); line-height: 1.05;
  letter-spacing: -0.01em; margin: 0 0 .6rem 0;
}
p.lede { color: var(--muted); font-size: 1.05rem; max-width: 46ch; margin: 0 0 1.6rem 0; }

h3.section {
  font-family: 'Bodoni Moda', serif; font-weight: 500; font-size: 1.6rem;
  margin: 2.4rem 0 .8rem 0;
}

.chip {
  display: inline-block; padding: .28rem .8rem; margin: 0 .4rem .4rem 0;
  border-radius: 999px; background: var(--accent-soft); color: var(--accent);
  font-size: .85rem; font-weight: 500;
}
.chip-label { color: var(--muted); font-size: .85rem; margin-right: .6rem; }
.chip-row { margin-bottom: .35rem; }

.cat-name { font-weight: 700; font-size: 1rem; margin-bottom: .5rem; }
.item-title { font-weight: 500; font-size: .95rem; line-height: 1.3; margin: .55rem 0 .15rem 0; }
.item-meta { color: var(--muted); font-size: .8rem; }
.match { display: inline-block; margin-top: .35rem; font-size: .78rem; color: var(--accent); font-weight: 500; }

[data-testid="stImage"] img { border-radius: 10px; border: 1px solid var(--line); background: #fff; }


div.stButton > button {
  border-radius: 999px; border: 1px solid var(--line); background: var(--card);
  color: var(--ink); font-weight: 500; padding: .35rem 1rem;
}
div.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
div.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
div.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; padding: .55rem 1.6rem; }
div.stButton > button[kind="primary"]:hover { background: #2432b3; color: #fff; }

[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--card); }
[data-testid="stSidebar"] { background: #e8e5ef; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp li,
.stApp div[data-testid="stMarkdownContainer"], .stApp h1, .stApp h2, .stApp h3 { color: var(--ink); }
.stApp .chip { color: var(--accent); }
.stApp .item-meta, .stApp .chip-label, .stApp p.lede { color: var(--muted); }
.stApp .match { color: var(--accent); }
div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] * { color: #fff !important; }
.no-img {
  aspect-ratio: 1 / 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  border: 1px dashed var(--line); border-radius: 10px; background: var(--card);
  color: var(--muted); font-size: .9rem; text-align: center;
}
.no-img span { font-size: .72rem; }
footer, #MainMenu { visibility: hidden; }

.stApp [data-testid="stTextArea"] > div,
.stApp [data-baseweb="textarea"],
.stApp [data-baseweb="base-input"] {
  background-color: #ffffff !important;
  border-radius: 12px !important;
  border: 1px solid var(--line) !important;
}
.stApp [data-baseweb="textarea"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--accent-soft) !important;
}
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-baseweb="textarea"] textarea,
.stApp [data-baseweb="base-input"] textarea {
  background-color: #ffffff !important;
  color: #1b1826 !important;
  -webkit-text-fill-color: #1b1826 !important;
  caret-color: #1b1826 !important;
  opacity: 1 !important;
  font-size: 1.02rem !important;
}
.stApp [data-testid="stTextArea"] textarea::placeholder,
.stApp [data-baseweb="textarea"] textarea::placeholder {
  color: #4a4658 !important;
  -webkit-text-fill-color: #4a4658 !important;
  opacity: 1 !important;
}
</style>

""",
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND (unchanged logic)
# ============================================================

@st.cache_resource
def load_data():
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    image_ids = np.load(IMAGE_IDS_PATH, allow_pickle=True)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata_df = pd.DataFrame.from_dict(metadata, orient="index")
    metadata_df.index = metadata_df.index.astype(str)
    return embeddings, image_ids, metadata_df


@st.cache_resource
def build_faiss_index(_embeddings):
    dimension = _embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 64
    total = _embeddings.shape[0]
    batch_size = 5000
    progress = st.progress(0, text="Building search index...")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        index.add(np.asarray(_embeddings[start:end], dtype=np.float32))
        progress.progress(int(end / total * 100), text=f"Building search index... {end:,}/{total:,}")
    progress.empty()
    return index


@st.cache_resource
def load_fashion_clip():
    processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL)
    model = CLIPModel.from_pretrained(FASHION_CLIP_MODEL).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    return tokenizer, model


def get_text_embeddings(queries, processor, model):
    inputs = processor(text=queries, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        feats = model.text_projection(out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def extract_preferences(user_request, tokenizer, model):
    prompt = f"""
You are a fashion recommendation assistant.

Extract the user's fashion preferences from the request below.

Return ONLY valid JSON using exactly this structure:

{{
    "styles": [],
    "colors": [],
    "occasion": "",
    "clothing_preferences": []
}}

User request:
{user_request}
"""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    response = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"styles": [], "colors": [], "occasion": "", "clothing_preferences": []}


def build_category_queries(p):
    base = " ".join(
        part
        for part in [
            ", ".join(p.get("styles", [])),
            ", ".join(p.get("colors", [])),
            p.get("occasion", ""),
            ", ".join(p.get("clothing_preferences", [])),
        ]
        if part
    )
    return {
        "outerwear": f"{base} fashionable outerwear jacket coat",
        "tops": f"{base} fashionable top shirt blouse",
        "bottoms": f"{base} fashionable pants trousers skirt",
        "shoes": f"{base} fashionable shoes",
        "bags": f"{base} fashionable handbag bag",
    }


def retrieve_category_items(query_embedding, category, faiss_index, image_ids, metadata_df, top_k=3):
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    similarities, positions = faiss_index.search(query, 500)
    results = []
    for similarity, position in zip(similarities[0], positions[0]):
        if position < 0:
            continue
        item_id = str(image_ids[position])
        if item_id not in metadata_df.index:
            continue
        row = metadata_df.loc[item_id]
        if row.get("semantic_category", "") != category:
            continue
        results.append(
            {
                "item_id": item_id,
                "category": category,
                "similarity": float(similarity),
                "title": row.get("title", ""),
                "description": row.get("description", ""),
            }
        )
        if len(results) >= top_k:
            break
    return results


# ============================================================
# UI HELPERS
# ============================================================

def chips(label, values):
    if isinstance(values, str):
        values = [values] if values else []
    inner = "".join(f'<span class="chip">{html.escape(str(v))}</span>' for v in values)
    inner = inner or '<span class="item-meta">none detected</span>'
    st.markdown(
        f'<div class="chip-row"><span class="chip-label">{label}</span>{inner}</div>',
        unsafe_allow_html=True,
    )


def find_image(item_id):
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = os.path.join(IMAGE_DIR, f"{item_id}.{ext}")
        if os.path.exists(p):
            return p
    return None


def show_item(item, large=True):
    path = find_image(item["item_id"])
    if path:
        st.image(path, use_container_width=True)
    else:
        st.markdown(
            f'<div class="no-img">No image<br><span>id {html.escape(item["item_id"])}</span></div>',
            unsafe_allow_html=True,
        )
    title = html.escape(str(item.get("title") or "Untitled item"))
    st.markdown(f'<div class="item-title">{title}</div>', unsafe_allow_html=True)
    if large and item.get("description"):
        st.markdown(
            f'<div class="item-meta">{html.escape(str(item["description"])[:110])}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<span class="match">Match {item["similarity"] * 100:.0f}%</span>',
        unsafe_allow_html=True,
    )


def set_example(text):
    st.session_state["request"] = text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Options per category", 1, 5, 3)
    st.caption(f"Running on {DEVICE.upper()}")
    st.caption("FashionCLIP + FAISS + Qwen2.5 preference extraction")


# ============================================================
# HERO + INPUT
# ============================================================

st.markdown('<h1 class="hero">Describe the look.<br>We\'ll build the outfit.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="lede">Tell the stylist the occasion, colors and fit you like. '
    "It picks a jacket, top, bottom, shoes and bag that go together.</p>",
    unsafe_allow_html=True,
)

user_request = st.text_area(
    "What are you looking for?",
    key="request",
    height=110,
    label_visibility="collapsed",
    placeholder="Example: a classy but comfortable outfit for shopping, neutral and pastel colors, nothing too tight.",
)

cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    col.button(label, key=f"ex_{label}", on_click=set_example, args=(text,), use_container_width=True)

go = st.button("Find my outfit", type="primary")


# ============================================================
# RUN
# ============================================================

if go:
    if not user_request.strip():
        st.warning("Describe the outfit you want, or pick an example above.")
        st.stop()

    with st.spinner("Finding your outfit..."):
        embeddings, image_ids, metadata_df = load_data()
        faiss_index = build_faiss_index(embeddings)
        processor, clip_model = load_fashion_clip()
        tokenizer, llm_model = load_llm()
        preferences = extract_preferences(user_request, tokenizer, llm_model)
        queries = build_category_queries(preferences)
        embs = get_text_embeddings(list(queries.values()), processor, clip_model)
        recs = {
            cat: retrieve_category_items(e, cat, faiss_index, image_ids, metadata_df, top_k=top_k)
            for cat, e in zip(queries.keys(), embs)
        }

    # --- what the stylist understood
    st.markdown('<h3 class="section">What the stylist understood</h3>', unsafe_allow_html=True)
    chips("Styles", preferences.get("styles", []))
    chips("Colors", preferences.get("colors", []))
    chips("Occasion", preferences.get("occasion", ""))
    chips("Wants", preferences.get("clothing_preferences", []))

    # --- the look: best match per category
    st.markdown('<h3 class="section">Your look</h3>', unsafe_allow_html=True)
    look_cols = st.columns(5, gap="medium")
    for col, cat in zip(look_cols, CATEGORIES):
        with col:
            st.markdown(f'<div class="cat-name">{cat.title()}</div>', unsafe_allow_html=True)
            items = recs.get(cat, [])
            if items:
                show_item(items[0])
            else:
                st.caption("No match found. Try describing this piece in your request.")

    # --- alternatives
    if top_k > 1:
        st.markdown('<h3 class="section">Other options</h3>', unsafe_allow_html=True)
        for cat in CATEGORIES:
            alts = recs.get(cat, [])[1:]
            if not alts:
                continue
            with st.expander(f"{cat.title()} ({len(alts)} more)"):
                alt_cols = st.columns(max(len(alts), 3), gap="medium")
                for col, item in zip(alt_cols, alts):
                    with col:
                        show_item(item, large=False)
