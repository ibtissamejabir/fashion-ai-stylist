# ============================================================
# IMPORTANT:
# Set this BEFORE importing numpy, torch, faiss, etc.
# ============================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "image_embeddings_full.npy"
)

IMAGE_IDS_PATH = os.path.join(
    BASE_DIR,
    "image_ids_full.npy"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "polyvore_item_metadata.json"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images"
)

FASHION_CLIP_MODEL = (
    "patrickjohncyh/fashion-clip"
)

LLM_MODEL = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)

DEVICE = "cpu"


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Multimodal AI Fashion Stylist",
    page_icon="👗",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r"
    )

    image_ids = np.load(
        IMAGE_IDS_PATH,
        allow_pickle=True
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    metadata_df = pd.DataFrame.from_dict(
        metadata,
        orient="index"
    )

    metadata_df.index = (
        metadata_df.index.astype(str)
    )

    return (
        embeddings,
        image_ids,
        metadata_df
    )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

@st.cache_resource
def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dimension,
        16,
        faiss.METRIC_INNER_PRODUCT
    )

    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 64

    batch_size = 5000

    total = embeddings.shape[0]

    progress = st.progress(
        0,
        text="Building FAISS index..."
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = np.asarray(
            embeddings[start:end],
            dtype=np.float32
        )

        index.add(batch)

        progress.progress(
            int(
                end / total * 100
            ),
            text=(
                f"Building FAISS index... "
                f"{end:,}/{total:,}"
            )
        )

        del batch

    progress.empty()

    return index


# ============================================================
# LOAD FASHIONCLIP
# ============================================================

@st.cache_resource
def load_fashion_clip():

    processor = (
        CLIPProcessor.from_pretrained(
            FASHION_CLIP_MODEL
        )
    )

    model = (
        CLIPModel.from_pretrained(
            FASHION_CLIP_MODEL
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return processor, model


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            LLM_MODEL
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float32
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return tokenizer, model


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

def get_text_embeddings(
    queries,
    processor,
    model
):

    text_inputs = processor(
        text=queries,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    text_inputs = {
        key: value.to(DEVICE)
        for key, value in text_inputs.items()
    }

    with torch.no_grad():

        text_outputs = (
            model.text_model(
                input_ids=(
                    text_inputs["input_ids"]
                ),
                attention_mask=(
                    text_inputs["attention_mask"]
                )
            )
        )

        text_pooled = (
            text_outputs.pooler_output
        )

        text_features = (
            model.text_projection(
                text_pooled
            )
        )

        text_features = (
            text_features
            /
            text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

    return (
        text_features
        .cpu()
        .numpy()
    )


# ============================================================
# LLM PREFERENCE EXTRACTION
# ============================================================

def extract_preferences(
    user_request,
    tokenizer,
    model
):

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

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    response = (
        response
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        preferences = json.loads(
            response
        )

    except json.JSONDecodeError:

        preferences = {
            "styles": [],
            "colors": [],
            "occasion": "",
            "clothing_preferences": []
        }

    return preferences


# ============================================================
# BUILD CATEGORY QUERIES
# ============================================================

def build_category_queries(
    preferences
):

    styles = ", ".join(
        preferences.get(
            "styles",
            []
        )
    )

    colors = ", ".join(
        preferences.get(
            "colors",
            []
        )
    )

    occasion = preferences.get(
        "occasion",
        ""
    )

    clothing_preferences = ", ".join(
        preferences.get(
            "clothing_preferences",
            []
        )
    )

    base = " ".join(
        part
        for part in [
            styles,
            colors,
            occasion,
            clothing_preferences
        ]
        if part
    )

    return {

        "outerwear":
            f"{base} fashionable outerwear jacket coat",

        "tops":
            f"{base} fashionable top shirt blouse",

        "bottoms":
            f"{base} fashionable pants trousers skirt",

        "shoes":
            f"{base} fashionable shoes",

        "bags":
            f"{base} fashionable handbag bag"
    }


# ============================================================
# RETRIEVE CATEGORY ITEMS
# ============================================================

def retrieve_category_items(
    query_embedding,
    category,
    faiss_index,
    image_ids,
    metadata_df,
    top_k=3
):

    query = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    search_k = 500

    similarities, positions = (
        faiss_index.search(
            query,
            search_k
        )
    )

    results = []

    for similarity, position in zip(
        similarities[0],
        positions[0]
    ):

        if position < 0:
            continue

        item_id = str(
            image_ids[position]
        )

        if item_id not in metadata_df.index:
            continue

        row = metadata_df.loc[
            item_id
        ]

        item_category = row.get(
            "semantic_category",
            ""
        )

        if item_category != category:
            continue

        results.append(
            {
                "item_id": item_id,

                "category": category,

                "similarity": float(
                    similarity
                ),

                "title": row.get(
                    "title",
                    ""
                ),

                "description": row.get(
                    "description",
                    ""
                )
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# DISPLAY ITEM
# ============================================================

def display_item(item):

    item_id = item["item_id"]

    image_path = os.path.join(
        IMAGE_DIR,
        f"{item_id}.jpg"
    )

    st.markdown(
        f"**{item['category'].title()}**"
    )

    if os.path.exists(
        image_path
    ):

        st.image(
            image_path,
            width=220
        )

    else:

        st.info(
            f"Image {item_id} is not downloaded locally."
        )

    title = item.get(
        "title",
        ""
    )

    description = item.get(
        "description",
        ""
    )

    if title:

        st.write(
            f"**{title}**"
        )

    if description:

        st.caption(
            description[:250]
        )

    st.caption(
        f"FashionCLIP similarity: "
        f"{item['similarity']:.4f}"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "👗 Multimodal AI Fashion Stylist"
)

st.write(
    "Describe the outfit you want and the AI "
    "will retrieve fashion items that match your request."
)

st.caption(
    f"Running on: {DEVICE.upper()}"
)


# ============================================================
# LOAD DATABASE
# ============================================================

with st.spinner(
    "Loading fashion database..."
):

    (
        embeddings,
        image_ids,
        metadata_df
    ) = load_data()

st.success(
    f"Fashion database loaded: "
    f"{len(image_ids):,} items"
)


# ============================================================
# BUILD FAISS
# ============================================================

with st.spinner(
    "Preparing fashion search engine..."
):

    faiss_index = build_faiss_index(
        embeddings
    )

st.success(
    "Fashion search engine ready."
)


# ============================================================
# LOAD FASHIONCLIP
# ============================================================

with st.spinner(
    "Loading FashionCLIP..."
):

    (
        processor,
        fashion_clip_model
    ) = load_fashion_clip()

st.success(
    "FashionCLIP loaded successfully."
)


# ============================================================
# USER INPUT
# ============================================================

user_request = st.text_area(
    "What are you looking for?",

    placeholder=(
        "Example: I want a classy but comfortable "
        "outfit for shopping. I prefer neutral "
        "and pastel colors and nothing too tight."
    ),

    height=120
)


# ============================================================
# BUTTON
# ============================================================

if st.button(
    "✨ Find My Outfit",
    type="primary"
):

    if not user_request.strip():

        st.warning(
            "Please describe the outfit you want."
        )

        st.stop()


    # ========================================================
    # LLM
    # ========================================================

    with st.spinner(
        "Understanding your preferences..."
    ):

        tokenizer, llm_model = (
            load_llm()
        )

        preferences = extract_preferences(
            user_request,
            tokenizer,
            llm_model
        )


    # ========================================================
    # PREFERENCES
    # ========================================================

    st.subheader(
        "🧠 Detected Preferences"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Styles:**",
            ", ".join(
                preferences.get(
                    "styles",
                    []
                )
            )
            or "None detected"
        )

        st.write(
            "**Colors:**",
            ", ".join(
                preferences.get(
                    "colors",
                    []
                )
            )
            or "None detected"
        )

    with col2:

        st.write(
            "**Occasion:**",
            preferences.get(
                "occasion",
                ""
            )
            or "None detected"
        )

        st.write(
            "**Clothing preferences:**",
            ", ".join(
                preferences.get(
                    "clothing_preferences",
                    []
                )
            )
            or "None detected"
        )


    # ========================================================
    # QUERIES
    # ========================================================

    queries = build_category_queries(
        preferences
    )


    # ========================================================
    # TEXT EMBEDDINGS
    # ========================================================

    with st.spinner(
        "Searching the fashion database..."
    ):

        query_categories = list(
            queries.keys()
        )

        query_texts = [
            queries[category]
            for category in query_categories
        ]

        query_embeddings = (
            get_text_embeddings(
                query_texts,
                processor,
                fashion_clip_model
            )
        )


    # ========================================================
    # RETRIEVE
    # ========================================================

    recommendations = {}

    for category, query_embedding in zip(
        query_categories,
        query_embeddings
    ):

        recommendations[category] = (
            retrieve_category_items(
                query_embedding,
                category,
                faiss_index,
                image_ids,
                metadata_df,
                top_k=3
            )
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    st.subheader(
        "👗 Recommended Items"
    )

    category_order = [
        "outerwear",
        "tops",
        "bottoms",
        "shoes",
        "bags"
    ]

    columns = st.columns(5)

    for column, category in zip(
        columns,
        category_order
    ):

        with column:

            st.markdown(
                f"### {category.title()}"
            )

            items = recommendations.get(
                category,
                []
            )

            if not items:

                st.warning(
                    "No items found."
                )

                continue

            for item in items:

                display_item(
                    item
                )

                st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FashionCLIP + FAISS + LLM preference extraction"
)