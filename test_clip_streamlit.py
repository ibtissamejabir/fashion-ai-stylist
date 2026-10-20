import streamlit as st

from transformers.models.clip.processing_clip import CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPModel

st.write("1. Imports OK")

st.write("2. Loading FashionCLIP...")

processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")

st.write("3. FashionCLIP OK")
