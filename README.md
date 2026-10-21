# 👗 Fashion AI Stylist — AI-Powered Fashion Recommendation System

An AI-powered fashion recommendation system designed to help users discover clothing items and create outfit ideas based on visual similarity and fashion preferences.

The project explores how **multimodal AI, image embeddings, and vector similarity search** can be combined to build an interactive fashion discovery experience.

## ✨ Project Overview

Finding clothing items that complement a particular style can be challenging when browsing large fashion catalogs. This project applies modern AI techniques to fashion recommendation, enabling users to explore visually similar items through an interactive application.

The system uses pretrained vision-language models to represent fashion images as numerical embeddings and uses similarity search to retrieve relevant recommendations.

## 🚀 Key Features

- **AI-Powered Fashion Retrieval:** Find visually similar fashion items using image embeddings.
- **Visual Similarity Search:** Retrieve relevant items using vector similarity search.
- **Multimodal AI:** Explore pretrained vision-language representations for fashion images.
- **Interactive Web Application:** A Streamlit interface for interacting with the recommendation system.
- **Scalable Retrieval Pipeline:** FAISS-based indexing and similarity search for efficient embedding retrieval.

*Features listed above should be adjusted to match the final functionality implemented in the application.*

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| PyTorch | Deep learning and model inference |
| FashionCLIP | Fashion-oriented image and text representations |
| FAISS | Vector indexing and similarity search |
| Streamlit | Interactive web application |
| NumPy | Numerical operations and embedding processing |
| Polyvore Dataset | Fashion images and associated metadata |

## ⚙️ How It Works

The recommendation pipeline follows these general steps:

1. **Fashion Data:** Load fashion images and the associated dataset metadata.
2. **Feature Extraction:** Use a pretrained vision-language model to generate image embeddings.
3. **Vector Indexing:** Store or index the embeddings for similarity search.
4. **Recommendation Retrieval:** Compare a query representation with indexed fashion items and retrieve the closest matches.
5. **User Interaction:** Present recommendations through the Streamlit application.

## 📂 Dataset

This project uses the **Polyvore fashion dataset**, which contains fashion-related images and metadata.

The dataset supports experimentation with visual fashion retrieval and recommendation.

Due to its size, the complete dataset is not included directly in this repository. Users should obtain the data from an authorized source and configure the appropriate local paths before running the project.

Please respect the original dataset's licensing and usage restrictions.

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fashion-ai-stylist
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by the project's environment configuration.

### 4. Configure the dataset

Download the Polyvore dataset from an authorized source and update the dataset paths in the project configuration or application code.

Ensure the required image files, metadata, and any precomputed embeddings or indexes are available.

### 5. Launch the application

```bash
streamlit run app.py
```

The exact setup may vary depending on the available model files, dataset configuration, and dependency versions.

## 💻 Example Use Case

A user wants to explore fashion items similar to a particular clothing piece.

The system represents the query image in an embedding space, searches for visually similar items, and retrieves relevant results from the fashion catalog.

This demonstrates how deep learning and vector search can support more intuitive fashion discovery.

## 🎯 Project Goals

- Apply pretrained deep learning models to a real-world recommendation problem.
- Explore image embeddings and multimodal representation learning.
- Implement vector similarity search for fashion retrieval.
- Integrate an AI pipeline into an interactive web application.
- Develop practical experience with the deployment and integration of AI components.

## 🔮 Future Improvements

- Incorporate text-based fashion queries.
- Recommend complete outfits rather than individual similar items.
- Add user preferences and personalized recommendations.
- Evaluate recommendation quality using appropriate retrieval metrics.
- Optimize model inference and index construction for larger catalogs.

## 👩‍💻 Author

**Ibtissame Jabir**

Artificial Intelligence Engineering Graduate

Interested in Machine Learning, Deep Learning, Computer Vision, and Multimodal AI.

- GitHub: [ibtissamejabir](https://github.com/ibtissamejabir)
- LinkedIn: [ibtissame-jabir](https://www.linkedin.com/in/ibtissame-jabir/)

---
