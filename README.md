# 🧠 ViT, CLIP & Stable Diffusion: Vision Transformer Applications

A comprehensive implementation of state-of-the-art Vision Transformer (ViT)-based models, OpenAI's CLIP, and Stable Diffusion for tasks like spoof detection, AI-powered image retrieval, style transfer, and product mockup generation.

---

## 🎯 Objective
Explore and evaluate modern generative and transformer-based models in:
1. **Spoof Detection**
2. **AI-Powered Visual Search with CLIP**
3. **Image Generation using Stable Diffusion**
4. **Mockup Generation for E-Commerce**

---

## 📂 Datasets
- **COCO** Validation Images + Annotations
- **CelebA-Spoof** (`nguyenkhoa/celeba-spoof-for-face-antispoofing-test` from HuggingFace)

---

## 🛡️ Part 1: Spoof Detection (25%)
Use ViT for binary classification of face images as *Real* or *Spoofed*.

### ✅ Tasks:
- **Dataset Preparation**: Load the CelebA Spoof dataset.
- **Model Training**: Fine-tune a ViT model for binary classification.
- **Evaluation**: Measure Accuracy, Precision, Recall, and F1-score.
- **Testing**:
  - Add your own **real** photo.
  - Add a **spoofed** version (e.g., image replay from a mobile screen).

---

## 🔎 Part 2: AI-Powered Visual Search with CLIP (35%)
Implement an image retrieval system using CLIP with ViT backbone.

### ✅ Tasks:
- Download and load:
  - COCO `val2017.zip`
  - `annotations_trainval2017.zip`
- Use `openai/clip-vit-base-patch32` model
- For a **text query**, retrieve the **top 5 most similar images**.
- Display each image with **similarity score** and **query text**.

📸 **Example Output:**
```
Query: "a dog running on the beach"
Image 1: Score 0.92
Image 2: Score 0.89
...
```

---

## 🎨 Part 3: Stable Diffusion - Style Transfer (20%)
Use pre-trained Stable Diffusion to generate diverse styles.

### ✅ Tasks:
- Load a pre-trained Stable Diffusion model
- Create a Python script to:
  - Take an input image
  - Generate new images based on prompts
- Tune Parameters:
  - `strength`, `guidance_scale`, `num_inference_steps`
  - Produce **3 variations** using one prompt, documenting the differences

### 🎯 Prompt Engineering:
Generate images for these prompts:
- "watercolor painting"
- "a pixel art"
- "in the style of Salvador Dalí"
- "in the style of Van Gogh"
- "low-poly fantasy style"

Compare how each style is interpreted.

---

## 🛍️ Part 4: Product Mockup Generation (20%)
Generate mockups for e-commerce using Stable Diffusion.

### ✅ Tasks:

#### 1. Basic Mockup:
- Input: Plain white mug (or other product base)
- Prompt: "vintage floral pattern"
- Output: Realistic product mockup

#### 2. Style Customization:
- Prompt: "minimalist geometric design"
- Tune `strength` and `guidance_scale`
- Generate **3 variations** to show differences in style intensity

#### 3. Style Diversity:
Generate product visuals for these styles:
- "Floral Art"
- "Graffiti"
- "Kawaii style"
- "Cyberpunk neon style"
- "Cartoon comic art"

---

## 📦 Installation
```bash
# Clone repository
https://github.com/yourusername/vit-clip-stable-diffusion-cv.git
cd vit-clip-stable-diffusion-cv

# Install dependencies
pip install -r requirements.txt
```

### 📋 Prerequisites
Ensure you have the following:
- Python 3.8+
- CUDA-enabled GPU
- `transformers`, `diffusers`, `torch`, `datasets`, `scikit-learn`, `matplotlib`

---

## 💡 Usage Examples
```bash
# Spoof Detection
python vit_spoof_detection.py --dataset celebA

# CLIP Retrieval
python clip_search.py --query "a cat sitting on a sofa"

# Style Transfer
python sd_stylize.py --input input.jpg --prompt "a watercolor painting"

# Product Mockup
python mockup_generator.py --base mug.jpg --prompt "floral art"
```

---

## 🚀 Deployment
This project can be containerized using Docker or deployed via a Flask-based REST API.

---

## 🛠️ Technologies Used
- [ViT](https://arxiv.org/abs/2010.11929)
- [CLIP](https://openai.com/research/clip)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [PyTorch](https://pytorch.org/)
- [COCO Dataset](https://cocodataset.org/)

---

## 🤝 Contributing
1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📚 Documentation
- [CLIP on HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32)
- [Stable Diffusion Docs](https://huggingface.co/docs/diffusers)
- [ViT Paper](https://arxiv.org/abs/2010.11929)

---

## 🙌 Acknowledgments
- [OpenAI](https://openai.com/)
- [Hugging Face](https://huggingface.co/)
- [CompVis](https://github.com/CompVis)

---

## 📜 License
[MIT License](https://choosealicense.com/licenses/mit/)

