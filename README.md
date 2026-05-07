# ComfyUI ToriiGate

ComfyUI custom nodes for [Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5) — an image captioning model specialized for digital art and anime-style images.

- 🤗 Model: [Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5)
- 🖥️ Reference Space: [Minthy/ToriiGate-0.5-Gradio](https://huggingface.co/spaces/Minthy/ToriiGate-0.5-Gradio)

---

## Nodes

- **ToriiGate Model Loader** — loads the model and processor. Hover over each option in ComfyUI for a description.
- **ToriiGate Grounding Builder** — builds optional character/tag grounding metadata to guide the caption.
- **ToriiGate Captioner** — generates a caption from a ComfyUI image using the loaded model.

---

## Installation

Clone into your `ComfyUI/custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/comfyui_toriigate.git
```

Install requirements into the same Python environment that runs ComfyUI:

```bash
# Standard Python / venv
pip install -r ComfyUI/custom_nodes/comfyui_toriigate/requirements.txt

# ComfyUI Portable (Windows) — run from the ComfyUI portable root
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui_toriigate\requirements.txt
```

Restart ComfyUI after installation.

---

## Credits

Model, training, and prompt formats by the original ToriiGate authors. This repository is a ComfyUI integration layer only.
