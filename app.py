import gradio as gr
import torch

from inference import TTA, load_model

device = torch.device("cpu")
model = load_model("EffNet/results/Modified_EffNet_f1_saved_model.pth", device)
tta = TTA(model, device, n_aug=5)

def classify(image):
    result = tta.predict(image)
    return result["all_probs"]


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Cassava Leaf Disease Classifier",
    description="Upload a cassava leaf image to identify the disease"
)

demo.launch()