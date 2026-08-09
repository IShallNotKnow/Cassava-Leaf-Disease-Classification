import gradio as gr
import torch
from inference import TTA, load_model
from huggingface_hub import hf_hub_download

student_weights = hf_hub_download(
    repo_id="IShallNotKnow/cassava-models",
    filename="Modified_EffNet_f1_saved_model.pth"
)

device = torch.device("cpu")
model = load_model(student_weights, device)
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