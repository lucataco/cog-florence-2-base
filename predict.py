# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, BaseModel, Input, Path
from PIL import Image
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForCausalLM 

MODEL_NAME = "microsoft/Florence-2-base"
MODEL_CACHE = "checkpoints"

TASKS = {
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
    "Caption to Phrase Grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "Object Detection": "<OD>",
    "Dense Region Caption": "<DENSE_REGION_CAPTION>",
    "Region Proposal": "<REGION_PROPOSAL>",
    "OCR": "<OCR>",
    "OCR with Region": "<OCR_WITH_REGION>"
}

class Output(BaseModel):
    text: str
    img: Optional[Path]

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
    return fig

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        task_input: str = Input(
            description="Input task",
            choices=TASKS.keys(),
            default="Caption"
        ),
        text_input: str = Input(description="Text Input(Optional)", default=None),
    ) -> Output:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")

        task = TASKS[task_input]
        if text_input is None:
            prompt = task
        else:
            prompt = task + text_input
    
        inputs = self.processor(prompt, img, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(img.width, img.height)
        )
        if task in ("<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>"):
            bbox_img = plot_bbox(img, parsed_answer[task])
            bbox_img.savefig("/tmp/output.png")
            return Output(text=str(parsed_answer), img=Path("/tmp/output.png"))
        return Output(text=str(parsed_answer))
