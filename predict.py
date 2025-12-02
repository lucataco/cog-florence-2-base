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
    "OCR with Region": "<OCR_WITH_REGION>",
    "Referring Expression Segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "Region to Segmentation": "<REGION_TO_SEGMENTATION>",
    "Open Vocabulary Detection": "<OPEN_VOCABULARY_DETECTION>",
    "Region to Category": "<REGION_TO_CATEGORY>",
    "Region to Description": "<REGION_TO_DESCRIPTION>",
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

def plot_polygons(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for polygons, label in zip(data['polygons'], data['labels']):
        for polygon in polygons:
            # Polygon is a flat list of x,y coordinates, reshape to pairs
            points = list(zip(polygon[::2], polygon[1::2]))
            if len(points) >= 3:
                poly_patch = patches.Polygon(points, linewidth=2, edgecolor='r', facecolor='red', alpha=0.3)
                ax.add_patch(poly_patch)
                # Add label at first point
                plt.text(points[0][0], points[0][1], label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
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
        text_input: str = Input(description="Text input. Required for: Caption to Phrase Grounding (caption text), Referring Expression Segmentation (expression to segment), Region to Segmentation/Category/Description (region as <loc_x1><loc_y1><loc_x2><loc_y2>), Open Vocabulary Detection (object to detect)", default=None),
    ) -> Output:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")

        task = TASKS[task_input]

        # Tasks that require text_input
        tasks_requiring_text = (
            "<CAPTION_TO_PHRASE_GROUNDING>",
            "<REFERRING_EXPRESSION_SEGMENTATION>",
            "<REGION_TO_SEGMENTATION>",
            "<REGION_TO_CATEGORY>",
            "<REGION_TO_DESCRIPTION>",
            "<OPEN_VOCABULARY_DETECTION>",
        )
        if task in tasks_requiring_text and not text_input:
            raise ValueError(f"text_input is required for {task_input}")

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
        # Tasks that return bounding boxes
        if task in ("<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>"):
            bbox_img = plot_bbox(img, parsed_answer[task])
            bbox_img.savefig("/tmp/output.png")
            return Output(text=str(parsed_answer), img=Path("/tmp/output.png"))
        # Tasks that return polygons (segmentation)
        if task in ("<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"):
            poly_img = plot_polygons(img, parsed_answer[task])
            poly_img.savefig("/tmp/output.png")
            return Output(text=str(parsed_answer), img=Path("/tmp/output.png"))
        return Output(text=str(parsed_answer))
