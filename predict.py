# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

MODEL_NAME = "microsoft/Florence-2-base"
MODEL_CACHE = "checkpoints"
# MODEL_CACHE_URL = "https://weights.replicate.delivery/default/microsoft/Florence-2-base/model.tar"

# def download_weights(url, dest):
#     start = time.time()
#     print("downloading url: ", url)
#     print("downloading to: ", dest)
#     subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
#     print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # if not os.path.exists(MODEL_CACHE):
        #     download_weights(MODEL_CACHE_URL, MODEL_CACHE)
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
        prompt: str = Input(description="Input prompt", default="<OD>"),
    ) -> str:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")
        inputs = self.processor(prompt, img, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(img.width, img.height))
        return str(parsed_answer)
