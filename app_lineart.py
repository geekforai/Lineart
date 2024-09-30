from flask import Flask, request, jsonify
from PIL import Image
import io
from settings import (
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_NUM_IMAGES,
    MAX_IMAGE_RESOLUTION,
    MAX_NUM_IMAGES,
    MAX_SEED,
)
from utils import randomize_seed_fn
from model import Model

app = Flask(__name__)
model = Model(task_name="lineart")

@app.route('/generate', methods=['POST'])
def generate_image():
    # Retrieve the image and prompt from the request
    data = request.json
    prompt = data.get("prompt")
    image_data = data.get("image")
    
    # Convert the image from base64 to PIL Image
    if image_data:
        image = Image.open(io.BytesIO(image_data))
    
    # Prepare other parameters (you can add more as needed)
    additional_prompt = data.get("additional_prompt", "best quality, extremely detailed")
    negative_prompt = data.get("negative_prompt", "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
    num_samples = data.get("num_samples", DEFAULT_NUM_IMAGES)
    image_resolution = data.get("image_resolution", DEFAULT_IMAGE_RESOLUTION)
    num_steps = data.get("num_steps", 20)
    guidance_scale = data.get("guidance_scale", 9.0)
    seed = data.get("seed", 0)

    # Process the image with the model
    result = model.process_lineart(
        image=image,
        prompt=prompt,
        additional_prompt=additional_prompt,
        negative_prompt=negative_prompt,
        num_samples=num_samples,
        image_resolution=image_resolution,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed
    )

    # Convert the result to a suitable format (e.g., base64 or file path)
    # This example assumes result is already in a suitable format
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
