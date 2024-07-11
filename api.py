import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
from abliterator import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI()


# Define the request body
class GenerateRequest(BaseModel):
    prompt: str
    feature_directions: List[float]
    modifier: float = 1.3
    max_tokens: int = 100


# Load the model and tokenizer
logger.info("Loading the model and tokenizer")
LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

model = ModelAbliterator(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    [
        get_baseline_instructions(),
        get_baseline_instructions(),
    ],
    local_files_only=True,
    activation_layers=["resid_pre"],
    chat_template=LLAMA3_CHAT_TEMPLATE,
)
model.blacklist_layer([0, 1, 2, 3, 29, 30, 31])
logger.info("Model and tokenizer loaded successfully")


def convert_to_bfloat16_tensor(feature_directions: List[float]) -> torch.Tensor:
    try:
        feature_directions_np = np.array(feature_directions, dtype=np.float32)
        feature_directions_tensor = torch.tensor(
            feature_directions_np, dtype=torch.bfloat16
        )
        return feature_directions_tensor
    except Exception as e:
        logger.error(
            "Error converting feature directions to bfloat16 tensor: %s", str(e)
        )
        raise


@app.post("/generate")
async def get_generation(request: GenerateRequest):
    logger.info("Received request with prompt: %s", request.prompt)
    logger.info("Feature vector size: %d", len(request.feature_directions))
    logger.info("Modifier: %f", request.modifier)

    prompt = request.prompt
    feature_directions = convert_to_bfloat16_tensor(request.feature_directions)

    try:
        logger.info("Clearing memory")
        clear_mem()

        with model:
            logger.info(
                "Applying refusal directions with modifier: %f", request.modifier
            )
            model.apply_refusal_dirs([feature_directions * request.modifier])

            # Generate a response using the modified model
            logger.info("Generating response")

            response = model.generate(
                prompt,
                max_tokens_generated=request.max_tokens,
                stop_at_eos=True,
                top_p=0.95,
                temperature=0.9,
            )

            response = response.strip()
            logger.info("Response generated and cleaned successfully")

            return response
    except Exception as e:
        logger.error("Error during generation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint was called")

    pytorch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "status": "Service is up and running",
        "pytorch_version": pytorch_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
