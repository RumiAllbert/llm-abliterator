from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from abliterator import ModelAbliterator, get_baseline_instructions, ChatTemplate
from transformers import AutoModelForCausalLM

# Initialize the FastAPI app
app = FastAPI()


# Define the request body
class GenerateRequest(BaseModel):
    prompt: str
    feature_directions: List[float]
    modifier: float = 1.3


# Load the model and tokenizer
model = ModelAbliterator(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    [
        get_baseline_instructions(),
        get_baseline_instructions(),
    ],
    activation_layers=["resid_pre"],
)
model.blacklist_layer([0, 1, 2, 3, 29, 30, 31])


@app.post("/generate")
async def get_generation(request: GenerateRequest):
    prompt = request.prompt
    feature_directions = torch.tensor(request.feature_directions)

    try:
        # Use this to generate a response without modifying the model
        with model:
            model.apply_refusal_dirs([feature_directions * request.modifier])

            # Generate a response using the modified model
            response = model.generate(prompt, max_tokens_generated=300)

            return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
