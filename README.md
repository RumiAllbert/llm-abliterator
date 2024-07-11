# 🚀 Prompt Induced Abliteration / Induction

Most of the code and implementation for this project comes from [FailSpy's Abliterator](https://github.com/FailSpy/abliterator). However, I have modified the code and setup to better fit my use case.

Original Paper: [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## 📋 Setup

### 1. Check PyTorch Version and GPU Availability

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### 2. Install Dependencies

Install the Hugging Face Transformers library and other dependencies.

*NOTE*: If using GPU (which you should) you migtht need to check which [torch version](https://pytorch.org/get-started/locally/) you need.
```bash
!pip install -q transformers einops transformer_lens scikit-learn torch
```
Can also pip install from a requirements file:
```bash
!pip install -r requirements.txt
```



### 3. Login to Hugging Face (if necessary)

Depending on the model you plan to use, you may need to log in to Hugging Face to download the model.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 🧠 Model Initialization

### 4. Load the Model and Tokenizer

Load the model and configure it. Note that the model has to be supported by transformers lens.

```python
from abliterator import *

# You can check abliterator.py for more information about the prompt data. These are just basic baseline instructions.
model = ModelAbliterator(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    [
        get_baseline_instructions(),
        get_baseline_instructions(),
    ],
    activation_layers=["resid_pre"],
)

# Blacklist the first and last layers. This is optional. You can blacklist any layers you want.
model.blacklist_layer([0, 1, 2, 3, 29, 30, 31])
```

### 5. Configure Prompt

Create a ChatTemplate. Modify this as you wish. This is just an example.
**Note:** The template will depend on which model you are using. This is for Llama3 architecture. For something like Phi3 it will look like this:

```python
phi3_template="""<|user|>\n{instruction}<|end|>\n<|assistant|>"""
```

```python
system_prompt = """You are highly optimistic. Your responses should reflect a positive and hopeful outlook on life. Emphasize the bright side of any situation, and express strong confidence that things will turn out well. Encourage others with uplifting and encouraging language."""
chat_template = ChatTemplate(
    model,
    "<|start_header_id|>system<|end_header_id|>\n"
    + system_prompt
    + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n{instruction}<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

### 6. Check Baseline Behavior

Let's see how the model responds as a baseline.

```python
model.test(
    N=32,
    test_set=model.baseline_inst_test[15:16],
    max_tokens_generated=100,
    drop_refusals=False,
)
```

Measure the effectiveness of our prompt.

```python
with chat_template:
    model.test(N=4, test_set=model.baseline_inst_test[30:33], drop_refusals=False)
```

## 🔍 Get the Activation Direction

### 7. Define File Paths

Set up paths for saving baseline and altered caches.

```python
import os
from tqdm.notebook import tqdm

MODEL = "llama3"
# MODEL = "phi3"

baseline_cache_path = f"/baseline_cache_{MODEL}_compressed.pkl.gz"
```

### 8. Calculate Baseline Cache

Calculate the baseline cache if it doesn't exist.

```python
if not os.path.exists(baseline_cache_path):
    print("Calculating baseline cache...")

    # Define prompt count
    prompt_count = 1500  # using more samples can better target the direction

    # Tokenize instructions for baseline
    baseline = model.tokenize_instructions_fn(
        model.baseline_inst_train[:prompt_count]
    )  # Use base system prompt

    # Get baseline cache
    baseline_cache = model.create_activation_cache(baseline, N=len(baseline))
    base_cache, _ = baseline_cache

    # Save baseline cache
    save_compressed_cache(base_cache, baseline_cache_path)

else:
    print("Baseline cache already exists.")

# Load baseline cache
baseline_cache = load_compressed_cache(baseline_cache_path, model)
```

### 9. Create Altered (trait) Cache

Create an altered cache using the ChatTemplate.

```python
with chat_template:
    # Tokenize instructions for altered tokens
    altered_toks = model.tokenize_instructions_fn(
        model.baseline_inst_train[:prompt_count]
    )

altered_cache = model.create_activation_cache(altered_toks, N=len(altered_toks))
```

Set trait and baseline caches.

```python
# Set trait and baseline caches
model.trait, _ = altered_cache
model.baseline = baseline_cache

# Get feature directions
feature_directions = model.refusal_dirs(
    invert=True
)  # inverted because we're attempting to induce the feature, otherwise it would be a refusal direction
```

### 10. Find the Best Direction

Find the direction that best expresses the desired behavior. Adjust the modifier value if the model is not behaving as expected.

```python
modifier = 1.3  # Lower is more stable. I've found 1.3 to 1.5 is good.

for block in feature_directions:
    with model:  # This line makes it so any changes we apply to the model's weights will be reverted on each loop
        model.apply_refusal_dirs([feature_directions[block] * modifier])
        print(block)

        model.test(
            N=32,
            test_set=model.baseline_inst_test[15:25],
            max_tokens_generated=64,
            drop_refusals=False,
        )
        print("=" * 100)
```

### 11. Clear Memory

Clear memory before proceeding if necessary.

```python
clear_mem()
```

## 🔧 Apply the Direction

### 12. Apply Refusal Directions

Apply the identified direction to the model. I have found that block (layer) 17 and 18 tend to give the best desired behavior.

```python
model.apply_refusal_dirs([feature_directions["blocks.18.hook_resid_pre"] * modifier])
```

### 13. Test Modified Model

Test the modified model to ensure it behaves as expected.

```python
model.test(
    N=32,
    test_set=model.baseline_inst_test[15:25],
    max_tokens_generated=64,
    drop_refusals=False,
)
```

## 💾 Save the Model

### 14. Save Model State

Save the model state for future use. This is kind of a wonky approach, but it works.

```python
cfg = model.model.cfg
state_dict = model.model.state_dict()

hf_model = AutoModelForCausalLM.from_pretrained(
    model.MODEL_PATH, torch_dtype=torch.bfloat16
)
lm_model = hf_model.model  # get the language model component

for l in range(cfg.n_layers):
    lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(
        einops.rearrange(
            state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=cfg.n_heads
        ).contiguous()
    )
    lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(
        torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
    )
```

### 15. Push to Hugging Face Hub

Push the model to the Hugging Face Hub.

```python
hf_model.push_to_hub("your-model-name")
```

### 16. Save Locally

Alternatively, save the model locally.

```python
hf_model.save_pretrained("your model name")
```

# ----------------------------

# FastAPI Model Generation Service

There is also a FastAPI service to generate text based on a given prompt and feature directions. It also provides a health check endpoint. This is useful if you have saved your feature directions and want to generate text on the fly.

## Endpoints

### Health Check

**URL**: `/health`

**Method**: GET

**Description**: Check the health of the service.

**Response**:
```json
{
  "status": "Service is up and running",
  "pytorch_version": "<PyTorch version>",
  "cuda_available": "<True/False>",
  "gpu_name": "<GPU Name if available>"
}
```

### Generate Text

**URL**: `/generate`

**Method**: POST

**Description**: Generate text based on the provided prompt and feature directions.

**Request Body**:
```json
{
  "prompt": "Your prompt text",
  "feature_directions": [0.1, 0.2, 0.3, ...],  // List of floats
  "modifier": 1.3,  // Optional, default is 1.3
  "max_tokens": 100  // Optional, default is 100
}
```

**Response**:
```json
{
  "response": "Generated text"
}
```

## Running the Service

To run the FastAPI service, execute the following command:

```bash
uvicorn <your_script_name>:app --host 0.0.0.0 --port 8888
```

Replace `<your_script_name>` with the name of your Python script containing the FastAPI app. I use `api.py` in this case.

## Example Usage

### Health Check

To check the health of the service, make a GET request to the `/health` endpoint:

```bash
curl -X GET "http://0.0.0.0:8888/health"
```

### Generate Text

To generate text, make a POST request to the `/generate` endpoint with the appropriate JSON payload:

```bash
curl -X POST "http://0.0.0.0:8888/generate" -H "Content-Type: application/json" -d '{
  "prompt": "Tell me a story about AI",
  "feature_directions": [0.5, -0.2, 0.1],
  "modifier": 1.3,
  "max_tokens": 150
}'
```

This will return a JSON response with the generated text.
