---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - enterprise
  - systems-engineering
  - cybersecurity
  - infrastructure
  - code-generation
  - peft
  - lora
  - qlora
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# The Substitution v1.0.0

**Enterprise Systems Engineering LLM**

Created by **Brandon Michael Jeanpierre Corporation d/b/a The Black Flag**
Organization: [theblackflagbmjc](https://huggingface.co/theblackflagbmjc)

## Model Description

The Substitution is a fine-tuned language model specialized for enterprise systems engineering tasks. Built on Qwen/Qwen2.5-7B-Instruct using QLoRA (4-bit quantization with Low-Rank Adaptation), it serves as a technical advisor for infrastructure architecture, systems administration, defensive cybersecurity, and enterprise operations.

### Capabilities

- Infrastructure architecture design and review
- Systems administration across macOS, Linux, and Windows
- Python-primary software engineering with multi-language support
- Defensive cybersecurity analysis and security governance (NIST CSF, SP 800-53, SP 800-171)
- Network engineering: subnetting, routing, topology design
- Database operations: MySQL, MSSQL, GraphQL
- Enterprise documentation and technical writing
- Cryptographic architecture and key management

### Intended Use

This model is intended for use within The Black Flag organizational infrastructure as a technical reasoning assistant. It operates under the governance framework of BMJC and the Doctrine of Inherent Entanglement.

## Training Details

### Base Model
- **Architecture:** Qwen/Qwen2.5-7B-Instruct
- **Parameters:** 7.6B

### Fine-tuning Method
- **Method:** QLoRA (4-bit NF4 quantization + LoRA)
- **LoRA Rank (r):** 64
- **LoRA Alpha:** 128
- **Target Modules:** all-linear
- **Trainable Parameters:** ~160M (~2.1% of total)

### Training Configuration
- **Epochs:** 3
- **Learning Rate:** 2e-4
- **Scheduler:** cosine
- **Effective Batch Size:** 32
- **Max Sequence Length:** 4096
- **Precision:** bfloat16

### Training Data
- OpenHermes-2.5 (instruction following)
- UltraChat (multi-turn dialogue)
- OpenOrca (reasoning)
- CodeSearchNet (code generation)
- GSM8K (mathematical reasoning)

## Evaluation Results

*Run `python evaluation/run_benchmarks.py` after training to populate.*

| Benchmark | Metric | Baseline (Qwen2.5-7B) |
|-----------|--------|----------------------:|
| HumanEval | pass@1 | 79.9% |
| GSM8K | accuracy | 82.6% |
| MMLU | accuracy | 74.2% |

## Usage

### Local Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "theblackflagbmjc/the-substitution",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "theblackflagbmjc/the-substitution",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are The Substitution, an enterprise systems engineering assistant."},
    {"role": "user", "content": "Design a VLAN segmentation strategy for a small office network with 50 users."},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

### API Server

```bash
python inference/api_server.py --host 0.0.0.0 --port 8000
```

### Interactive CLI

```bash
python inference/local_inference.py
```

## Hardware Requirements

| Environment | Minimum | Recommended |
|-------------|---------|-------------|
| Apple Silicon | M1 Pro 16GB | M1 Pro 32GB+ |
| NVIDIA GPU | 8GB VRAM (4-bit) | 16GB+ VRAM |
| CPU | 16GB RAM | 32GB+ RAM |

## Limitations

- Specialized for enterprise systems engineering; general conversational ability may be reduced compared to the base model
- Code generation is Python-primary; other languages are supported but with less depth
- Security guidance is defensive and governance-focused
- Knowledge is bounded by training data cutoff and base model capabilities

## License

Apache 2.0 (inherited from base model)

---

*Brandon Michael Jeanpierre Corporation d/b/a The Black Flag*
*Delaware Entity #7336243 | IRS 501(c)(3) EIN: 92-2858861*
*8 The Green, Ste A, Dover, DE 19901*
