# @title ## Upload model to Hugging Face { display-mode: "form" }
# @markdown Enter your HF username and the name of Colab secret that stores your [Hugging Face access token](https://huggingface.co/settings/tokens).
username = 'theblackflag' # @param {type:"string"}
token = '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' # @param {type:"string"}
license = "" # @param ["apache-2.0", "cc-by-nc-4.0", "mit", "openrail"] {allow-input: true}

!pip install -qU huggingface_hub

import yaml

from huggingface_hub import ModelCard, ModelCardData, HfApi
from google.colab import userdata
from jinja2 import Template

if branch == "main":
    template_text = """
---
license: {{ license }}
base_model:
{%- for model in models %}
  - {{ model }}
{%- endfor %}
tags:
- merge
- mergekit
- lazymergekit
{%- for model in models %}
- {{ model }}
{%- endfor %}
---

# {{ model_name }}

{{ model_name }} is a merge of the following models using [LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing):

{%- for model in models %}
* [{{ model }}](https://huggingface.co/{{ model }})
{%- endfor %}

## 🧩 Configuration

```yaml
{{- yaml_config -}}
```

## 💻 Usage

```python
!pip install -qU transformers accelerate

from transformers import AutoTokenizer
import transformers
import torch

model = "{{ username }}/{{ model_name }}"
messages = [{"role": "user", "content": "What is a large language model?"}]

tokenizer = AutoTokenizer.from_pretrained(model)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```
"""

    # Create a Jinja template object
    jinja_template = Template(template_text.strip())

    # Get list of models from config
    data = yaml.safe_load(yaml_config)
    if "models" in data:
        models = [data["models"][i]["model"] for i in range(len(data["models"])) if "parameters" in data["models"][i]]
    elif "parameters" in data:
        models = [data["slices"][0]["sources"][i]["model"] for i in range(len(data["slices"][0]["sources"]))]
    elif "slices" in data:
        models = [data["slices"][i]["sources"][0]["model"] for i in range(len(data["slices"]))]
    else:
        raise Exception("No models or slices found in yaml config")

    # Fill the template
    content = jinja_template.render(
        model_name=MODEL_NAME,
        models=models,
        yaml_config=yaml_config,
        username=username,
    )

elif branch == "mixtral":
    template_text = """
---
license: {{ license }}
base_model:
{%- for model in models %}
  - {{ model }}
{%- endfor %}
tags:
- moe
- frankenmoe
- merge
- mergekit
- lazymergekit
{%- for model in models %}
- {{ model }}
{%- endfor %}
---

# {{ model_name }}

{{ model_name }} is a Mixture of Experts (MoE) made with the following models using [LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing):

{%- for model in models %}
* [{{ model }}](https://huggingface.co/{{ model }})
{%- endfor %}

## 🧩 Configuration

```yaml
{{- yaml_config -}}
```

## 💻 Usage

```python
!pip install -qU transformers bitsandbytes accelerate

from transformers import AutoTokenizer
import transformers
import torch

model = "{{ username }}/{{ model_name }}"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)

messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```
"""

    # Create a Jinja template object
    jinja_template = Template(template_text.strip())

    # Fill the template
    data = yaml.safe_load(yaml_config)
    models = [model['source_model'] for model in data['experts']]

    content = jinja_template.render(
        model_name=MODEL_NAME,
        models=models,
        yaml_config=yaml_config,
        username=username,
        license=license
    )

# Save the model card
card = ModelCard(content)
card.save('merge/README.md')

# Defined in the secrets tab in Google Colab
api = HfApi(token=userdata.get(token))

# Upload merge folder
api.create_repo(
    repo_id=f"{username}/{MODEL_NAME}",
    repo_type="model",
    exist_ok=True,
)
api.upload_folder(
    repo_id=f"{username}/{MODEL_NAME}",
    folder_path="merge",
)