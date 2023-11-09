
This repo allows you to have your own choose-your-own-adventure stories using llama and stable-difufsion


usage: flaskApp.py [-h] [--model_id MODEL_ID] [--prompt_suffix PROMPT_SUFFIX]
                   [--image_sizes IMAGE_SIZES [IMAGE_SIZES ...]] [--llm_model LLM_MODEL]
                   [--failure_threshold FAILURE_THRESHOLD]

Generate a story

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   model id
  --prompt_suffix PROMPT_SUFFIX
                        prompt suffix
  --image_sizes IMAGE_SIZES [IMAGE_SIZES ...]
                        image sizes
  --llm_model LLM_MODEL
                        llm model
  --failure_threshold FAILURE_THRESHOLD
                        this is the number that must be rolled on a 20 sided die to succeed


IMPORTANT!  You will need to download a gguf lama model (for example from https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/)  and pass it to flaskApp.py

for example

python flaskApp.py --llm_model mistral-7b-openorca.Q5_K_M.gguf

