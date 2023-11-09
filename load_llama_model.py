from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from ctransformers import AutoModelForCausalLM


def chatML(messages,llm,max_new_tokens=100):
    prompt=""
    for message in messages:
        prompt+="""<|im_start|>{role}
{content}<|im_end|>
""".format(role=message['role'],content=message['content'])
        

    prompt+="<|im_start|>assistant\n"
        
    result=llm(prompt, max_new_tokens=max_new_tokens)

    result=result[0]['generated_text']

    print("about to die",result)

    #truncate result at first <|im_end|> if it exists
    result=result[:result.find("<|im_end|>")].strip()

    #parse back into messages and return
    newMessage={"role":"assistant","content":result}
    
    return messages+[newMessage]
        
    
class Chatbot:
    def __init__(self,llm,systemPrompt):
        self.llm=llm
        self.systemPrompt=systemPrompt
        self.messages=[{"role":"system","content":systemPrompt}]

    def chat(self,text):
        message={"role":"user","content":text}
        self.messages.append(message)
        self.messages=chatML(self.messages,self.llm)
        return self.messages[-1]['content']



def getllama(modelName='nous'):
    
    #load llama model

    if modelName=="mistral-dolphin":
        model_name_or_path = "D:/lmstudio/TheBloke/dolphin-2.1-mistral-7B-GGUF"
        model_file="dolphin-2.1-mistral-7b.Q5_K_M.gguf"
        llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, model_file=model_file, model_type="mistral", gpu_layers=50)
        #tokenizer = AutoTokenizer.from_pretrained("TheBloke/dolphin-2.0-mistral-7B-GGUF", use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        return tokenizer,llm



    if modelName == "nous-llama2-7b":
        model_name_or_path = "D:/img/llama/Nous-Hermes-Llama-2-7B-GPTQ"
        model_basename = "model"

        use_triton = False

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=False,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None)
        

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

        return tokenizer,pipe


    if modelName=='nous':
        quantized_model_dir = "D:\\img\\llama\\Nous-Hermes-13B-GPTQ"
    #quantized_model_dir = "D:\\img\\llama\\Nous-Hermes-Llama2-13b-GPTQ"
    elif modelName=='wizard':
        quantized_model_dir = "D:\\img\\llama\\WizardLM-7B-uncensored-GPTQ"
    #quantized_model_dir = "D:\\img\\llama\\\mpt-7b-storywriter-4bit-128g"
    #quantized_model_dir = "D:\\img\\llama\\vicuna-7B-1.1-GPTQ-4bit-128g"



    llama_tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False)

    def get_config(has_desc_act):
        return BaseQuantizeConfig(
            bits=4,  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            #group_size=64,#falcon
            desc_act=has_desc_act
        )

    def get_model(model_base, triton, model_has_desc_act):
        if model_has_desc_act:
            model_suffix="latest.act-order"
        else:
            model_suffix="compat.no-act-order"
        
        if modelName=='nous':
            model_suffix="no-act.order"#nous
        elif modelName=='wizard':
            model_suffix="compat.no-act-order"#wizard
        #model_suffix="no-act-order"#vicuna?
        model_basename=f"{model_base}.{model_suffix}"
        #model_basename=f"{model_base}"#nous llama2
        return AutoGPTQForCausalLM.from_quantized(quantized_model_dir, 
                                                  use_safetensors=True, #wizard
                                                  #use_safetensors=False, #vicuna
                                                  model_basename=model_basename, 
                                                  device="cuda:0", 
                                                  use_triton=triton, 
                                                  quantize_config=get_config(model_has_desc_act),
                                                  #trust_remote_code=True,#falcon/mpt
                                                  )
    
    if modelName=='nous':
        llama_model = get_model("nous-hermes-13b-GPTQ-4bit-128g", triton=False, model_has_desc_act=False)#nous
    #llama_model = get_model("gptq_model-4bit-128g", triton=False, model_has_desc_act=False)#nous llama2
    elif modelName=='wizard':
        llama_model = get_model("WizardLM-7B-uncensored-GPTQ-4bit-128g", triton=False, model_has_desc_act=False)
    #llama_model = get_model("WizardLM-7B-uncensored-GPTQ-4bit-128g", triton=False, model_has_desc_act=False)
    #llama_model = get_model("gptq_model-4bit-64g", triton=False, model_has_desc_act=False)
    #llama_model = get_model("model", triton=False, model_has_desc_act=False)
    #llama_model = get_model("vicuna-7B-1.1-GPTQ-4bit-128g", triton=False, model_has_desc_act=False)

    llm = pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=llama_tokenizer,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        #trust_remote_code=True,#falcon/mpt
    )

    return llama_tokenizer,llm