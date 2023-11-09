import sys
sys.path.append("D:\\img\\IP-Adapter\\")


from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline, DiffusionPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, AutoencoderTiny, DDIMInverseScheduler, DDIMScheduler
import hashlib
import os
import uuid
from PIL import Image
from load_llama_model import getllama, Chatbot
from ip_adapter import IPAdapterXL, IPAdapterPlus
import gc
import numpy as np
import tomesd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import random



# from pickScore import calc_probs


# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# import torchaudio
# from audiocraft.models import MusicGen
# import soundfile as sf
# import librosa
# from pydub import AudioSegment


pipe, text_generator, tokenizer, cfg = None, None, None, None

image_prompt_file = None
attack_names_template = None
descriptions_template = None
llm = None
img2img = None
ref_pipe = None
text2music = None

ip_model = None

ip_xl = False

do_save_memory = True

chatbot = None


def setup(
    _image_prompt_file="image_prompts.txt",
    _attack_names_template="attack_names.txt",
    _descriptions_template="attack_descriptions.txt",
    model_id="xyn-ai/anything-v4.0",
    textModel="EleutherAI/gpt-neo-2.7B",
    _use_llama=True,
    upscale_model=None,
    vaeModel=None,
    llamaModel="nous-llama2-7b",
    lora=None,
    # ip_adapter_base_model="D:\\img\\auto1113\\stable-diffusion-webui\\models\\Stable-diffusion\\dreamshaperXL10_alpha2Xl10.safetensors",
    # ip_image_encoder_path = "D:\\img\\IP-Adapter\\IP-Adapter\\sdxl_models\\image_encoder",
    # ip_ckpt = "D:\\img\\IP-Adapter\\IP-Adapter\\sdxl_models\\ip-adapter_sdxl.bin",
    #ip_adapter_base_model="D:\\img\\auto1113\\stable-diffusion-webui\\models\\Stable-diffusion\\reliberate_v20.safetensors",
    ip_adapter_base_model = "SG161222/Realistic_Vision_V4.0_noVAE",
    ip_image_encoder_path="D:\\img\\IP-Adapter\\IP-Adapter\\models\\image_encoder",
    ip_ckpt="D:\\img\\IP-Adapter\\IP-Adapter\\models\\ip-adapter-plus_sd15.bin",
    ip_vae_model_path = "stabilityai/sd-vae-ft-mse",
    #ip_adapter_base_model="waifu-diffusion/wd-1-5-beta2",
    #ip_ckpt="D:\\img\\IP-Adapter\\IP-Adapter\\models\\wd15_ip_adapter_plus.bin",
    #ip_vae_model_path = "redstonehero/kl-f8-anime2"
    save_memory=True,
    need_img2img=True,
    need_ipAdapter=True,
    need_music=True
):

    global pipe, text_generator, tokenizer, cfg, image_prompt_file, attack_names_template, descriptions_template, llm, img2img, ref_pipe, text2music
    image_prompt_file = _image_prompt_file
    attack_names_template = _attack_names_template
    descriptions_template = _descriptions_template

    global ip_model, ip_xl

    global use_llama

    global do_save_memory
    do_save_memory = save_memory

    use_llama = _use_llama

    ip_xl= "XL" in ip_adapter_base_model

    print("LOADING IMAGE MODEL")

    if 'xl' in model_id.lower():

        if model_id.endswith(".ckpt") or model_id.endswith(".safetensors"):

            pipe = StableDiffusionXLPipeline.from_single_file(
                model_id, torch_dtype=torch.float16, use_safetensors=True
            )

        else:                
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True
            )

        if lora is not None:
            pipe.load_lora_weights(lora)

        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        pipe.enable_vae_tiling()
        #pipe.vae = AutoencoderTiny.from_pretrained(
        #    "madebyollin/taesdxl", torch_dtype=torch.float16)

        # img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        #    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        # )#todo fixme

        # check for upscale model
        if upscale_model is None:
            upscale_model = model_id


        #chec if we need img2img
        if need_img2img:
        
            if upscale_model.endswith(".ckpt") or upscale_model.endswith(".safetensors"):
            
                img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    upscale_model, torch_dtype=torch.float16, use_safetensors=True)
            else:
                img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    upscale_model, torch_dtype=torch.float16, use_safetensors=True)
                # img2img.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
            img2img.enable_vae_tiling()


            #move to cuda if not saving memory
            if do_save_memory==False:
                pipe = pipe.to("cuda")
                img2img = img2img.to("cuda")
            

    else:
        print("LOADING IMG2iMG MODEL")

        # check if vae is None
        if vaeModel is not None:
            vae = AutoencoderKL.from_pretrained(
                vaeModel, torch_dtype=torch.float16)
        else:
            vae = None

        # check if model_id is a .ckpt or .safetensors file
        if model_id.endswith(".ckpt") or model_id.endswith(".safetensors"):
            pipe = StableDiffusionPipeline.from_single_file(model_id,
                                                            torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)

        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.safety_checker = None
        tomesd.apply_patch(pipe, ratio=0.5)

        if vae is not None:
            pipe.vae = vae

        # pipe = pipe.to("cuda")

        # move pipe to CPU
        pipe = pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()


        if need_img2img:

            dummy_path = "runwayml/stable-diffusion-v1-5"

            # load upscale model
            if upscale_model is not None:
                # check if model_id is a .ckpt or .safetensors file
                if upscale_model.endswith(".ckpt") or model_id.endswith(".safetensors"):
                    uppipe = StableDiffusionPipeline.from_single_file(upscale_model,
                                                                    torch_dtype=torch.float16)
                else:
                    uppipe = StableDiffusionPipeline.from_pretrained(
                        upscale_model, torch_dtype=torch.float16)

            else:
                uppipe = pipe

            uppipe.scheduler = UniPCMultistepScheduler.from_config(
                uppipe.scheduler.config)
            uppipe.enable_attention_slicing()
            uppipe.enable_xformers_memory_efficient_attention()
            uppipe.safety_checker = None
            tomesd.apply_patch(uppipe, ratio=0.5)

            if vae is not None:
                uppipe.vae = vae

            # image to image model
            if model_id.endswith(".ckpt") or model_id.endswith(".safetensors"):

                img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    dummy_path,  # dummy model
                    # revision=revision,
                    scheduler=uppipe.scheduler,
                    unet=uppipe.unet,
                    vae=uppipe.vae,
                    safety_checker=uppipe.safety_checker,
                    text_encoder=uppipe.text_encoder,
                    tokenizer=uppipe.tokenizer,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    cache_dir="./AI/StableDiffusion"
                )

            else:
                img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    # revision=revision,
                    scheduler=uppipe.scheduler,
                    unet=uppipe.unet,
                    vae=uppipe.vae,
                    safety_checker=uppipe.safety_checker,
                    text_encoder=uppipe.text_encoder,
                    tokenizer=uppipe.tokenizer,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    cache_dir="./AI/StableDiffusion"
                )

            del uppipe

            img2img.enable_attention_slicing()
            img2img.enable_xformers_memory_efficient_attention()
            tomesd.apply_patch(img2img, ratio=0.5)

            # move img2img to CPU
            if save_memory:
                img2img = img2img.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                img2img = img2img.to("cuda")


    if need_ipAdapter:

        # load ip adapter
        print("LOADING IP ADAPTER")
        # load SDXL pipeline
        if "XL" in ip_adapter_base_model:
            ippipe = StableDiffusionXLPipeline.from_single_file(
                ip_adapter_base_model,
                torch_dtype=torch.float16,
                add_watermarker=False,
            )
            ippipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", torch_dtype=torch.float16).to('cuda')
            
            ippipe = ippipe.to('cuda')
            ip_model = IPAdapterXL(ippipe, ip_image_encoder_path, ip_ckpt, 'cuda')
        
        
        else:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
            ip_vae = AutoencoderKL.from_pretrained(ip_vae_model_path).to(dtype=torch.float16)
            ippipe = StableDiffusionPipeline.from_pretrained(
                ip_adapter_base_model,
                torch_dtype=torch.float16,
                scheduler=noise_scheduler,
                vae=ip_vae,
                feature_extractor=None,
                safety_checker=None
            )
            ippipe = ippipe.to('cuda')
            ip_model = IPAdapterPlus(ippipe, ip_image_encoder_path, ip_ckpt, 'cuda', num_tokens=16)
            
        # move to cpu
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cpu')
            ip_model.pipe = ip_model.pipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
        else:
            ip_model.image_encoder = ip_model.image_encoder.to('cuda')
            ip_model.pipe = ip_model.pipe.to('cuda')
            

        print("LOADED IP ADAPTER", ip_model)

    if need_music:

        print("LOADING MUSIC MODEL")

        # text to music model
        # text2music = MusicGen.get_pretrained('small')

        cfg = {
            "genTextAmount_min": 30,
            "genTextAmount_max": 100,
            "no_repeat_ngram_size": 8,
            "repetition_penalty": 2.0,
            "MIN_ABC": 4,
            "num_beams": 2,
            "temperature": 2.0,
            "MAX_DEPTH": 5
        }


def generate_music(description, duration=8, save_dir="./static/samples"):
    text2music.set_generation_params(duration=duration)
    wav = text2music.generate([description])
    sample_rate = 32000
    # generate unique filename .mp3
    filename = str(uuid.uuid4()) + ".mp3"
    # add filename to save_dir
    filename = os.path.join(save_dir, filename)
    # save file
    wav = wav.cpu()
    normalized_audio_tensor = wav / torch.max(torch.abs(wav))
    # convert tensor to numpy array
    single_audio = normalized_audio_tensor[0, 0, :].numpy()
    sf.write('temp.wav', single_audio, sample_rate)
    AudioSegment.from_wav('temp.wav').export(filename, format='mp3')


def generate_attributes(level):
    attributes = ["Strength", "Dexterity", "Wisdom",
                  "Intelligence", "Constitution", "Charisma"]
    total_points = level * 10

    # Generate random partitions of total_points
    partitions = sorted(random.sample(
        range(1, total_points), len(attributes) - 1))
    partitions = [0] + partitions + [total_points]

    # Calculate the differences between adjacent partitions
    attribute_values = {
        attributes[i]: partitions[i + 1] - partitions[i]
        for i in range(len(attributes))
    }

    return attribute_values


def generate_attacks(level, attributes):
    num_attacks = random.randint(1, 3)
    attacks = []

    for _ in range(num_attacks):
        prompt = generate_prompt(attack_names_template)
        # Generate another prompt for the attack description
        description = generate_prompt(descriptions_template)
        # You can adjust the damage calculation based on attributes if desired
        damage = random.randint(1, level * 2)

        attack = {
            "name": prompt,
            "description": description,
            "damage": damage
        }

        attacks.append(attack)

    return attacks


def generate_level_and_rarity(level=None):
    # Adjust these probabilities as desired
    if level is None:
        level_probabilities = [0.5, 0.25, 0.15, 0.07, 0.03]
        level = random.choices(range(1, 6), weights=level_probabilities)[0]

    rarity_mapping = {1: "Bronze", 2: "Bronze",
                      3: "Silver", 4: "Silver", 5: "Platinum"}
    rarity = rarity_mapping[level]

    return level, rarity


def generate_image(prompt, prompt_suffix="", width=512, height=512,
                   n_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                   num_inference_steps=15, batch_size=1,
                   ref_image=None,
                   style_fidelity=1.0,
                   attention_auto_machine_weight=1.0,
                   gn_auto_machine_weight=1.0,
                   ref_image_scale=0.6):

    global pipe, ref_pipe
    global ip_pipe

    # add prompt suffix
    prompt += prompt_suffix

    if ref_image is not None:
        '''
        #move pipe to cuda
        ref_pipe = ref_pipe.to("cuda")

        images = ref_pipe([prompt]*batch_size, negative_prompt=[n_prompt]*batch_size,
                          width=width, height=height, num_inference_steps=num_inference_steps, ref_image=ref_image,
                          style_fidelity=style_fidelity,
                          attention_auto_machine_weight=attention_auto_machine_weight,
                          gn_auto_machine_weight=gn_auto_machine_weight
                          ).images


        #move pipe to cpu and clear cache
        ref_pipe = ref_pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        '''
        # use ip adapter
        # move ip_model to cuda
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cuda')
            ip_model.pipe = ip_model.pipe.to('cuda')

        print("GOT REerence image, scale", ref_image_scale)


        if ip_xl:
            images = ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=30, seed=420,
                                    prompt=prompt+prompt_suffix, scale=ref_image_scale)
        else:
            images = ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=30, seed=420,
                                    prompt=prompt+prompt_suffix, scale=ref_image_scale)

        # move ip_model to cpu
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cpu')
            ip_model.pipe = ip_model.pipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

    else:
        # move pipe to cuda
        if do_save_memory:
            pipe = pipe.to("cuda")

        images = pipe([prompt]*batch_size, negative_prompt=[n_prompt]*batch_size,
                      width=width, height=height, num_inference_steps=num_inference_steps).images

        # move pipe to cpu and clear cache
        if do_save_memory:
            pipe = pipe.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    # choose top scoring image
    image = images[0]

    return image


def upscale_image(image, prompt,
                  n_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                  width=1024, height=1024,
                  num_inference_steps=15,
                  strength=0.25):

    global img2img

    # move img2img to cuda
    if do_save_memory:
        img2img = img2img.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()

    # resize image
    image = image.resize((width, height), Image.LANCZOS)

    img2 = img2img(
        prompt=prompt,
        negative_prompt=n_prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=num_inference_steps,
    ).images[0]

    # move to cpu and clear cache
    if do_save_memory:
        img2img = img2img.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    return img2


def generate_prompt(template_file, kwargs=None, max_new_tokens=60):

    global llm, use_llama

    template = open(template_file, "r").read()

    # find {TEXT} in template and replace with generated text
    if "{TEXT}" in template:
        index = template.find("{TEXT}")
        template = template[:index]+"\n"

    # formate template using kwargs
    if kwargs is not None:
        template = template.format(**kwargs)

    # print("huh?",template,kwargs)

    # strip whitespace (for luck)
    template = template.strip()

    if use_llama:
        # move llm to cuda
        # llm = llm.cuda()
        # llm.cuda()#doesn't work, don't know why... ignore for now

        # generate text
        result = llm(template,
                     max_new_tokens=max_new_tokens,
                     do_sample=True,
                     num_beams=2,
                     no_repeat_ngram_size=12,
                     temperature=2.0)
        start_index = template.rfind(":")

        generated_text = result[0]['generated_text'][start_index+1:]

        print("got text", result[0]['generated_text'])

        # move to cpu and clear cache
        # llm = llm.to("cpu")
        # llm.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    else:
        inputs = tokenizer(template, return_tensors="pt", return_attention_mask=False)
        #move inputs to cuda
        inputs['input_ids']=inputs['input_ids'].to('cuda')
        amt = inputs['input_ids'].shape[1]
        outputs = text_generator.generate(**inputs, 
                                          max_length=amt+cfg["genTextAmount_max"],
                                          do_sample=True, temperature=0.2, top_p=0.9, use_cache=True, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
        _generated_text = tokenizer.batch_decode(outputs)[0]
        start_index = template.rfind(":")
        generated_text = _generated_text[start_index+1:]

        #get rid of <|endoftext|>
        generated_text=generated_text.replace("<|endoftext|>","")

        '''

        inputs = tokenizer(
            template, return_tensors="pt")
        
        
        input_ids=inputs.input_ids
        amt = input_ids.shape[1]

        

        generated_text = text_generator.generate(
            inputs,
            do_sample=True,
            min_length=amt+cfg["genTextAmount_min"],
            max_length=amt+cfg["genTextAmount_max"],
            #return_full_text=False,
            no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
            repetition_penalty=cfg["repetition_penalty"],
            num_beams=cfg["num_beams"],
            temperature=cfg["temperature"]
        )[0]["generated_text"]

        

        outputs = text_generator.generate(**inputs, max_length=amt+cfg["genTextAmount_min"], do_sample=True, temperature=0.2, top_p=0.9, use_cache=True, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.batch_decode(outputs)[0]

        '''
        
        
        

    # prompt is first non empty line w/o colon
    new_prompt = "default prompt"
    lines = generated_text.split("\n")
    for line in lines:
        if len(line.strip()) > 0 and ":" not in line:
            new_prompt = line
            break

    if new_prompt == "default prompt":
        print("WARNING: no prompt generated")
        new_prompt = generated_text

    # print(template,"\n==\n",generated_text,"\n==\n",new_prompt)

    return new_prompt


def hash(s):
    sha256_hash = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return sha256_hash


def generate_card(num_inference_steps=15, prompt_suffix=", close up headshot, anime portrait, masterpiece", level=None):
    level, rarity = generate_level_and_rarity(level=level)
    attributes = generate_attributes(level)
    prompt = generate_prompt(image_prompt_file)
    image = generate_image(
        prompt, num_inference_steps=num_inference_steps, prompt_suffix=prompt_suffix)
    # hash prompt to get filename
    image_file_name = "./static/images/"+hash(prompt)+".png"
    image.save(image_file_name)
    # attacks = generate_attacks(level, attributes)
    attacks = []

    card = {"level": level,
            "rarity": rarity,
            "attributes": attributes,
            "image": image_file_name,
            "attacks": attacks,
            "description": prompt}

    return card


def generate_background_image(background_prompt_file="./background_prompts.txt", suffix="high quality landscape painting"):
    prompt = generate_prompt(background_prompt_file)
    image = generate_image(prompt, width=768, height=512, prompt_suffix=suffix)
    image_file_name = "./static/images/"+hash(prompt)+".png"
    image.save(image_file_name)
    return {"description": prompt, "image": image_file_name}


def generate_map_image(map_prompt_file="./map_prompts.txt", suffix="hand drawn map, detailed, full color"):
    prompt = generate_prompt(map_prompt_file)
    image = generate_image(prompt, width=768, height=512, prompt_suffix=suffix)
    image_file_name = "./static/images/"+hash(prompt)+".png"
    image.save(image_file_name)
    return {"description": prompt, "image": image_file_name}


if __name__ == "__main__":
    setup()
    card = generate_card()
    print(card)
