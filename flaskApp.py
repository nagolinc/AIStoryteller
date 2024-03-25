from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
from datetime import datetime
import os
import argparse
import generation_functions
from generation_functions import format_messages

from datetime import datetime


import re
import random


# import openai

import json

from llama_cpp.llama import Llama, LlamaGrammar

# import subprocess
import subprocess

import threading
import time

from threading import Lock

text_generation_lock = Lock()


# Put your URI end point:port here for your local inference server (in LM Studio)
# openai.api_base = "http://localhost:1234/v1"
# Put in an empty API Key
# openai.api_key = ""

# Adjust the following based on the model type
# Alpaca style prompt format:
prefix = "<|im_start|>"
suffix = "<|im_end|>"

# prefix = "### Instruction:\n"
# suffix = "\n### Response:"

app = Flask(__name__)


class Chatbot:
    def __init__(self, systemPrompt, max_messages=7):
        self.systemPrompt = systemPrompt
        self.messages = [{"role": "system", "content": systemPrompt}]
        self.max_messages = max_messages

        self.defaultOutput = """{
        "sceneAction": "and then something completely different happened",
        "imageDescription": "a picture of something exciting",
        "musicDescription": "some music",
        "options": [
            "option 1",
            "option 2",
            "option 3"
        ]       
        }"""

        self.summary = ""

        self.plot_twist_counter = 0
        self.next_plot_twist = 0

    # use llama
    def chat(self, text, temperature=0.0, max_messages=7):
        # if story summary is empty, use text as story summary
        if self.summary == "":
            self.summary = text
            messageText = """
storySummary: {self.summary}
""".format(
                self=self
            )

        else:
            # roll a die to see if we should add a new summary
            dieValue = random.randint(1, 20)
            if dieValue == 1:
                dieResult = "critical fail"
            elif dieValue < args.failure_threshold:
                dieResult = "fail"
            elif dieValue < 20:
                dieResult = "success"
            else:
                dieResult = "critical success"

            # if args.accumulate_summary:
            if True:
                messageText = "story summary: {self.summary}".format(self=self)
            else:
                messageText = ""

            messageText += """Player action: {text}
Action result: {dieResult}
""".format(
                self=self, text=text, dieResult=dieResult
            )

        # add scene hint
        if args.scene_hint != "":
            # messageText = f">{args.scene_hint}\n{messageText}"
            messageText += f">{args.scene_hint}\n"

        # add plot twist
        if self.plot_twist_counter >= self.next_plot_twist:
            messageText += (
                "include the following plot twist:\n"
                + random.choice(plot_twists)
                + "\n"
            )
            self.plot_twist_counter = 1
            self.next_plot_twist = random.randint(
                args.plot_twist_frequency[0], args.plot_twist_frequency[1]
            )
        else:
            print(
                "plot twist counter", self.plot_twist_counter, "/", self.next_plot_twist
            )
            self.plot_twist_counter += 1

        message = {"role": "user", "content": messageText}
        self.messages.append(message)

        # trim self.messages if needed
        if len(self.messages) > self.max_messages:
            self.messages = [self.messages[0]] + self.messages[-self.max_messages + 1 :]

        f = format_messages(self.messages, mode=args.llm_mode)

        # print("SENDING", format_messages(self.messages))
        print("SENDING", f)
        # print("SENDING", messageText)

        response = llm(
            f,
            grammar=grammar,
            max_tokens=args.max_tokens,
            repeat_penalty=1.2,
            # frequency_penalty=1.0,
            # presence_penalty=1.0,
            temperature=temperature,
        )

        outputText = response["choices"][0]["text"]

        print("GOT RESPONSE", response)
        print("GOT RESPONSE\n" + outputText)

        if not self.validateJson(outputText):
            print("\n\n\n======\n!!!!INVALID JSON!!!!", outputText)

            # use default output
            outputText = self.defaultOutput

            print("======\n\n\n")

        # let's remove the 'options' field from the json
        jsonData = json.loads(outputText)
        jsonData.pop("options")
        newOutputText = json.dumps(jsonData, indent=4)

        # add message to self.messages
        outMessage = {"role": "assistant", "content": newOutputText}

        self.messages.append(outMessage)

        # print("WHAT???",self.messages)

        # append summary
        data = json.loads(outputText)

        if args.accumulate_summary:

            # if summary doesn't end in ". " then add ".
            if self.summary.endswith(". "):
                pass
            elif self.summary.endswith("."):
                self.summary += " "
            else:
                self.summary += ". "
            self.summary += data["summary"]

        # rewrite options
        if False:  # this doesn't work and is slow
            options = data["options"]
            rewritten = self.rewrite_options(options, self.messages)
            for i in range(len(options)):
                data["options"][i] = rewritten[i]["rewrittenSentence"]

            outputText = json.dumps(data)

        return outputText

    def rewrite_options(self, options, messages=[]):

        input_text = json.dumps(options)

        sp = specific_systemPrompt
        someNames = random.sample(names, 3)
        sp += "\n Herea are some names you can use if you want: " + " ".join(someNames)

        messages = messages + [
            {"role": "system", "content": specific_systemPrompt},
            {"role": "user", "content": input_text},
        ]

        result = llm(
            format_messages(messages, mode=args.llm_mode),
            grammar=specific_grammar,
            max_tokens=1000,
            repeat_penalty=1.2,
            # frequency_penalty=1.0,
            # presence_penalty=1.0,
            temperature=1.0,
        )

        print("SPECIFIC RESPONSE", result)

        data = json.loads(result["choices"][0]["text"])

        return data["result"]

    def validateJson(self, text):
        try:
            json.loads(text)
        except ValueError as e:
            return False

        # assert that json contains image,description, summary and options
        requiredFields = [
            "imageDescription",
            "sceneAction",
            "musicDescription",
            # "summary",
            "options",
        ]
        for field in requiredFields:
            if field not in text:
                return False

        return True


def generate_video(prompt, image):
    # let's create a unique filename based off of datetime and the prompt
    # replace non alphanumeric in prompt with _ and trim to 100 chars
    prompt_filename = re.sub(r"\W+", "_", prompt)[:100]
    # prepend timestamp
    prompt_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt_filename}"

    temp_video_path = f"./static/samples/temp_video.mp4"
    video_path = f"./static/samples/{prompt_filename}_video.mp4"

    # image = image.resize((args.movie_size[0], args.movie_size[1]))

    # save image for inspection
    image.save(f"./static/samples/temp.png")
    # filename = generateGif(prompt, image, prompt_suffix="", n_prompt=None, num_frames=16,do_upscale=True,width=576,height=320)
    # video_path = f"./static/samples/{filename}"

    # temp_video_path_up = generate_video_animdiff(
    temp_video_path_up = generation_functions.generate_video_camera_transforms(
        prompt,
        image,
        temp_video_path,
        prompt_suffix=args.prompt_suffix,
        n_prompt=args.n_prompt,
        num_frames=args.num_frames,
        base_fps=args.base_fps,
        do_upscale=args.upscale,
        upscale_size=args.video_upscale_size,
        final_size=args.movie_size,
        img2img_strength=args.img2img_strength,
        num_video_interpolation_steps=args.num_video_interpolation_steps,
    )

    # Re-encode the video using FFmpeg
    cmd = [
        "ffmpeg",
        "-i",
        temp_video_path_up,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        video_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return video_path


chatbots = {}


allVideos = {}
allPrompts=[]
currentPrompt = None
lastPrompt = None
currentIndex = 0
numExtraVideos = 0
currrentPrompt_isNew = True
currentMusic = None


userPrompts = []


@app.route("/chat", methods=["POST"])
def chat():
    global currentPrompt, allVideos, currentIndex, numExtraVideos, currrentPrompt_isNew, currentMusic, lastPrompt,allPrompts

    with text_generation_lock:
        user_input = request.json["user_input"]

        # username and story_id
        username = request.json["username"]
        story_id = request.json["story_id"]

        # check if we have a chatbot for this story_id
        key = f"{username}_{story_id}"
        if key not in chatbots:
            # create new chatbot
            chatbots[key] = Chatbot(systemPrompt)
            userPrompts.append(user_input)
            thisIsANewStory = True
        else:
            thisIsANewStory = False

        chatbot = chatbots[key]

        chat_output = chatbot.chat(user_input, temperature=args.temperature)

        # parse json
        chat_output = json.loads(chat_output)

        if thisIsANewStory:

            print("WE NEED TO DO THE FIRST VIDEO")

            if args.useAnimDiff:
                prompt = chat_output["imageDescription"] + args.prompt_suffix
                video_path = generation_functions.generateVideoAnimDiff(
                    prompt,
                    width=args.image_sizes[0],
                    height=args.image_sizes[1],
                )
                
                if args.vid2vid:
                    video_path = generation_functions.vid2vid(
                        video_path,
                        prompt,
                        width=args.image_sizes[2],
                        height=args.image_sizes[3],
                        strength=args.vid2vid_strength,
                    )

            else:

                # generate image
                image = generation_functions.generate_image(
                    chat_output["imageDescription"],
                    prompt_suffix=args.prompt_suffix,
                    width=args.image_sizes[0],
                    height=args.image_sizes[1],
                    num_inference_steps=args.num_inference_steps,
                    cfg_scale=args.cfg_scale,
                    clip_skip=args.clip_skip,
                )

                image.save("static/samples/temp0.png")

                prompt = f"{chat_output['sceneAction']}{args.prompt_suffix}"

                # upscale
                if args.upscale:
                    # resize to image_sizes[2],image_sizes[3]
                    img = image.resize(
                        (args.image_sizes[2], args.image_sizes[3]), Image.LANCZOS
                    )
                    image = generation_functions.lcm_image_to_image(
                        img, prompt, prompt_suffix=args.prompt_suffix
                    )

                    image.save("static/samples/temp1.png")

                # save image in static/samples
                # replce non alphanumeric in chat_output["image"] with _ and tripm to 100 chars
                image_filename = re.sub(r"\W+", "_", chat_output["imageDescription"])[
                    :100
                ]
                # prepend timestamp
                image_filename = (
                    f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image_filename}.png"
                )
                # image path is in /static/samples
                image_path = os.path.join("static", "samples", image_filename)
                image.save(image_path)

                chat_output["image_path"] = image_path

                # generate video
                video_path = generate_video(chat_output["sceneAction"], image)

            # generate music
            music_path = generation_functions.generate_music(
                chat_output["musicDescription"]
            )

            # add video to allVideos[prompt] and set currentPrompt
            currentPrompt = chat_output["imageDescription"]
            lastPrompt = currentPrompt
            allPrompts+=[currentPrompt]
            allVideos[currentPrompt] = [video_path]
            currrentPrompt_isNew = False
            currentIndex = 0
            numExtraVideos = 0

            chat_output["video_path"] = video_path
            chat_output["music_path"] = music_path
        else:
            lastPrompt = currentPrompt
            currentPrompt = chat_output["imageDescription"]
            currrentPrompt_isNew = True
            print("SAVING MUSIC AND VIDEO FOR LATER")

        # sceneAction
        voice = 0
        gender = "male"
        text = chat_output["sceneAction"]
        speech_path, duration = generation_functions.textToSpeech(text, voice, gender)

    chat_output["speech_path"] = speech_path
    return jsonify(chat_output)


# render template chat.html

saved_music = {}


@app.route("/get_music", methods=["POST", "GET"])
def get_music():
    music_description = request.values.get("musicDescription")

    if music_description in saved_music:
        music_path = saved_music[music_description]
    else:
        with text_generation_lock:
            music_path = generation_functions.generate_music(music_description)
            saved_music[music_description] = music_path

    return jsonify({"music_path": music_path})


def generate_another_video():
    global currentPrompt, allVideos, currentIndex, numExtraVideos, currrentPrompt_isNew, allPrompts

    if currentPrompt is None:
        return None

    if numExtraVideos > 3:
        return None

    with text_generation_lock:
        print("GENERATING ANOTHER VIDEO, with prompt", currentPrompt)

        if args.useAnimDiff:
            prompt = currentPrompt + args.prompt_suffix
            video_path = generation_functions.generateVideoAnimDiff(
                prompt,
                width=args.image_sizes[0],
                height=args.image_sizes[1], 
            )
            
            if args.vid2vid:
                video_path = generation_functions.vid2vid(
                    video_path,
                    prompt,
                    width=args.image_sizes[2],
                    height=args.image_sizes[3],
                    strength=args.vid2vid_strength,
                )

        else:

            image = generation_functions.generate_image(
                currentPrompt,
                prompt_suffix=args.prompt_suffix,
                width=args.image_sizes[0],
                height=args.image_sizes[1],
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                clip_skip=args.clip_skip,
            )

            prompt = f"{currentPrompt}{args.prompt_suffix}"

            # upscale
            if args.upscale:
                # resize to image_sizes[2],image_sizes[3]
                img = image.resize(
                    (args.image_sizes[2], args.image_sizes[3]), Image.LANCZOS
                )
                image = generation_functions.lcm_image_to_image(img, prompt)

            # save image in static/samples
            # replce non alphanumeric in chat_output["image"] with _ and tripm to 100 chars
            image_filename = re.sub(r"\W+", "_", currentPrompt)[:100]
            # prepend timestamp
            image_filename = (
                f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image_filename}.png"
            )
            # image path is in /static/samples
            image_path = os.path.join("static", "samples", image_filename)
            image.save(image_path)

            # generate video
            video_path = generate_video(currentPrompt, image)

        if currrentPrompt_isNew:
            allPrompts+=[currentPrompt]
            allVideos[currentPrompt] = [video_path]
            currrentPrompt_isNew = False
            numExtraVideos = 1
            currentIndex = 0
        else:
            # add to allVideos[prompt]
            allVideos[currentPrompt].append(video_path)
            numExtraVideos += 1
            currentIndex = len(allVideos[currentPrompt]) - 2

    return video_path


def generate_another_video_thread():
    while True:
        path = generate_another_video()
        time.sleep(1)
        if path is None:
            print("no currentPrompt,SLEEPING")
            time.sleep(4)


def start_background_thread():
    thread = threading.Thread(target=generate_another_video_thread, daemon=True)
    thread.start()


@app.route("/get_video/")
def get_video():
    global currentPrompt, allVideos, currentIndex, numExtraVideos, currrentPrompt_isNew

    if currentPrompt in allVideos:
        thisPrompt = currentPrompt
    else:
        thisPrompt = None
        for prompt in allPrompts[::-1]:
            if prompt in allVideos:
                thisPrompt = prompt
                break
    
    print('returning video for prompt',thisPrompt)

    if currentIndex < len(allVideos[thisPrompt]) - 1:
        currentIndex += 1
        numExtraVideos -= 1
    else:
        # shuffle videos
        random.shuffle(allVideos[thisPrompt])
        currentIndex = 0
        numExtraVideos = 0

    video_path = allVideos[thisPrompt][currentIndex]
    return jsonify({"video_path": video_path})


@app.route("/")
def index():
    return render_template("chat.html")


samplePrompts = [
    "you are jack, a brave hero fights a dragon",
    "you are emma, a witch from the future fights a evil warlord",
    "you are bob, a space pirate fights a alien",
    "you are jane, a princess fights a evil queen",
    "you are john, a knight fights a evil king",
]


@app.route("/get_prompt/")
def get_prompt():
    with text_generation_lock:
        random.seed()  # apparently dataset is doing it?
        # use prompt suppliment
        prompt_suppliment = ""
        for prompt_suppliments in all_prompt_suppliments:
            # prompt_suppliment += random.choice(list(prompt_suppliments.keys()))
            l = list(prompt_suppliments.keys())
            index = random.randint(0, len(l) - 1)
            print("index", index)
            to_add = l[index]
            print("to_add", to_add)
            prompt_suppliment += to_add
            prompt_suppliment += ", "
        # remove last comma
        prompt_suppliment = prompt_suppliment[:-2]

        print("prompt_suppliment", prompt_suppliment)

        # remove last space
        prompt_suppliment = prompt_suppliment.strip()

        if len(userPrompts) > 5:
            gotPrompts = random.sample(userPrompts, 5)
        else:
            gotPrompts = userPrompts + random.sample(
                samplePrompts, 5 - len(userPrompts)
            )

        prompt = generation_functions.create_prompt(
            gotPrompts,
            prompt_hint=args.prompt_hint,
            prompt_suppliment=prompt_suppliment,
            temperature=args.temperature,
            max_tokens=1000,
        )

        return jsonify({"prompt": prompt})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a story")

    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-0.9",
        help="model id",
    )

    # prompt suffix
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default=", anime, high resolution digital art, studio ghibili, masterpice, close up headshot",
        help="prompt suffix",
    )

    # image sizes, default = 1024,1024,2048,2048
    parser.add_argument(
        "--image_sizes",
        type=int,
        nargs="+",
        default=[768, 424, 1536, 848],
        help="image sizes",
    )

    # llm model
    parser.add_argument(
        "--llm_model",
        type=str,
        default="D:\lmstudio\TheBloke\Mistral-7B-OpenOrca-GGUF\mistral-7b-openorca.Q5_K_M.gguf",
        help="llm model",
    )

    # failure threshold (from 1 to 20)
    parser.add_argument(
        "--failure_threshold",
        type=int,
        default=8,
        help="this is the number that must be rolled on a 20 sided die to succeed",
    )

    # response template = response_template.json
    parser.add_argument(
        "--response_template",
        type=str,
        default="response_template.json",
        help="response template",
    )

    # text completion model (llama, VLLM or GPT)
    parser.add_argument(
        "--text_completion_model",
        type=str,
        default="llama",
        help="text completion model (llama, VLLM or GPT)",
    )

    # --do_not_upscale (set args.upscale==false)
    parser.add_argument("--do_not_upscale", dest="upscale", action="store_false")

    # num_inference_steps
    parser.add_argument(
        "--num_inference_steps", type=int, default=12, help="num_inference_steps"
    )

    # system prompt file "story_prompt.txt"
    parser.add_argument(
        "--system_prompt_file",
        type=str,
        default="story_prompt.txt",
        help="system prompt file",
    )

    # video upscale size
    parser.add_argument("--video_upscale_size", type=int, nargs=2, default=[768, 424])

    # resolution of the final output movie
    parser.add_argument("--movie_size", type=int, nargs=2, default=[768, 424])

    # negative prompt
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="sexy, naked, topless, nude, nsfw, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
    )

    # number of frames to generate
    parser.add_argument("--num_frames", type=int, default=24)

    # grammar file "grammar.gbnf"
    parser.add_argument("--grammar_file", type=str, default="grammar.gbnf")

    # scene hint (a message to send before each scene)
    parser.add_argument("--scene_hint", type=str, default="")

    # temperature
    parser.add_argument("--temperature", type=float, default=1.0)

    # prompt suppliments
    parser.add_argument("--prompt_suppliment_files", type=str, nargs="+", default=[])

    # prompt hint
    parser.add_argument("--prompt_hint", type=str, default="")

    # plot twists (plotTwists.txt)
    parser.add_argument("--plot_twists_file", type=str, default="plotTwists.txt")

    # plot twist frequency = [3,5]
    parser.add_argument("--plot_twist_frequency", type=int, nargs=2, default=[3, 5])

    # --llm_mode (default is "im_start")
    parser.add_argument("--llm_mode", type=str, default="im_start")

    # --n_gpu_layers (default is 20)
    parser.add_argument("--n_gpu_layers", type=int, default=60)

    # --sample_prompts_file (default is none)
    parser.add_argument("--sample_prompts_file", type=str, default=None)

    # --max_tokens (default is 8192)
    parser.add_argument("--max_tokens", type=int, default=8192)

    # accumulate summary
    parser.add_argument(
        "--accumulate_summary", dest="accumulate_summary", action="store_true"
    )

    # img2img strength
    parser.add_argument("--img2img_strength", type=float, default=0.5)

    # lcm model (if different from llm model) default is none
    parser.add_argument("--lcm_model", type=str, default=None)

    # clip skip, default =1
    parser.add_argument("--clip_skip", type=int, default=1, help="clip skip")

    # cfg scale
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="cfg scale")

    # --doNotSaveMemory (store false save_memory)
    parser.add_argument("--doNotSaveMemory", dest="save_memory", action="store_false")

    # num_video_interpolation_steps
    parser.add_argument(
        "--num_video_interpolation_steps",
        type=int,
        default=3,
        help="num_video_interpolation_steps",
    )

    # base_fps default 2
    parser.add_argument("--base_fps", type=int, default=2, help="base_fps")

    # --useAnimDiff (default is false)
    parser.add_argument("--useAnimDiff", dest="useAnimDiff", action="store_true")
    
    #animDiffBase default "emilianJR/epiCRealism"
    parser.add_argument('--animDiffBase', type=str, default="emilianJR/epiCRealism")
    
    #vid2vid (store true)
    parser.add_argument("--vid2vid", dest="vid2vid", action="store_true")
    
    #vid2vid_strength default 0.25
    parser.add_argument("--vid2vid_strength", type=float, default=0.25)

    args = parser.parse_args()

    # sample prompts
    if args.sample_prompts_file is not None:
        with open(args.sample_prompts_file) as f:
            samplePrompts = f.readlines()
            samplePrompts = [x.strip() for x in samplePrompts]
            print("sample prompts", samplePrompts)

    # plot twists
    with open(args.plot_twists_file) as f:
        plot_twists = f.readlines()

    # prompt suppliments
    all_prompt_suppliments = []
    for prompt_suppliment_file in args.prompt_suppliment_files:
        # if file is json
        if prompt_suppliment_file.endswith(".json"):
            with open(prompt_suppliment_file) as f:
                prompt_suppliments = json.load(f)
                all_prompt_suppliments.append(prompt_suppliments)
        elif prompt_suppliment_file.endswith(".txt"):
            with open(prompt_suppliment_file) as f:
                prompt_suppliments = {}
                for line in f:
                    line = line.strip()
                    if line:
                        prompt_suppliments[line] = 1
                all_prompt_suppliments.append(prompt_suppliments)

    systemPrompt = open(args.system_prompt_file, "r").read()

    """
    if args.upscale:
        need_lcm_img2img=True
    else:
        need_lcm_img2img=False
    """
    need_lcm_img2img = True

    if args.useAnimDiff:
        need_animdiff = True
        need_txt2img = False
        need_lcm_img2img = False
    else:
        need_animdiff = False
        need_txt2img = True
        need_lcm_img2img = True

    generation_functions.setup(
        model_id=args.model_id,
        need_txt2img=need_txt2img,
        need_img2img=False,
        need_ipAdapter=False,
        need_music=True,
        need_lcm_img2img=need_lcm_img2img,
        need_tts=True,
        lcm_model=args.lcm_model,
        save_memory=args.save_memory,
        need_animDiff=need_animdiff,
        animDiffBase=args.animDiffBase,
        need_vid2vid=args.vid2vid,
    )

    if args.text_completion_model == "llama":
        grammar_text = open(args.grammar_file).read()
        grammar = LlamaGrammar.from_string(grammar_text)
        llm = Llama(
            args.llm_model, n_gpu_layers=args.n_gpu_layers, n_ctx=args.max_tokens
        )
        generation_functions.llm = llm

    elif args.text_completion_model == "VLLM":
        # read example_response_template
        response_template = open(args.response_template, "r").read()

    # specific
    specific_systemPrompt = open("specific_system_prompt.txt", "r").read()
    specific_grammar_text = open("specific.gbnf").read()
    specific_grammar = LlamaGrammar.from_string(specific_grammar_text)
    names = open("names.txt").read().splitlines()

    start_background_thread()
    app.run(debug=True, use_reloader=False)
