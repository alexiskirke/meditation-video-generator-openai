import json
import random
import shutil
import time
from PIL import Image, ImageDraw, ImageFont
import textwrap
import platform
from pedalboard_native import LowShelfFilter
from typing import List
import os
from openai import OpenAI
from .mp3_mixer import MP3Mixer
from .mp3_merger import MP3Merger
import requests
from moviepy.editor import ImageClip, AudioFileClip
from pedalboard import Pedalboard, Reverb, Gain, LowpassFilter
from pedalboard.io import AudioFile

YELLOW = (255, 255, 0, 255)  # used to build the image text. 100% opaque
BLUE = [0, 0, 255]  # used to build the image banner background. opacity filled in later.
WHITE = (255, 255, 255, 255)  # used to build the image banner text. 100% opaque
BLACK = [0, 0, 0]  # used to build the image banner text. 100% opaque
class MeditationVideoGenerator:
    def __init__(self, topic: str = "Mindfulness", length: float = 10, api_key: str = "",
                 base_on_text: bool = False,
                 text: str = "", num_sentences: int = 6,
                 bass_boost: bool = False,
                 balance_even: float = 0.0, balance_odd: float = 0.0,
                 two_voices: bool = False,
                 expand_on_section: bool = False,
                 beautiful_lady: bool = False,
                 sounds_dir: str = "ambient_files",
                 limit_parts: int = 0,
                 num_loops: int = 0,
                 in_spanish: bool = False,
                 affirmations_only: bool = False,
                 binaural: bool = False,
                 all_voices: bool = False,
                 binaural_fade_out_duration: float = 4,
                 binaural_file_gain_db: int = -40,
                 start_beat_freq: float = 5,
                 end_beat_freq: float = 0.5,
                 base_freq: float = 110.0,
                 sample_rate: int = 44100,
                 num_samples_to_chop: int = 30000,
                 fade_in_time: float = 4,
                 fade_out_time: float = 4,
                 expansion_size: int = 4,
                 force_working_dir_overwrite: bool = False,
                 force_voice: str = "",
                 image_background_opacity: float = 0.5,
                 banner_at_bottom: bool = True,
                 banner_height_ratio: float = 0.35,
                 max_banner_words: int = 6,
                 power_ratio = None
                 ):
        """
        Initializes the MeditationGenerator with an OpenAI API key.
        :param topic: Topic of the meditation.
        :param length: Length of the meditation in minutes.
        :param base_on_text: If True, generate a meditation based on the text provided.
        :param text: Text to generate the meditation from.
        :param api_key: OpenAI API key for accessing the speech synthesizer.
        :param num_sentences: Number of sentences in each part of the meditation.
        :param bass_boost: If True, apply bass boost to the audio.
        :param balance_even: The balance for even segments (0.0 is centered, -1.0 is left, 1.0 is right).
        :param balance_odd: The balance for odd segments (0.0 is centered, -1.0 is left, 1.0 is right).
        :param two_voices: If True, use two different voices for odd and even segments.
        :param expand_on_section: If True, expand on each section of the meditation.
        :param beautiful_lady: If True, add a beautiful lady to the video.
        :param sounds_dir: Directory containing ambient sounds.
        :param limit_parts: Limit the number of parts in the meditation using the prompt.
        :param num_loops: allows meditation parts to be looped multiple times.
        :param in_spanish: If True, generate the meditation in Spanish.
        :param affirmations_only: If True, generate affirmations only not meditations.
        :param binaural: If True, generate binaural beats.
        :param all_voices: If True, choose from all available voices.
        :param binaural_fade_out_duration: Duration in seconds of the fade-out at the end of the binaural beats.
        :param binaural_file_gain_db: How much binaural beats should be quieter than the input voice audio.
        :param start_beat_freq: Initial frequency difference for the binaural effect.
        :param end_beat_freq: Final frequency difference for the binaural effect.
        :param base_freq: Base frequency for the binaural beats.
        :param sample_rate: Sample rate for mixer.
        :param num_samples_to_chop: Number of samples to chop off the beginning of the ambient sound files.
        :param fade_in_time: Duration in seconds of the fade-in at the beginning of the ambient sound overlay.
        :param fade_out_time: Duration in seconds of the fade-out at the end of the ambient sound overlay.
        :param ambient_file_gain_db: How much ambient sound should be quieter than the input voice audio.
        :param expansion_size: Size of the expansion for the text, if expand_on_section is True.
        :param force_voice: Allows user to force selection of OpenAI voice, e.g. onyx
        :param image_background_opacity: Opacity of the background image (0 = transparent, 1 = opaque).
        :param banner_at_bottom: If True, the banner is at the bottom of the image.
        :param banner_height_ratio: Ratio of the banner height to the image height.
        :param max_banner_words: Maximum number of words in the banner.
        :param power_ratio: Ratio (higher means louder voice) of the power of the binaural beats or ambient to the power of the voice audio.
        """
        # setting any of these to false switches off that part of the pipeline
        self.pipeline: dict[str, bool] = {
            "texts": True,
            "audio_files": True,
            "combine_audio_files": True,
            "audio_fx": True,
            "background_audio": True,
            "image": True,
            "video": True,
            "keywords": True
        }
        # construct the directory from the introspection of the class
        ambient_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), sounds_dir)
        # if it exists and it is empty of mp3s, then delete it
        if (os.path.exists(ambient_files_dir) and
            not [f for f in os.listdir(ambient_files_dir) if '.mp3' in f]):
            print("Deleting empty ambient files directory.")
            shutil.rmtree(ambient_files_dir)
        # if it doesn't exist then create it and download the files in the urls in
        # the ambient_files_urls.txt file
        if not os.path.exists(ambient_files_dir):
            # without this, the server will say:
            """Not Acceptable!
            An appropriate representation of the requested resource could not be found on this server. This error was generated by Mod_Security."""
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
                'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1',
                'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.105 Mobile Safari/537.36',
                'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
            ]
            headers = {
                'User-Agent': random.choice(user_agents),
                # 'Referer': 'https://example.com',  # Optional: Add if the server requires it
            }
            print("Creating ambient files directory and downloading ambient music.")
            try:
                os.makedirs(ambient_files_dir)
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ambient_files_urls.txt"), "r") as f:
                    urls = f.read().splitlines()
                    for url in urls:
                        print(f"Downloading {os.path.basename(url)}")
                        # Send a GET request to the URL
                        response = requests.get(url, headers=headers)
                        with open(os.path.join(ambient_files_dir, os.path.basename(url)), 'wb') as file:
                                file.write(response.content)
                        time.sleep(random.randint(1, 2))
            except Exception as e:
                # empty the directory
                shutil.rmtree(ambient_files_dir)
                raise e
        self.in_spanish = in_spanish
        self.limit_parts = limit_parts
        self.balance_odd = balance_odd
        self.balance_even = balance_even
        self.voice_choices = ['onyx', 'shimmer'] #, 'echo']
        self.two_voices = two_voices
        self.num_loops = num_loops
        if two_voices:
            choices = self.voice_choices.copy()
            self.voice_odd = random.choice(choices)
            choices.remove(self.voice_odd)
            self.voice_even = random.choice(choices)
        self.expand_on_section = expand_on_section
        self.image_quality = "standard"  # hd
        self.image_model = "dall-e-3"
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        self.subsections = []
        self.subsection_audio_files = []
        self.length = length
        self.topic = topic
        self.topic_based_filename = self.topic.replace(" ", "_")[:20]  # in case topic too long
        # working directory for storing audio files etc
        self.working_directory = self.topic_based_filename
        # create the working directory if it doesnt exist
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        # otherwise clear the directory of files
        elif not force_working_dir_overwrite:
            resp = input(
                f"The working directory '{self.working_directory}' already exists. Do you want to overwrite it? (y/n): ")
            if resp.lower() == "y" or resp.lower() == "yes":
                shutil.rmtree(self.working_directory)
                os.makedirs(self.working_directory)
        else:
            print("WARNING: Forced overwriting of working directory")
            shutil.rmtree(self.working_directory)
            os.makedirs(self.working_directory)
        self.sounds_dir = sounds_dir
        self.mixer = MP3Mixer(mp3_file="TBD", binaural=binaural, working_dir=self.working_directory,
                              sounds_dir=self.sounds_dir,
                              binaural_fade_out_duration=binaural_fade_out_duration,
                              # duration in seconds of the fade-out at the end of the binaural beats
                              # how much binaural beats should be quieter than the input voice audio
                              start_beat_freq=start_beat_freq,  # initial frequency difference for the binaural effect
                              end_beat_freq=end_beat_freq,  # final frequency difference for the binaural effect
                              base_freq=base_freq,  # base frequency for the binaural beats
                              sample_rate=sample_rate,
                              num_samples_to_chop=num_samples_to_chop,
                              # number of samples to chop off the beginning of the ambient sound files
                              fade_in_time=fade_in_time,
                              # duration in seconds of the fade-in at the beginning of the ambient sound overlay
                              fade_out_time=fade_out_time,
                              # duration in seconds of the fade-out at the end of the ambient sound overlay
                              power_ratio=power_ratio,  # how much ambient sound should be quieter than the input voice audio
                              )
        self.merger = MP3Merger(self.subsection_audio_files, duration=int(self.length * 60),
                                balance_odd=self.balance_odd,
                                balance_even=self.balance_even)
        self.bass_boost = bass_boost
        self.beautiful_lady = beautiful_lady
        self.voice_list = ['onyx', 'shimmer', 'echo', 'alloy', 'fable', 'nova']
        if force_voice:
            self.voice = force_voice
        elif all_voices:
            self.voice = random.choice(self.voice_list)
        else:
            self.voice = random.choice(self.voice_list[:2])
        self.pedalboard_fx_list = []
        if self.bass_boost:
            # The below is ideal for 1st male voice, but not first female voice
            if self.voice != 'shimmer':
                self.pedalboard_fx_list += [
                    LowShelfFilter(cutoff_frequency_hz=150, gain_db=5.0, q=1)
                ]
            else:
                self.pedalboard_fx_list += [
                    LowShelfFilter(cutoff_frequency_hz=150, gain_db=3.0, q=1)
                ]
        self.pedalboard_fx_list += [
            LowpassFilter(10000),
            Reverb(room_size=0.04, wet_level=0.04)
        ]
        self.engine = "tts-1-hd"
        self.num_sentences = num_sentences
        self.technique = random.choice(["Watching the breath", "Body sensation", "Watching the thoughts",
                                        "Listening to sounds (but only mention sounds within the meditation track)."])
        if affirmations_only:
            output_type = "affirmation"
        else:
            output_type = "meditation"
        if base_on_text:
            self.prompt = f"""Generate a {output_type} based on the following text. 
            Do not use any contractions at all, for example isn't, there's or we're:\n{text}"""
        else:
            self.prompt = f"""Generate a {output_type} on the topic '{self.topic}'.
            Do not use any contractions at all, for example isn't, there's or we're."""
        self.prompt += f"The {output_type} should be {self.length} minutes long.\n"
        self.prompt += f"""
        Start with a welcome message part which has an introduction to the {output_type}.
        Finish with a closing message part which has a conclusion to the {output_type}.
        Break it down into multiple parts which will be read with pauses between them during {output_type}.
        Each part should only contain a thought by you on the topic and finish with a single action request, 
        but may contain multiple pieces of background information or motivations as well.
        The action request should be based on the meditation technique '{self.technique}'.
        THE ACTION SHOULD BE THE LAST SENTENCE OF THE PART! Do not relate the action to the topic, just to the technique.
        Some may only be reminders to continue the focus on what was instructed a previous part.
        Do NOT put two actions in one subsection. Do NOT ask the listener to both take an action and consider or think about something!
        """
        if self.limit_parts > 0:
            if self.limit_parts < 3:
                self.limit_parts = 3
                print("WARNING: limit_parts must be at least 3 to include introduction and conclusion parts. Setting to 3.")
            self.prompt += f"""
            The entire JSON {output_type} should contain NO MORE THAN {self.limit_parts} meditation_part  keys.\n"""
        if not self.expand_on_section:
            self.prompt += f"""
            Each meditation_part value should be no more than {self.num_sentences} sentences long.\n"""
        self.expansion_size = expansion_size
        if self.expand_on_section:
            self.prompt += f"""Start each part's text with a {self.expansion_size} sentence 
                                    details relating to that part. 
                                    Insure the details
                                   demonstrate your in-depth knowledge of the section (and topic) and help 
                                   the listener. Do not ask the listener to take actions or consider thoughts in this. \n """
        if not affirmations_only:
            self.prompt += f"""
            Whatever the topic of the meditation, embed it within the following meditation technique:\n
            {self.technique}. Do not talk about 'think about' or 'consider'.\n
            """
        json_subprompt = '''
        Return the responses in JSON format like:\n
        [
            {"meditation_part_1": "The first part of the meditation text."},
            {"meditation_part_2": "The second part of the meditation text."},
            {"meditation_part_3": "The third part of the meditation text."}
            etc.
        ]
        Add contextual information for each sentence, such as [careful] or [serious] or [happy] to help humanise the speech.
        '''
        # repeat the below for emphasis
        if self.limit_parts > 0:
            self.prompt += f"""
            URGENT: The entire JSON {output_type} should contain NO MORE THAN {self.limit_parts} meditation_part JSON keys.\n""".upper()
        if not self.expand_on_section:
            self.prompt += f"""
            URGENT: Each meditation_part value should be no more than {self.num_sentences} sentences long.\n""".upper()
        if affirmations_only:
            json_subprompt = json_subprompt.replace("meditation", "affirmation")
        self.prompt += json_subprompt
        if self.in_spanish:
            self.prompt += "\n Responde en espanol."
        self.opacity = image_background_opacity
        self.banner_at_bottom = banner_at_bottom
        self.banner_height_ratio = banner_height_ratio
        self.max_banner_words = max_banner_words
        races = ["white", "black", "asian", "hispanic", "pakistani", "iranian", "pacific islander"]
        if not self.beautiful_lady:
            self.image_prompt = f"""
                Image only. 
                Generate a beautiful image based on the {output_type} topic '{self.topic}'.
                It should be relaxing and be photorealistic. 
                """
            if random.random() < 0.5:
                self.image_prompt += " It should include a beautiful person."
            if random.random() < 0.5:
                self.image_prompt += " It should be in outer space."
            elif random.random() < 0.5:
                self.image_prompt += " It should be under or on the ocean."
            if random.random() < 0.5:
                self.image_prompt += " It should be in the style of a random famous painter."
            else:
                self.image_prompt += " It should be in the style of a random famous artist."
            self.image_prompt += " REMEMBER: image only. PHOTOREALISTIC."
        else:
            self.image_prompt = f"""Generate a beautiful image of a beautiful lady meditating.
                It should be relaxing and be photorealistic. Remember: image only.
                They should be {random.choice(races)}."""

    def send_prompt(self, prompt: str, use_json: bool = False) -> str:
        if prompt.strip() == "":
            raise ValueError("Error: send_prompt - Empty prompt.")
        content = {
            "type": "text",
            "text": prompt,
        }
        if use_json:
            content["response_format"] = "json"
            if "json" not in prompt and "JSON" not in prompt:
                raise ValueError("Error: send_prompt - JSON must be mentioned in a JSON prompt.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        content
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    # first part of pipeline
    def generate_meditation_texts(self) -> List[str]:
        """
        Generates meditation text divided into subsections based on the given topic
        and stores in working directory.

        :return: List of text subsections for the meditation.
        """
        # check self. working_directory exists
        if not os.path.exists(self.working_directory):
            msg = f"Error: generate_meditations_texts - working_directory does not exist: {self.working_directory}"
            raise FileNotFoundError(msg)
        # check self.topic_based_filename exists
        if not self.topic_based_filename:
            raise ValueError("Error: topic_based_filename is not set.")
        # check prompt is set
        if not self.prompt:
            msg = "Error: generate_meditations_texts - Prompt is not set."
            raise ValueError(msg)
        # check is contains JSON or json
        if "json" not in self.prompt and "JSON" not in self.prompt:
            msg = "Error: generate_meditations_texts - JSON must be mentioned in a JSON prompt."
            raise ValueError(msg)
        # Generate the meditation text using OpenAI's LLM
        full_text = self.send_prompt(self.prompt, use_json=True)
        full_text = full_text.replace("```json", "").replace("```", "")
        try:
            full_json = json.loads(full_text)
            # Split the text into subsections
            self.subsections = [value for d in full_json for key, value in d.items()]
        except Exception as e:
            try:
                full_json = eval(full_text)
                self.subsections = [value for d in full_json for key, value in d.items()]
            except Exception as e:
                # write it to a file for debugging
                filename = os.path.join(self.working_directory,
                                        f"{self.topic_based_filename}_raw_meditation_text_debug.json")
                with open(filename, "w") as f:
                    f.write(full_text)
                msg = f"Error: generate_meditations_texts -  Could not parse the response from the API. Check the file {filename} for more information."
                raise ValueError(msg) from e

        # write to file
        filename = os.path.join(self.working_directory, f"{self.topic_based_filename}_meditation_text.json")
        with open(filename, "w") as f:
            json.dump(full_json, f, indent=4)
        return self.subsections

    # used by methods below
    def synthesize_speech(self, text: str, filename: str, voice: str = "") -> None:
        """
        Synthesizes speech from the given text and saves it as an MP3 file.

        :param text: Text to be converted to speech.
        :param filename: Name of the output MP3 file.
        :param voice: Voice to be used for the speech synthesis.
        """
        # check text not empty
        if text.strip() == "":
            raise ValueError("Error: synthesize_speech - Empty text.")
        # check filename not empty
        if filename.strip() == "":
            raise ValueError("Error: synthesize_speech - Empty filename.")
        # check filename ends with .mp3
        if not filename.endswith(".mp3"):
            raise ValueError("Error: synthesize_speech - Filename must end with .mp3.")
        if not voice:
            voice = self.voice
        try:
            # https://stackoverflow.com/questions/77952454/method-in-python-stream-to-file-not-working
            with self.client.audio.speech.with_streaming_response.create(
                model=self.engine,
                voice=voice,
                input=text
            ) as response:
                response.stream_to_file(filename)

        except Exception as e:
            msg = f"Error: synthesize_speech - OpenAI API call failed for the text: {text}"
            raise RuntimeError(msg) from e

    # second part of the pipeline
    def create_meditation_text_audio_files(self) -> List[str]:
        """
        Creates MP3 files for each subsection.

        :return: List of paths to the generated MP3 files.
        """
        self.subsection_audio_files = []
        # check self.subsections exists
        if not self.subsections:
            msg = "Error: create_meditation_text_audio_files - Subsections are not set."
            raise ValueError(msg)
        # check self.working_directory exists
        if not os.path.exists(self.working_directory):
            msg = f"Error: create_meditation_text_audio_files - working_directory does not exist: {self.working_directory}"
            raise FileNotFoundError(msg)
        for i, subsection in enumerate(self.subsections):
            filename = os.path.join(self.working_directory, f"meditation_part_{i + 1}.mp3")
            print(f"Generating audio for part {i + 1} of {len(self.subsections)}...")
            if self.two_voices:
                # check self.voice_even exists
                if not self.voice_even:
                    msg = "Error: create_meditation_text_audio_files - voice_even is not set."
                    raise ValueError(msg)
                # check self.voice_odd exists
                if not self.voice_odd:
                    msg = "Error: create_meditation_text_audio_files - voice_odd is not set."
                    raise ValueError(msg)
                if i % 2 == 0:
                    self.synthesize_speech(subsection, filename, voice=self.voice_even)
                else:
                    self.synthesize_speech(subsection, filename, voice=self.voice_odd)
            else:
                self.synthesize_speech(subsection, filename)
            self.subsection_audio_files.append(filename)
        if self.num_loops:
            subsection_audio_files_orig = self.subsection_audio_files.copy()
            # now copy the files num_loops times but update the filenames
            # so the index part fo the filename is unique and ordered
            # and update the self.subsection_audio_files
            base_index = len(subsection_audio_files_orig)
            for i in range(1, self.num_loops):  # num_loops 1 means no repeat
                for j, file in enumerate(subsection_audio_files_orig):
                    new_file = file.replace(f"meditation_part_{j + 1}", f"meditation_part_{base_index + j + 1}")
                    shutil.copy(file, new_file)
                    self.subsection_audio_files.append(new_file)
                base_index += len(subsection_audio_files_orig)
        return self.subsection_audio_files

    # third part of the pipeline
    def merge_meditation_audio(self) -> str:
        """
        merges them into a single MP3 file.

        :return: Path to the merged MP3 file.
        """
        # check self.working_directory exists
        if not os.path.exists(self.working_directory):
            msg = f"Error: merge_meditation_audio - working_directory does not exist: {self.working_directory}"
            raise FileNotFoundError(msg)
        if not self.subsection_audio_files:
            # get all files in working dir that start with meditation_part
            self.subsection_audio_files = [os.path.join(self.working_directory, f) for f in
                                           os.listdir(self.working_directory) if f.startswith("meditation_part")]
            # if empty list then return error
            if not self.subsection_audio_files:
                msg = "Error: merge_meditation_audio - No files in list and none found to merge."
                raise ValueError(msg)
            # ensure sorted in increasing order
            self.subsection_audio_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.merger.mp3_files = self.subsection_audio_files
        # merger.spread_out_all_files()  # puts extra silence between phrases
        merged_file = self.merger.merge()
        # check self.topic_based_filename not empty
        if self.topic_based_filename.strip() == "":
            msg = "Error: merge_meditation_audio - Empty topic_based_filename."
            raise ValueError(msg)
        # write to file
        filename = os.path.join(self.working_directory, f"{self.topic_based_filename}_meditation_text_merged.mp3")
        # rename merged file to filename
        shutil.move(merged_file, filename)
        return filename

    # fourth part of the pipeline
    # use pedalboard to add reverberation, echo, or other audio effects to the audio file
    def add_audio_fx(self) -> str:
        """
        Adds audio effects to the given audio file using Pedalboard.

        :return: Path to the output audio file with effects applied.
        """
        # check self.working_directory exists
        if not os.path.exists(self.working_directory):
            msg = f"Error: add_audio_fx - working_directory does not exist: {self.working_directory}"
            raise FileNotFoundError(msg)
        # check self.topic_based_filename not empty
        if self.topic_based_filename.strip() == "":
            msg = "Error: add_audio_fx - Empty topic_based_filename."
            raise ValueError(msg)
        # check a file ending _meditation_text_merged.mp3 exists in working dir
        if not os.path.exists(
                os.path.join(self.working_directory, f"{self.topic_based_filename}_meditation_text_merged.mp3")):
            msg = f"Error: add_audio_fx - File {self.topic_based_filename}_meditation_text_merged.mp3 does not exist in working directory."
            raise FileNotFoundError(msg)
        filename = os.path.join(self.working_directory,
                                f"{self.topic_based_filename}_meditation_text_merged.mp3")
        # Load the MP3 file
        with AudioFile(filename) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate
        # Create a Pedalboard with desired effects
        board = Pedalboard(self.pedalboard_fx_list)
        # Apply the effects
        effected = board(audio, sample_rate)
        filename = os.path.join(self.working_directory,
                                f"{self.topic_based_filename}_meditation_text_merged_fx.mp3")
        # Write the effected audio to a new file
        with AudioFile(filename, 'w', sample_rate, effected.shape[0]) as f:
            f.write(effected)
        return filename

    # fifth part of the pipeline
    # generate binaural beats using MP3Mixer and merge with the merged_file
    def mix_meditation_audio(self) -> str:
        """
        Mixes the meditation audio with binaural beats.

        :param merged_file: Path to the merged MP3 file.
        :return: Path to the mixed MP3 file.
        """
        # check self.working_directory exists
        if not os.path.exists(self.working_directory):
            msg = f"Error: mix_meditation_audio - working_directory does not exist: {self.working_directory}"
            raise FileNotFoundError(msg)
        # check self.topic_based_filename not empty
        if self.topic_based_filename.strip() == "":
            msg = "Error: mix_meditation_audio - Empty topic_based_filename."
            raise ValueError(msg)
        # check a file ending _meditation_text_merged_fx.mp3 exists in working dir
        if not os.path.exists(
                os.path.join(self.working_directory, f"{self.topic_based_filename}_meditation_text_merged_fx.mp3")):
            msg = f"Error: add_audio_fx - File {self.topic_based_filename}_meditation_text_merged_fx.mp3 does not exist in working directory."
            raise FileNotFoundError(msg)
        merged_file = os.path.join(self.working_directory,
                                   f"{self.topic_based_filename}_meditation_text_merged_fx.mp3")
        self.mixer.mp3_file = merged_file
        mixed_file = self.mixer.mix_audio()
        base_filename = f"{self.topic_based_filename}_meditation_text_merged_fx_mixed.mp3"
        filename = os.path.join(self.working_directory, base_filename)
        # rename mixed file to filename
        os.rename(mixed_file, filename)
        # copy to base_filename as well
        # so that video and mp3 are in the top level directory in case user wants to upload
        # the mp3 to spotify or something.
        shutil.copy(filename, base_filename)
        return filename

    def find_optimal_font_size_and_wrap(self, text, max_width, max_height, font_path):
        font_size = 1
        lines = []
        while True:
            # if raises error then assume font_path is not valid
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError:
                msg = f"Error: find_optimal_font_size_and_wrap - Invalid font path: {font_path}"
                msg += "This is probably because the OS front path has been incorrectly detected by this package."
                raise OSError(msg)
            # Estimate the number of characters per line
            avg_char_width = font.getbbox('X')[2]  # Use 'X' to get the width of a capital letter
            max_chars_per_line = max(1, int(max_width / avg_char_width))
            # Wrap text
            wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
            # Calculate total height
            line_height = font.getbbox('X')[3]  # Use 'X' to get the height of a capital letter
            total_height = line_height * len(wrapped_lines)
            if total_height > max_height * 0.8:  # Reduce to 80% of max height to leave more space at the bottom
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
                break
            font_size += 1
            lines = wrapped_lines
        return font, lines

    def add_banner_with_text(self, image_path: str, banner_text: str,
                                     output_path: str):
        """
        Add a banner with text to an image.

        :param image_path: Path to the input image.
        :param banner_text: Text to display on the banner.
        :param output_path: Path to save the output image.
        """
        # check for valid image path
        if not os.path.exists(image_path):
            msg = f"Error: add_banner_with_text - Image file does not exist: {image_path}"
            raise FileNotFoundError(msg)

        # if it fails to load image now, assume that is not a valid image file
        try:
            image = Image.open(image_path)
        except Exception as e:
            msg = f"Error: add_banner_with_text - Failed to load image file: {image_path}"
            msg += f"\nUnderlying error: {str(e)}"
            raise ValueError(msg)
        width, height = image.size
        # check width and height large enough to do any of the below
        if width < 100 or height < 100:
            msg = f"Error: add_banner_with_text - Image dimensions too small: {width} x {height}"
            raise ValueError(msg)
        # check for valid banner text
        if banner_text == "":
            msg = "Error: add_banner_with_text - banner text is empty."
            raise ValueError(msg)
        # Convert text to uppercase for the banner
        text = banner_text.upper()

        # Calculate the gap size (2.5% of the image height and width)
        gap_size = int(height * 0.025)

        # Calculate the banner height (e.g. 25% of the image height) minus gaps
        banner_height = int(height * self.banner_height_ratio) - 2 * gap_size

        # check self.banner_height_ratio doesn't make banner height <= 0
        if banner_height <= 0:
            msg = f"Error: add_banner_with_text - Banner height too small: {banner_height}"
            msg += f"\nCheck self.banner_height_ratio value: {self.banner_height_ratio}"
            raise ValueError(msg)
        # Calculate the banner width minus gaps
        banner_width = width - 2 * gap_size
        # check banner width > 0
        if banner_width <= 0:
            msg = f"Error: add_banner_with_text - Banner width too small: {banner_width}"
            msg += f"\nCheck image width: {width}"
            raise ValueError(msg)
        # Create a drawing context
        draw = ImageDraw.Draw(image, "RGBA")
        background_colour = tuple(BLUE + [int(self.opacity * 255)])
        # Define the banner area (black rectangle with gaps around it)
        if self.banner_at_bottom:
            banner_area = [(gap_size, height - banner_height - gap_size), (gap_size + banner_width, height - gap_size)]
        else:
            banner_area = [(gap_size, gap_size), (gap_size + banner_width, gap_size + banner_height)]
        # check banner area elements all > 0
        if any([x <= 0 for point in banner_area for x in point]):
            msg = f"Error: add_banner_with_text - Invalid banner area: {banner_area}"
            msg += f"\nCheck image dimensions: {width} x {height} and banner height ratio: {self.banner_height_ratio}"
            raise ValueError(msg)
        draw.rectangle(banner_area, fill=background_colour)
        # check if macOS, Windows or Linux
        if platform.system() == "Darwin":
            font_path = "/Library/Fonts/Arial.ttf"
        elif platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/Arial.ttf"
        elif platform.system() == "Linux":
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        # if not macOS, Windows or Linux, use default font path raise error
        else:
            msg = "Error: add_banner_with_text - Unsupported OS (only supports Darwin, Windows, Linux). Or this package is detecting your OS incorrectly."
            raise OSError(msg)
        #font_path = "/Library/Fonts/Arial.ttf"  # macOS example
        # font_path = "C:/Windows/Fonts/Arial.ttf"  # Windows example
        # font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux example
        # Find optimal font size and wrap text
        font, lines = self.find_optimal_font_size_and_wrap(text, banner_width, banner_height, font_path)
        # Calculate total text height
        line_height = font.getbbox('X')[3]
        total_text_height = line_height * len(lines)
        if self.banner_at_bottom:
            y_position = height - banner_height - gap_size + (banner_height - total_text_height) * 0.4
        else:
            # Position the text vertically centered in the banner, with extra gap at the bottom
            y_position = gap_size + (banner_height - total_text_height) * 0.4  # Adjust this factor to move text up or down
        # check y_position >= 0
        if y_position < 0:
            msg = f"Error: add_banner_with_text - Invalid y_position: {y_position}"
            msg += f"\nCheck banner_height_ratio: {self.banner_height_ratio} and image height: {height}"
            raise ValueError(msg)
        # Add each line of text to the banner
        for line in lines:
            text_width = font.getbbox(line)[2]
            x_position = gap_size + (banner_width - text_width) / 2
            draw.text((x_position, y_position), line, fill=YELLOW,
                      font=font)
            y_position += line_height
        # check for valid output path
        if not os.path.exists(os.path.dirname(output_path)):
            msg = f"Error: add_banner_with_text - Output directory does not exist: {output_path}"
            raise FileNotFoundError(msg)
        # Save the image
        image.save(output_path)
        return output_path

    # sixth part of the pipeline
    # generate an image based on the meditation topic using dall-e
    # the image will be used as a background for the video
    # it should be relaxing and imply meditation
    def generate_meditation_image(self) -> str:
        """
        Generates an image based on the meditation topic using DALL-E.

        :return: Path to the generated image.
        """
        # check self.image_model not empty
        if self.image_model.strip() == "":
            msg = "Error: generate_meditation_image - Empty image_model."
            raise ValueError(msg)
        # check self.image_prompt not empty
        if self.image_prompt.strip() == "":
            msg = "Error: generate_meditation_image - Empty image_prompt."
            raise ValueError(msg)
        # check self.image_quality not empty
        if self.image_quality.strip() == "":
            msg = "Error: generate_meditation_image - Empty image_quality."
            raise ValueError(msg)
        # check self.working_directory exists
        if not os.path.exists(self.working_directory):
            msg = f"Error: generate_meditation_image - working_directory does not exist: {self.working_directory}"
            raise FileNotFoundError(msg)
        # check self.topic_based_filename not empty
        if self.topic_based_filename.strip() == "":
            msg = "Error: generate_meditation_image - Empty topic_based_filename."
            raise ValueError(msg)
        # 1920x1080 is the resolution of the video
        try:
            response = self.client.images.generate(
                model=self.image_model,
                prompt=self.image_prompt,
                size="1792x1024",
                quality=self.image_quality,
                n=1,
            )
        except Exception as e:
            msg = f"Error: generate_meditation_image - OpenAI API call failed: {str(e)}"
            raise RuntimeError(msg) from e
        image_url = response.data[0].url
        # use requests to get image
        response = requests.get(image_url)
        # save the image
        image_path = os.path.join(self.working_directory,
                                  f"{self.topic_based_filename}_meditation_image_no_banner.jpg")
        with open(image_path, "wb") as f:
            f.write(response.content)
        # add banner

        if len(self.topic.split()) > self.max_banner_words:
            prompt = f"Give me a less than {self.max_banner_words} word summary of the following topic: {self.topic}"
            response = self.send_prompt(prompt)
        else:
            response = self.topic
        output_image_path = os.path.join(self.working_directory,
                                  f"{self.topic_based_filename}_meditation_image.jpg")
        image_path = self.add_banner_with_text(image_path, response, output_image_path)
        return image_path

    # generate an mp4 which consists of an audio file combined with an image file
    def create_meditation_video(self) -> str:
        """
        Creates a video from the meditation audio and image.

        :return: Path to the output video file.
        """
        # check self.working_directory exists
        if not os.path.exists(self.working_directory):
            msg = f"Error: create_meditation_video - working_directory does not exist: {self.working_directory}"
            raise FileNotFoundError(msg)
        # check self.topic_based_filename not empty
        if self.topic_based_filename.strip() == "":
            msg = "Error: create_meditation_video - Empty topic_based_filename."
            raise ValueError(msg)
        # check image file exists
        if not os.path.exists(os.path.join(self.working_directory,
                                            f"{self.topic_based_filename}_meditation_image.jpg")):
                msg = f"Error: create_meditation_video - Image file {self.topic_based_filename}_meditation_image.jpg does not exist."
                raise FileNotFoundError(msg)
        #Load the image and create a clip
        image_clip = ImageClip(os.path.join(self.working_directory,
                                            f"{self.topic_based_filename}_meditation_image.jpg"))
        # check audio file exists
        if not os.path.exists(os.path.join(self.working_directory,
                                            f"{self.topic_based_filename}_meditation_text_merged_fx_mixed.mp3")):
                msg = f"Error: create_meditation_video - Audio file {self.topic_based_filename}_meditation_text_merged_fx_mixed.mp3 does not exist."
                raise FileNotFoundError(msg)
        # Load the audio file
        audio_filename = os.path.join(self.working_directory,
                                      f"{self.topic_based_filename}_meditation_text_merged_fx_mixed.mp3")
        audio_clip = AudioFileClip(audio_filename)
        # Set the duration of the image clip to match the audio duration
        image_clip = image_clip.set_duration(audio_clip.duration)
        # Set the audio of the clip
        video_clip = image_clip.set_audio(audio_clip)
        # Set the output path
        output_path = f"{self.topic_based_filename}_meditation_video.mp4"
        # Write the result to a file
        video_clip.write_videofile(output_path, fps=24, threads=32)
        # copy audio file to output directory
        return output_path

    def run_meditation_pipeline(self, content: str = "") -> tuple[list[str], str]:
        """
        Generates a full meditation experience including text, audio, and video.
        If content string is non-empty, it will be used as the meditation text
        instead of generating one.

        :param content: The meditation text to use. If empty one is generated.
        :return: list of the meditation subsection texts and the output filename
        """
        # note the below code is deliberately done in a repetitive way
        # to make it easier to understand
        # (It could have been implemented using a dictionary of functions and a loop)
        if content == "":
            print("Creating GENERATIVE MEDITATION")
            if self.pipeline.get("texts"):
                print("Generating meditation texts...")
                self.generate_meditation_texts()
            else:
                print("Skipping meditation texts...")
        else:
            print("Creating FREE TEXT MEDITATION")
            # content is a list of \n\n divided free text sections. Split into a list
            free_text = content.split("\n\n")
            # strip() all strings in the list
            free_text = [ft.strip() for ft in free_text]
            # remove any empty strings
            free_text = [ft for ft in free_text if ft]
            # check not empty
            if not free_text:
                msg = "Error: run_meditation_pipeline - Empty free_text list."
                raise ValueError(msg)
            self.subsections = free_text  # this is what would've been generated
        # generate the audio
        if self.pipeline.get("audio_files"):
            print("Generating meditation audio sub-files...")
            self.create_meditation_text_audio_files()
        else:
            print("Skipping generating meditation audio sub-files...")
        if self.pipeline.get("combine_audio_files"):
            # merge the audio
            print("Combining audio files with silences...")
            self.merge_meditation_audio()
        else:
            print("Skipping combining audio files...")
        if self.pipeline.get("audio_fx"):
            print("Adding audio effects...")
            self.add_audio_fx()
        else:
            print("Skipping audio effects...")
        if self.pipeline.get("background_audio"):
            # mix the audio
            print("Adding background sounds/music to spoken audio...")
            self.mix_meditation_audio()
        else:
            print("Skipping adding background sounds/music to spoken audio...")
        if self.pipeline.get("image"):
            # generate the image
            print("Generating meditation image for video...")
            self.generate_meditation_image()
        else:
            print("Skipping image generation image for video...")
        if self.pipeline.get("video"):
            # generate the video
            print("Creating meditation video...")
            filename = self.create_meditation_video()
        else:
            print("Skipping video generation...")
            filename = "Video generation skipped by user choice."
        if self.pipeline.get("keywords"):
            kw, kw_str = self.generate_keywords()
            print(f"Suggested Keywords: \n{kw_str}")
            if self.in_spanish:
                # check self.topic is not empty
                if self.topic.strip() == "":
                    msg = "Error: run_meditation_pipeline - Empty topic."
                    raise ValueError(msg)
                print(f"Topic in Spanish: \n{self.translate_text(self.topic)}")
        else:
            print("Skipping keyword generation...")
            kw, kw_str = [], ""
        return self.subsections, filename

    def generate_keywords(self, num_keywords: int = 30):
        """
        Generate keywords for the topic.

        :param num_keywords: The number of keywords to generate.

        """
        # check num_keywords is a positive integer
        if not isinstance(num_keywords, int) or num_keywords <= 0:
            msg = "Error: generate_keywords - num_keywords must be a positive integer."
            raise ValueError(msg)
        # Get the top 30 keywords for the topic
        keywords_prompt = f"""
        Act as a social media manager who is an expert in Youtube.
        You are tasked with generating keywords for a guided meditation and affirmation video.
        Generate at least {num_keywords} keywords of one word each for such a video with the topic '{self.topic}'.
        In addition generate 10 single word keywords about mindfulness, meditation, and relaxation.
        Only respond with the keywords and give NO prefix or suffix to your response.
        Return them as a comma-separated list as follows:
        keyword1, keyword2, keyword3, ...
        """
        print(f"Generating keywords for the topic '{self.topic}'...")
        keywords_str = self.send_prompt(keywords_prompt)
        try:
            keywords = [k.lower().strip() for k in keywords_str.split(",")]
            if self.in_spanish:
                keywords = [self.translate_text(k) for k in keywords]
                keywords_str = ", ".join(keywords)
        except Exception as e:
            msg = f"Error parsing OpenAI API response for keywords. See file 'keywords_raw_response.txt'.\n{e}"
            # write raw response to file
            with open("keywords_raw_response.txt", "w") as f:
                f.write(keywords_str)
            raise ValueError(msg)
        return keywords, keywords_str

    def translate_text(self, text: str, target_language: str = "Spanish") -> str:
        """
        Translate text to a target language.

        :param text: The text to translate.
        :param target_language: The target language to translate to.
        :return: The translated text.
        """
        # check text is a non-empty string
        if not isinstance(text, str) or text.strip() == "":
            msg = "Error: translate_text - text must be a non-empty string."
            raise ValueError(msg)
        # check target_language is a non-empty string
        if not isinstance(target_language, str) or target_language.strip() == "":
            msg = "Error: translate_text - target_language must be a non-empty string."
            raise ValueError(msg)
        # Translate the text to the target language
        translation_prompt = f"""
        Act as a professional translator who is an expert in translating text to different languages.
        You are tasked with translating the below TEXT to {target_language}:
        <TEXT>'{text}'</TEXT>
        Respond only with the translated text, with no prefix or suffix to your response.
        """
        print(f"Translating text to {target_language}...")
        translated_text = self.send_prompt(translation_prompt)
        return translated_text

    def delete_meditation_workspace(self):
        """
        Delete the meditation workspace - i.e. the directory and its contents.
        """
        if os.path.exists(self.working_directory):
            # remove files in dir first
            for file in os.listdir(self.working_directory):
                file_path = os.path.join(self.working_directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting file: {e}")
            shutil.rmtree(self.working_directory)
            print(f"Deleted meditation workspace at '{self.working_directory}'.")
