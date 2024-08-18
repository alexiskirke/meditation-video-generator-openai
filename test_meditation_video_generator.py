import base64
import json
import os
import shutil
import sys

import numpy as np
import openai
from moviepy.editor import ImageClip
from pydub import AudioSegment

from meditation_video_generator import MeditationVideoGenerator
from keys import KEY
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from keys import KEY

class DotDict(dict):
    """A dictionary that supports dot notation and nested dictionaries.
    This code ensures that all nested dictionaries within the original
    dictionary (including those inside lists) are converted to DotDict
    objects, allowing for recursive dot notation access to nested structures.
    """
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [DotDict(item) if isinstance(item, dict) else item for item in value]
            self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def test_meditation_video_generator_live():

    # 1. test size of tiny generated subsections
    mvg = MeditationVideoGenerator(api_key=KEY, force_working_dir_overwrite=True,
                                   length=1, num_sentences=2, expand_on_section=False,
                                   limit_parts=1)
    # uses samples since gpt is non-deterministic
    subsections_samples = []
    num_samples = 3
    for samples in range(num_samples):
        subsections = mvg.generate_meditation_texts()
        subsections_samples.append(subsections)
    # calculate the mean length of the subsections
    mean_length = np.mean([len(s) for s in subsections_samples])
    # check there are approximately length+2 subsections
    assert 3 <= mean_length < 5
    # raise a warning if there are not limit_part + 2 subsections, but it's not fatal
    if not sum([len(s) for s in subsections_samples]) != 3*num_samples:
        print("Warning: There are not limit_part + 2 subsections. But there are ", [len(s) for s in subsections_samples])
    # check the average number of sentences across all subsections is approximately 2
    # join subsections_samples into one list of nums
    subsections = [item for sublist in subsections_samples for item in sublist]
    assert 2 <= np.mean([len(s.split(".")) for s in subsections]) < 4

    # 2. check that extended versions are longer.
    prev_subsections = subsections
    mvg = MeditationVideoGenerator(api_key=KEY, force_working_dir_overwrite=True,
                                      length=1, num_sentences=1, expand_on_section=True, limit_parts=1)
    subsections = mvg.generate_meditation_texts()
    # check that mean number of sentences is greater than before
    assert np.mean([len(s.split(".")) for s in subsections]) > \
           np.mean([len(s.split(".")) for s in prev_subsections])


    # 3. test size of larger generated subsections
    mvg = MeditationVideoGenerator(api_key=KEY, force_working_dir_overwrite=True,
                                   length=2, num_sentences=3, expand_on_section=False, limit_parts=9)
    # uses samples since gpt is non-deterministic
    subsections_samples = []
    num_samples = 3
    for samples in range(num_samples):
        subsections = mvg.generate_meditation_texts()
        subsections_samples.append(subsections)
    # calculate the mean length of the subsections
    mean_length = np.mean([len(s) for s in subsections_samples])
    # check there are approximately length+2 subsections
    assert 7 <= mean_length < 11
    # raise a warning if there are not limit_part + 2 subsections, but it's not fatal
    if not sum([len(s) for s in subsections_samples]) != 3 * num_samples:
        print("Warning: There are not limit_part + 2 subsections. But there are ",
              [len(s) for s in subsections_samples])
    # check the average number of sentences across all subsections is approximately 3
    # join subsections_samples into one list of nums
    subsections = [item for sublist in subsections_samples for item in sublist]
    assert 2 <= np.mean([len(s.split(".")) for s in subsections]) <= 4


    # 4. check that doing an affirmation with free text works
    mvg = MeditationVideoGenerator(api_key=KEY, force_working_dir_overwrite=True,
                                   affirmations_only=True)
    mvg.pipeline = {
        "texts": True
    }
    subsections, _ = mvg.run_meditation_pipeline(content="I am a good person.\n\nI am a kind person.")
    assert len(subsections) == 2
    # also check that the first line is the first sentence of the first subsection
    assert subsections[0] == "I am a good person."
    assert subsections[1] == "I am a kind person."

    # 5. check that num_loops works
    mvg = MeditationVideoGenerator(api_key=KEY, force_working_dir_overwrite=True,
                                   num_loops=2)
    mvg.pipeline = {
            "texts": True,
            "audio_files": True
        }
    mvg.run_meditation_pipeline(content="I am a good person.\n\nI am a kind person.")
    assert len(mvg.subsection_audio_files) == 2 * 2



def test_meditation_video_generator_edge_cases():
    mvg = MeditationVideoGenerator(api_key=KEY, force_working_dir_overwrite=True)
    # create 3 silent audio files using audiosegment in working dir
    for i in range(1,4):
        AudioSegment.silent(duration=1000).export(f"fakefile{i}.mp3", format="mp3")

    # 1. Error: send_prompt - Empty prompt.
    with pytest.raises(ValueError, match="Error: send_prompt - Empty prompt."):
        mvg.send_prompt("")

    # 2. Error: send_prompt - JSON must be mentioned in a JSON prompt.
    with pytest.raises(ValueError, match="Error: send_prompt - JSON must be mentioned in a JSON prompt."):
        mvg.send_prompt("Hello", use_json=True)

    # 3. Error: topic_based_filename is not set.
    mvg.topic_based_filename = ""
    with pytest.raises(ValueError, match="Error: topic_based_filename is not set."):
        mvg.generate_meditation_texts()

    # 4. Error: generate_meditations_texts - Prompt is not set.
    mvg.topic_based_filename = "test"
    mvg.prompt = ""
    with pytest.raises(ValueError, match="Error: generate_meditations_texts - Prompt is not set."):
        mvg.generate_meditation_texts()

    mock_return_value_content = """[
            {"meditation_part_1": "The first part of the meditation text."},
            {"meditation_part_2": "The second part of the meditation text."},
            {"meditation_part_3": "The third part of the meditation text."}
        ]
    """
    mock_return_value_content = mock_return_value_content.replace("'", '"')
    mock_return_value = {
        "choices": [
            {
                "message": {
                    "content": mock_return_value_content
                }
            }
        ]
    }

    mock_return_value = DotDict(mock_return_value)
    openai.api_key = "mock_key"

    # 5. Error: generate_meditations_texts - JSON must be mentioned in a JSON prompt.
    mvg.prompt = "Hello"
    with pytest.raises(ValueError, match="Error: generate_meditations_texts - JSON must be mentioned in a JSON prompt."):
        mvg.generate_meditation_texts()

    # 7 Error: generate_meditations_texts - working_directory does not exist
    mvg = MeditationVideoGenerator(num_sentences=1, limit_parts=2, length=2, api_key=KEY, topic="Mindfulness", force_working_dir_overwrite=True)
    with patch.object(mvg.client.chat.completions, 'create', return_value=mock_return_value):
        mvg.working_directory = "i_do_not_exist"
        with pytest.raises(FileNotFoundError, match="Error: generate_meditations_texts - working_directory does not exist"):
            mvg.generate_meditation_texts()

    # 8 Generate a meditation text with mock value has 3 subsections
    mvg = MeditationVideoGenerator(length=2, api_key="mock_key", force_working_dir_overwrite=True)
    with patch.object(mvg.client.chat.completions, 'create', return_value=mock_return_value):
        result = mvg.generate_meditation_texts()
        assert len(result) == 3

    # 9. Error: synthesize_speech - Empty text.
    with pytest.raises(ValueError, match="Error: synthesize_speech - Empty text."):
        mvg.synthesize_speech(text="", filename="Mindfulness_1.mp3")

    # 10. Error: synthesize_speech - Empty filename.
    with pytest.raises(ValueError, match="Error: synthesize_speech - Empty filename."):
        mvg.synthesize_speech(text="Hello", filename="")

    # 11. Error: synthesize_speech - Filename must end with .mp3.
    with pytest.raises(ValueError, match="Error: synthesize_speech - Filename must end with .mp3."):
        mvg.synthesize_speech(text="Hello", filename="Mindfulness_1")

    # 12. Error: create_meditation_text_audio_files - Subsections are not set.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(ValueError, match="Error: create_meditation_text_audio_files - Subsections are not set."):
        mvg.create_meditation_text_audio_files()

    # 13. Error: create_meditation_text_audio_files - working_directory does not exist:
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.working_directory = "i_do_not_exist"
    mvg.subsections = ["The first part of the meditation text.", "The second part of the meditation text.", "The third part of the meditation text."]
    with pytest.raises(FileNotFoundError, match="Error: create_meditation_text_audio_files - working_directory does not exist"):
        mvg.create_meditation_text_audio_files()

    # 14. Error: create_meditation_text_audio_files - voice_even is not set.
    with patch.object(mvg.client.audio.speech, 'create'):
        mvg = MeditationVideoGenerator(force_working_dir_overwrite=True, two_voices=True)
        mvg.voice_even = ""
        mvg.subsections = ["The first part of the meditation text.", "The second part of the meditation text.", "The third part of the meditation text."]
        with pytest.raises(ValueError, match="Error: create_meditation_text_audio_files - voice_even is not set."):
            mvg.create_meditation_text_audio_files()

    # 15. Error: create_meditation_text_audio_files - voice_odd is not set.
    with patch.object(mvg.client.audio.speech, 'create'):
        mvg = MeditationVideoGenerator(force_working_dir_overwrite=True, two_voices=True)
        mvg.voice_odd = ""
        mvg.subsections = ["The first part of the meditation text.", "The second part of the meditation text.", "The third part of the meditation text."]
        with pytest.raises(ValueError, match="Error: create_meditation_text_audio_files - voice_odd is not set."):
            mvg.create_meditation_text_audio_files()

    # test merge_mediation_audio
    # 16. Error: merge_meditation_audio - working_directory does not exist:
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.working_directory = "i_do_not_exist"
    with pytest.raises(FileNotFoundError, match="Error: merge_meditation_audio - working_directory does not exist"):
        mvg.merge_meditation_audio()

    # 16.5: Error: merge_meditation_audio - No files in list and none found to merge.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(ValueError, match="Error: merge_meditation_audio - No files in list and none found to merge."):
        mvg.merge_meditation_audio()

    # 17. Error: merge_meditation_audio - Empty topic_based_filename.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.topic_based_filename = ""
    mvg.subsection_audio_files = ["fakefile1.mp3", "fakefile2.mp3", "fakefile3.mp3"]
    with pytest.raises(ValueError, match="Error: merge_meditation_audio - Empty topic_based_filename."):
        mvg.merge_meditation_audio()

    # add_audio_fx
    # 17. Error: add_audio_fx - working_directory does not exist:
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.working_directory = "i_do_not_exist"
    with pytest.raises(FileNotFoundError, match="Error: add_audio_fx - working_directory does not exist"):
        mvg.add_audio_fx()

    # 18. Error: add_audio_fx - Empty topic_based_filename.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.topic_based_filename = ""
    with pytest.raises(ValueError, match="Error: add_audio_fx - Empty topic_based_filename."):
        mvg.add_audio_fx()

    # 19. meditation_text_merged.mp3 does not exist in working directory.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(FileNotFoundError, match="meditation_text_merged.mp3 does not exist in working directory"):
        mvg.add_audio_fx()

    # mix_meditation_audio
    # 20. Error: mix_meditation_audio - working_directory does not exist:
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.working_directory = "i_do_not_exist"
    with pytest.raises(FileNotFoundError, match="Error: mix_meditation_audio - working_directory does not exist"):
        mvg.mix_meditation_audio()

    # 21. Error: mix_meditation_audio - Empty topic_based_filename.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.topic_based_filename = ""
    with pytest.raises(ValueError, match="Error: mix_meditation_audio - Empty topic_based_filename."):
        mvg.mix_meditation_audio()

    # 22. _meditation_text_merged_fx.mp3 does not exist in working directory.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(FileNotFoundError, match="_meditation_text_merged_fx.mp3 does not exist in working directory"):
        mvg.mix_meditation_audio()

    # 23 Error: generate_meditation_image - Empty image_model.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.image_model = ""
    with pytest.raises(ValueError, match="Error: generate_meditation_image - Empty image_model."):
        mvg.generate_meditation_image()

    # 24 Error: generate_meditation_image - Empty image_prompt.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.image_model = "fakemodel"
    mvg.image_prompt = ""
    with pytest.raises(ValueError, match="Error: generate_meditation_image - Empty image_prompt."):
        mvg.generate_meditation_image()

    # 25 Error: generate_meditation_image - Empty image_quality.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.image_model = "fakemodel"
    mvg.image_prompt = "fakeprompt"
    mvg.image_quality = ""
    with pytest.raises(ValueError, match="Error: generate_meditation_image - Empty image_quality."):
        mvg.generate_meditation_image()

    # 26 Error: generate_meditation_image - working_directory does not exist:
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.working_directory = "i_do_not_exist"
    with pytest.raises(FileNotFoundError, match="Error: generate_meditation_image - working_directory does not exist"):
        mvg.generate_meditation_image()

    # 27. Error: generate_meditation_image - Empty topic_based_filename.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.topic_based_filename = ""
    with pytest.raises(ValueError, match="Error: generate_meditation_image - Empty topic_based_filename."):
        mvg.generate_meditation_image()

    # 28. Error: generate_meditation_image - OpenAI API call failed
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.image_model = "fakemodel"
    mvg.image_prompt = "fakeprompt"
    mvg.image_quality = "fakequality"
    with pytest.raises(RuntimeError, match="Error: generate_meditation_image - OpenAI API call failed"):
        mvg.generate_meditation_image()

    # create_meditation_video
    # 29. Error: create_meditation_video - working_directory does not exist:
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.working_directory = "i_do_not_exist"
    with pytest.raises(FileNotFoundError, match="Error: create_meditation_video - working_directory does not exist"):
        mvg.create_meditation_video()

    # 30. Error: create_meditation_video - Empty topic_based_filename.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.topic_based_filename = ""
    with pytest.raises(ValueError, match="Error: create_meditation_video - Empty topic_based_filename."):
        mvg.create_meditation_video()

    # 31. _meditation_image.jpg does not exist.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(FileNotFoundError, match="_meditation_image.jpg does not exist."):
        mvg.create_meditation_video()

    # 32. _meditation_text_merged_fx_mixed.mp3 does not exist.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    # create trivial Mindfulness_meditation_image.jpg file in working directory
    # has to be a loadable jpeg file
    # use moviepy to write a black pixel image
    black_pixel = np.zeros((1, 1, 3), dtype=np.uint8)
    black_pixel[0, 0] = [0, 0, 0]
    black_pixel = ImageClip(black_pixel)
    black_pixel.save_frame(os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg"))
    with pytest.raises(FileNotFoundError, match="_meditation_text_merged_fx_mixed.mp3 does not exist."):
        mvg.create_meditation_video()

    # 33. Error: run_meditation_pipeline - Empty free_text list.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(ValueError, match="Error: run_meditation_pipeline - Empty free_text list."):
        mvg.run_meditation_pipeline(content=" ")

    # 34. Error: run_meditation_pipeline - Empty topic. (spanish translating pipeline)
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    mvg.in_spanish = True
    mvg.topic = ""
    mvg.pipeline = {
        "texts": False,
        "audio_files": False,
        "combine_audio_files": False,
        "audio_fx": False,
        "background_audio": False,
        "image": False,
        "keywords": True,
        "video": False
    }

    mock_return_value = {
        "choices": [
            {
                "message": {
                    "content": "keyword1, keyword2, keyword3, keyword4, keyword5",
                }
            }
        ]
    }
    mock_return_value = DotDict(mock_return_value)
    with patch.object(mvg.client.chat.completions, 'create', return_value=mock_return_value):
        with pytest.raises(ValueError, match="Error: run_meditation_pipeline - Empty topic."):
            mvg.run_meditation_pipeline()

    # 35. Error: generate_keywords - num_keywords must be a positive integer.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(ValueError, match="Error: generate_keywords - num_keywords must be a positive integer."):
        mvg.generate_keywords(num_keywords=-1)

    # 36. Error: translate_text - text must be a non-empty string.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(ValueError, match="Error: translate_text - text must be a non-empty string."):
        mvg.translate_text(text=" ")

    # 37. Error: translate_text - target_language must be a non-empty string.
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    with pytest.raises(ValueError, match="Error: translate_text - target_language must be a non-empty string."):
        mvg.translate_text(text="test", target_language=" ")

    # TESTS OF THE BANNER GENERATION SYSTEM
    # add_banner_with_text()
    # image_path: str, banner_text: str, output_path: str
    # 38. Check add_banner_with_text() raises for invalid image_path f"Error: add_banner_with_text - Image file does not exist: {image_path}"
    mvg = MeditationVideoGenerator(force_working_dir_overwrite=True)
    image_path = "i_do_not_exist.jpg"
    banner_text = "test"
    output_path = "output.jpg"
    with pytest.raises(FileNotFoundError, match=f"Error: add_banner_with_text - Image file does not exist: {image_path}"):
        mvg.add_banner_with_text(image_path, banner_text, output_path)

    # 39. check image_path is a valid image file
    # Error: add_banner_with_text - Failed to load image file
    # create a text file and name it as a jpg
    with open(os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg"), "w") as f:
        f.write("test")
    image_path = os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg")
    with pytest.raises(ValueError, match="Error: add_banner_with_text - Failed to load image file"):
        mvg.add_banner_with_text(image_path, banner_text, output_path)

    # 40. check banner_text is a non-empty string
    # create a 200 x 200 pixel image
    black_pixel = np.zeros((200, 200, 3), dtype=np.uint8)
    image_path = os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg")
    black_pixel = ImageClip(black_pixel)
    black_pixel.save_frame(image_path)
    banner_text = ""
    with pytest.raises(ValueError, match="banner text is empty"):
        mvg.add_banner_with_text(image_path, banner_text, output_path)

    # 41. check image not too small
    black_pixel = np.zeros((10, 10, 3), dtype=np.uint8)
    image_path = os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg")
    black_pixel = ImageClip(black_pixel)
    black_pixel.save_frame(image_path)
    banner_text = ""
    with pytest.raises(ValueError, match="Image dimensions too small"):
        mvg.add_banner_with_text(image_path, banner_text, output_path)
    # 42. check self.banner_height_ratio is valid - LEAVE OUT - Too hard to simulate
    # 43. check self.banner_font_size is valid - LEAVE OUT- Too hard to simulate
    # 44. check output path directory exists
    image_path = os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg")
    black_pixel = np.zeros((200, 200, 3), dtype=np.uint8)
    black_pixel = ImageClip(black_pixel)
    black_pixel.save_frame(image_path)
    banner_text = "test"
    output_path = os.path.join(mvg.working_directory, "does_not_exist", "output.jpg")
    with pytest.raises(FileNotFoundError, match=f"Error: add_banner_with_text - Output directory does not exist: {os.path.join(mvg.working_directory, 'does_not_exist')}"):
        mvg.add_banner_with_text(image_path, banner_text, output_path)

    # delete the black pixel file
    if os.path.exists(os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg")):
        os.remove(os.path.join(mvg.working_directory, "Mindfulness_meditation_image.jpg"))

    # remove output.mp3
    if os.path.exists("output.mp3"):
        os.remove("output.mp3")
    # remove merged_output.mp3
    if os.path.exists("merged_output.mp3"):
        os.remove("merged_output.mp3")

    if os.path.exists("fakefile1.mp3"):
        # remove fakefile1.mp3 etc
        for i in range(1, 4):
            os.remove(f"fakefile{i}.mp3")
        # empty the working directory
        for file in os.listdir(mvg.working_directory):
            os.remove(os.path.join(mvg.working_directory, file))
        # delete the working directory
        shutil.rmtree(mvg.working_directory)
