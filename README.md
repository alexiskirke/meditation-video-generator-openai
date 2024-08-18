# MeditationVideoGenerator

The `MeditationVideoGenerator` class is a comprehensive solution for creating meditation videos. It leverages various libraries and services to generate text, synthesize speech, apply audio effects, and combine audio and video into a cohesive meditation experience.

## Installation

```bash
pip install meditation-video-generator-openai
```

## Usage

To use the `MeditationVideoGenerator` class, you need to initialize it with various parameters and then run the pipeline to generate the meditation video.
The generation also displays up to 30 suggested keywords you can use if uploading to social media.
A banner is placed at the bottom of the generated image with a title that summarises your topic.
The system uses copyright-free (or copyright-YouTube-authorised) music/sound extracts unless you specify binaural beats.
This music is downloaded the first time you create the MeditationVideoGenerator object.
Note: when the meditation has been generated, the working directory is left in place but can be manually deleted. The video and audio file are copied into your own working directory.


## Known Issues

GPT-4o is a statistical model and has therefore occasionally returned JSON that is not parsable by this code - this will raise and error and abort. 
Similarly, any parameters which attempt to control the elements returned from GPT-4o may not give the precise results expected.
OpenAI have a rate limiter on their API. For example, if you create a meditation with dozens of text parts, then during synthesis you may get rate limited and the process aborts.
Sometimes a perfectly innocent meditation topic can trigger OpenAI's content filter for picture generation. This will cause the meditation to abort. (But you can re-run from that point on by adjusting the pipeline, see below.)

### Initialization Parameters

- `topic: str` - The topic of the meditation.
- `length: float` - The length of the meditation in minutes.
- `api_key: str` - OpenAI API key for accessing the speech synthesizer.
- `base_on_text: bool` - If True, generate a meditation based on (but not consisting only of) the provided text.
- `text: str` - Text to generate the meditation from.
- `num_sentences: int` - Number of sentences in each part of the meditation.
- `bass_boost: bool` - If True, apply bass boost to the voice audio.
- `balance_even: float` - The balance for even segments (0.0 is centered, -1.0 is left, 1.0 is right).
- `balance_odd: float` - The balance for odd segments (0.0 is centered, -1.0 is left, 1.0 is right).
- `two_voices: bool` - If True, use two different voices for odd and even parts of the meditation.
- `expand_on_section: bool` - If True, expand on each section of the meditation in more detail.
- `sounds_dir: str` - Directory containing ambient sounds.
- `limit_parts: int` - Limit the number of parts in the meditation.
- `num_loops: int` - Allows all voice parts to be looped multiple times across the whole meditation (e.g. repeat a single affirmation).
- `in_spanish: bool` - If True, generate the meditation in Spanish.
- `affirmations_only: bool` - If True, generate affirmations only, not meditations.
- `binaural: bool` - If True, generate binaural beats.
- `all_voices: bool` - If True, choose from all available voices (not just the "best" two).
- `binaural_fade_out_duration: float` - Duration of the fade-out at the end of the binaural beats.
- `start_beat_freq: float` - Initial frequency difference for the binaural effect.
- `end_beat_freq: float` - Final frequency difference for the binaural effect.
- `base_freq: float` - Base frequency for the binaural beats.
- `sample_rate: int` - Sample rate for the mixer.
- `num_samples_to_chop: int` - Number of samples to chop off the beginning of the ambient sound files.
- `fade_in_time: float` - Duration of the fade-in at the beginning of the ambient sound overlay.
- `fade_out_time: float` - Duration of the fade-out at the end of the ambient sound overlay.
- `expansion_size: int` - Size of the expansion for the text, if `expand_on_section` is `True`.
- `force_voice: str` - Allows user to force selection of OpenAI voice, e.g., `onyx`.
- `image_background_opacity: float` - Opacity of the background image (0 = transparent, 1 = opaque).
- `banner_at_bottom: bool` - If `True`, the banner is at the bottom of the image.
- `banner_height_ratio: float` - Ratio of the banner height to the image height.
- `max_banner_words: int` - Maximum number of words in the banner.
- `power_ratio: float` - Ratio (higher means louder voice) of the power of the binaural beats or ambient to the power of the voice audio.

### Example Usages

The following will generate a 10-minute meditation video with audio and a fixed image inspired by the Shire from Lord of the Rings, with approximately 7 sentences per meditation segment:
```python
from meditation_video_generator import MeditationVideoGenerator

# Initialize the meditation video generator
mvg = MeditationVideoGenerator(length=10, api_key="YOUR_API_KEY",
                               topic="""A calming stroll through Hobbiton in the Shire""",
                               binaural=False,
                               num_sentences=7, 
                               bass_boost=True)
subsections, video_filename = mvg.run_meditation_pipeline()
# subsections is a list of the generated meditation texts
# video_filename is the path to the generated meditation video
```

Generate a 5-minute meditation video with binaural beats and more detailed wordings. The binaural base frequency is 220Hz and over the 5 minutes the beat frequency will reduce from 32Hz to 2Hz:
```python
from meditation_video_generator import MeditationVideoGenerator

mvg = MeditationVideoGenerator(length=5, api_key="YOUR_API_KEY",
                               topic="""Dealing with work pressures""",
                               binaural=True,
                               bass_boost=True,
                               expand_on_section=True,
                               start_beat_freq = 32,  
                               end_beat_freq = 2,  
                               base_freq = 220)
subsections, video_filename = mvg.run_meditation_pipeline()
```

Generate a 20-minute affirmation video:
```python
from meditation_video_generator import MeditationVideoGenerator

mvg = MeditationVideoGenerator(length=20, api_key="YOUR_API_KEY",
                               topic="""You are loved""",
                               affirmations_only=True)
subsections, video_filename = mvg.run_meditation_pipeline()
```

Generate a 20-minute affirmation video just repeating the phrase "I am a good person who deserves good things":
```python
from meditation_video_generator import MeditationVideoGenerator

mvg = MeditationVideoGenerator(length=20, api_key="YOUR_API_KEY",
                               topic="I am a good person who deserves good things",
                               num_loops=200)
subsections, video_filename = mvg.run_meditation_pipeline(
    content = "I am a good person who deserves good things")
```

Generate a 5-minute affirmation video directly from the provided text.
Use two different voices for odd and even parts of the meditation with one voice on the left and the other on the right.
Note the double newlines in the content string:
```python
from meditation_video_generator import MeditationVideoGenerator

# Initialize the meditation video generator
mvg = MeditationVideoGenerator(length=5, api_key="YOUR_API_KEY",
                               topic="A beautiful poem",
                               two_voices=True,
                               balance_even=-0.85,
                               balance_odd=0.85)
subsections, video_filename = mvg.run_meditation_pipeline(
    content = """Do not stand at my grave and weep

    I am not there. I do not sleep.
    
    I am a thousand winds that blow.
    
    I am the diamond glints on snow.
    
    I am the sunlight on ripened grain.
    
    I am the gentle autumn rain.
    
    When you awaken in the morning's hush
    
    I am the swift uplifting rush
    
    Of quiet birds in circled flight.
    
    I am the soft stars that shine at night.
    
    Do not stand at my grave and cry;
    
    I am not there. I did not die. """)
```

Generate just the image and video and keywords, assuming a previous run of the pipeline did the rest:
```python
from meditation_video_generator import MeditationVideoGenerator

# Initialize the meditation video generator
mvg = MeditationVideoGenerator(length=20, api_key="YOUR_API_KEY",
                               topic="""Python coders are the best""",
                               affirmations_only=True)
mvg.pipeline = {
            "texts": False,
            "audio_files": False,
            "combine_audio_files": False,
            "audio_fx": False,
            "background_audio": False,
            "image": True,
            "video": True,
            "keywords": True
        }
subsections, video_filename = mvg.run_meditation_pipeline()
```

