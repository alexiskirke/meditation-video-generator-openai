import pytest
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.generators import WhiteNoise
from meditation_video_generator.mp3_mixer import MP3Mixer
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use tkinter for matplotlib
matplotlib.use('TkAgg')


def noise_mix():
    test_dir = "test_mp3_mixer"
    # Create the test directory if it does not exist
    if os.path.isdir(test_dir):
        # Clear out files
        for file in os.listdir(test_dir):
            if os.path.isfile(f"{test_dir}/{file}"):
                os.remove(f"{test_dir}/{file}")
            else:
                os.rmdir(f"{test_dir}/{file}")
        os.rmdir(test_dir)
    os.makedirs(test_dir)

    # Clean up all files
    for file in os.listdir(test_dir):
        os.remove(f"{test_dir}/{file}")
    os.rmdir(test_dir)


def test_edge_cases():
    test_dir = "test_mp3_mixer"
    # Create the test directory if it does not exist
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    else:
        # If it does exist, clear out all files
        for file in os.listdir(test_dir):
            if os.path.isfile(f"{test_dir}/{file}"):
                os.remove(f"{test_dir}/{file}")
            else:
                for filename in os.listdir(f"{test_dir}/{file}"):
                    os.remove(f"{test_dir}/{file}/{filename}")
                os.rmdir(f"{test_dir}/{file}")

    # Create fake MP3 files for overlay ambient
    with open(f"{test_dir}/test.mp3", "wb") as f:
        # one second tone 440 Hz as audiosegment to mp3 to f
        Sine(440).to_audio_segment(duration=1000).export(f, format="mp3")
    with open(f"{test_dir}/test2.mp3", "wb") as f:
        #WhiteNoise().to_audio_segment(duration=1000).export(f, format="mp3")
        Sine(440).to_audio_segment(duration=1000).export(f, format="mp3")
    with open(f"{test_dir}/test3_shorter.mp3", "wb") as f:
        Sine(440).to_audio_segment(duration=500).export(f, format="mp3")
        #WhiteNoise().to_audio_segment(duration=500).export(f, format="mp3")
    with open(f"{test_dir}/test4_silent.mp3", "wb") as f:
        AudioSegment.silent(duration=1000).export(f, format="mp3")

    # 1. Test for unfound spoken_file_a
    mp3_mixer = MP3Mixer(mp3_file="not_used_in_test.mp3")
    spoken_file_a = f"{test_dir}/i_dont_exist.mp3"
    ambient_file_b = f"{test_dir}/test2.mp3"
    with pytest.raises(FileNotFoundError):
        mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)

    # 2. Test for unfound ambient file
    spoken_file_a = f"{test_dir}/test.mp3"
    ambient_file_b = f"{test_dir}/i_dont_exist.mp3"
    with pytest.raises(FileNotFoundError):
        mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)

    # 3. Test for shorter ambient file
    spoken_file_a = f"{test_dir}/test.mp3"
    ambient_file_b = f"{test_dir}/test3_shorter.mp3"
    with pytest.raises(ValueError):
        mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)

    # 4. Test for equal files
    mp3_mixer.num_samples_to_chop = 0
    mp3_mixer.fade_in_time = 0
    mp3_mixer.fade_out_time = 0
    spoken_file_a = f"{test_dir}/test.mp3"
    ambient_file_b = f"{test_dir}/test.mp3"
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    assert result.duration_seconds == 1.0

    # 5. Test for shorter spoken file
    spoken_file_a = f"{test_dir}/test3_shorter.mp3"
    ambient_file_b = f"{test_dir}/test2.mp3"
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    assert result.duration_seconds == 0.5

    # 6. Test for fade in
    mp3_mixer.fade_in_time = 1
    spoken_file_a = f"{test_dir}/test.mp3"
    ambient_file_b = f"{test_dir}/test.mp3"
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    assert result.duration_seconds == 1.0

    # 7. Test for fade out
    mp3_mixer.fade_in_time = 0
    mp3_mixer.fade_out_time = 1
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    assert result.duration_seconds == 1.0

    # 8. Test for fade in and fade out
    mp3_mixer.fade_in_time = 1
    mp3_mixer.fade_out_time = 1
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    assert result.duration_seconds == 1.0

    # 9. Test fade in makes first half of file lower energy than second half of file
    mp3_mixer.fade_in_time = 1
    mp3_mixer.fade_out_time = 0
    #mp3_mixer.power_ratio = 100 #mp3_mixer.power_ratio/100  # lower_voice part
    # spoken_file_a = f"{test_dir}/test4_silent.mp3"
    spoken_file_a = f"{test_dir}/test.mp3"
    ambient_file_b = f"{test_dir}/test2.mp3"
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    result_np = np.array(result.get_array_of_samples())
    first_half = result_np[:len(result_np) // 2]
    second_half = result_np[len(result_np) // 2:]
    energy_first_half = np.sum(np.abs(first_half))
    energy_second_half = np.sum(np.abs(second_half))
    assert energy_first_half < energy_second_half

    # 10. Test fade out makes first half of file higher energy than second half of file
    mp3_mixer.fade_in_time = 0
    mp3_mixer.fade_out_time = 1
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    result_np = np.array(result.get_array_of_samples())
    first_half = result_np[:len(result_np) // 2]
    second_half = result_np[len(result_np) // 2:]
    energy_first_half = np.sum(np.abs(first_half))
    energy_second_half = np.sum(np.abs(second_half))
    assert energy_first_half > energy_second_half

    # 11. Test that a 1-second fade in and fade out give an energy ratio of first to second half close to 1
    mp3_mixer.fade_in_time = 1
    mp3_mixer.fade_out_time = 1
    result = mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)
    result_np = np.array(result.get_array_of_samples())
    first_half = result_np[:len(result_np) // 2]
    second_half = result_np[len(result_np) // 2:]
    energy_first_half = np.sum(np.abs(first_half))
    energy_second_half = np.sum(np.abs(second_half))
    assert 0.9 < energy_first_half / energy_second_half < 1.1

    # 12. Test a negative fade in
    mp3_mixer.fade_in_time = -1
    mp3_mixer.fade_out_time = 0
    with pytest.raises(ValueError):
        mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)

    # 13. Test a negative fade out
    mp3_mixer.fade_in_time = 0
    mp3_mixer.fade_out_time = -1
    with pytest.raises(ValueError):
        mp3_mixer.overlay_ambient(spoken_file_a=spoken_file_a, ambient_file_b=ambient_file_b)

    # Now some tests for mix_audio()

    # 14. Is mp3_file an empty string?
    mp3_mixer = MP3Mixer(mp3_file="")
    with pytest.raises(ValueError):
        mp3_mixer.mix_audio()

    # 15. Does mp3_file exist?
    mp3_file = f"{test_dir}/i_dont_exist.mp3"
    mp3_mixer = MP3Mixer(mp3_file=mp3_file)
    with pytest.raises(FileNotFoundError):
        mp3_mixer.mix_audio()

    # 16. Is it an MP3 file?
    # Create a text file in test-dir
    text_file = f"{test_dir}/test.txt"
    with open(text_file, "w") as f:
        f.write("This is a text file.")
    mp3_file = text_file
    mp3_mixer = MP3Mixer(mp3_file=mp3_file)
    with pytest.raises(ValueError):
        mp3_mixer.mix_audio()
    os.remove(text_file)

    # 17. Check start beat freq > 0
    mp3_file = f"{test_dir}/test.mp3"
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, start_beat_freq=-1, binaural=True)
    with pytest.raises(ValueError):
        mp3_mixer.mix_audio()

    # 18. Check end beat freq > 0
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, end_beat_freq=-1, binaural=True)
    with pytest.raises(ValueError):
        mp3_mixer.mix_audio()

    # 19. Check base freq > 0
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, base_freq=-1, binaural=True)
    with pytest.raises(ValueError):
        mp3_mixer.mix_audio()

    # 20. Check binaural fade out duration >= 0
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, binaural_fade_out_duration=-1, binaural=True)
    with pytest.raises(ValueError):
        mp3_mixer.mix_audio()

    # 21. If binaural is false then start_beat_freq, end_beat_freq, binaural_fade_out_duration, and base_freq can have any values
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, start_beat_freq=-1, end_beat_freq=-1,
                         base_freq=-1, binaural_fade_out_duration=-1, binaural=False)
    result = mp3_mixer.mix_audio()
    assert result == "output.mp3"
    # delete the output file
    os.remove("output.mp3")

    # 22. Binaural = False case and check if sounds_dir exists
    sounds_dir = "i_dont_exist"
    sounds_dir = os.path.join(os.getcwd(), sounds_dir)
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, binaural=False, sounds_dir=sounds_dir)
    with pytest.raises(FileNotFoundError):
        mp3_mixer.mix_audio()

    # 23. Binaural = False case and check if sounds_dir contains any MP3 files
    # Create a new dir which only has a text file
    sounds_dir = f"{test_dir}/sounds"
    os.mkdir(sounds_dir)
    text_file = f"{sounds_dir}/test.txt"
    with open(text_file, "w") as f:
        f.write("This is a text file.")
    sounds_dir = os.path.join(os.getcwd(), sounds_dir)
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, binaural=False, sounds_dir=sounds_dir)
    with pytest.raises(FileNotFoundError):
        mp3_mixer.mix_audio()
    os.remove(text_file)

    # 24. Binaural = False case and dir empty
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, binaural=False, sounds_dir=sounds_dir)
    with pytest.raises(FileNotFoundError):
        mp3_mixer.mix_audio()
    os.rmdir(sounds_dir)

    # 25. Test for valid power ratio values
    mp3_mixer = MP3Mixer(mp3_file=mp3_file, power_ratio=-1)
    with pytest.raises(ValueError):
        mp3_mixer.mix_audio()

    # end of tests
    # empty and remove the test directory
    for file in os.listdir(test_dir):
        os.remove(f"{os.path.join(os.getcwd(),test_dir,file)}")
    os.rmdir(test_dir)




