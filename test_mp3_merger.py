import pytest
from pydub import AudioSegment
from pydub.generators import WhiteNoise, Sine
from meditation_video_generator.mp3_merger import MP3Merger
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def noise_merge(N: int):
    test_dir = "test_mp3_merger"
    # Create the test directory if it does not exist

    # Delete it if it exists
    if os.path.isdir(test_dir):
        # Clear out files
        for file in os.listdir(test_dir):
            os.remove(f"{test_dir}/{file}")
        os.rmdir(test_dir)
    os.makedirs(test_dir)

    noise_duration = 1000
    for i in range(N):
        noise = WhiteNoise()
        noise_segment = noise.to_audio_segment(duration=noise_duration)  # 1 second
        noise_segment.export(f"{test_dir}/noise_{i}.mp3", format="mp3")

    # Test the mp3 merger using the generated noise files
    files = [f"{test_dir}/noise_{i}.mp3" for i in range(N)]
    total_test_duration = 2 * N * int(noise_duration / 1000) + 1
    mp3_merger = MP3Merger(files, duration=total_test_duration)

    mp3_merger.merge()
    print(f"Merged MP3 file saved as {mp3_merger.output_file}")

    # Load in the mp3 and test it contains the correct number of tones
    merged_mp3 = AudioSegment.from_mp3(mp3_merger.output_file)
    duration = merged_mp3.duration_seconds
    print(f"Total duration of merged MP3 file: {duration} seconds")
    print("Expected duration with buffers: ", total_test_duration +
          mp3_merger.rear_buffer / 1000 + mp3_merger.front_buffer / 1000)
    print("Expected silence duration without buffers: ", total_test_duration - N * noise_duration / 1000)

    assert pytest.approx(duration,
                         0.01) == total_test_duration + mp3_merger.rear_buffer / 1000 + mp3_merger.front_buffer / 1000

    # Convert to a numpy array
    audio_array = merged_mp3.get_array_of_samples()
    # Do an amplitude analysis to find the number of tones
    audio_array = np.abs(np.array(audio_array))
    # Threshold the audio array at power = +/- 1
    audio_array = np.where(audio_array > 1, 1, 0)

    # Clean up all files
    for file in os.listdir(test_dir):
        os.remove(f"{test_dir}/{file}")
    os.rmdir(test_dir)

    # Delete output file
    os.remove(mp3_merger.output_file)

    PLOT = False
    if PLOT:
        matplotlib.use('TkAgg')  # for PyCharm
        # Plot against seconds not samples
        x_seconds = np.linspace(0, duration, len(audio_array))
        plt.plot(x_seconds, audio_array)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Amplitude of the merged MP3 file")
        plt.show()


def test_noise_merge():
    noise_merge(2)
    noise_merge(4)
    noise_merge(5)
    noise_merge(10)


def test_edge_cases():
    # Test the edge cases of the mp3 merger
    # Create a fake mp3 called test.mp3 which is just a text file
    with open("test.mp3", "w") as f:
        f.write("hello")
    with open("test2.mp3", "w") as f:
        f.write("hello")

    # 0. At least two MP3 files are required
    with pytest.raises(ValueError, match="At least two MP3 files are required."):
        mp3_merger = MP3Merger(["test.mp3"], duration=10)
        mp3_merger.merge()

    # 1. No MP3 files provided
    with pytest.raises(ValueError, match="No MP3 files provided."):
        mp3_merger = MP3Merger([], duration=10)
        mp3_merger.merge()

    # 2. Duration must be greater than zero
    with pytest.raises(ValueError, match="Duration must be greater than zero."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=0)
        mp3_merger.merge()
    with pytest.raises(ValueError, match="Duration must be greater than zero."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=-1)
        mp3_merger.merge()

    # 3. All input files must exist
    with pytest.raises(ValueError, match="All input files must exist."):
        mp3_merger = MP3Merger(["idontexist.mp3", "test2.mp3"], duration=10)
        mp3_merger.merge()

    # 4. All input files must be in MP3 format
    with open("test.txt", "w") as f:
        f.write("hello")
    with pytest.raises(ValueError, match="All input files must be in MP3 format."):
        mp3_merger = MP3Merger(["test.txt", "test2.mp3"], duration=10)
        mp3_merger.merge()
    os.remove("test.txt")

    # 5. Front buffer must be positive
    with pytest.raises(ValueError, match="Front buffer must be positive."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=10, front_buffer=-1)
        mp3_merger.merge()
    # Rear buffer must be positive
    with pytest.raises(ValueError, match="Rear buffer must be positive."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=10, rear_buffer=-1)
        mp3_merger.merge()

    # 6. Balance for odd segments must be between -1 and 1
    with pytest.raises(ValueError, match="Balance for odd segments must be between -1 and 1."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=10, balance_odd=-2)
        mp3_merger.merge()
    with pytest.raises(ValueError, match="Balance for odd segments must be between -1 and 1."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=10, balance_odd=2)
        mp3_merger.merge()

    # 7. Balance for even segments must be between -1 and 1
    with pytest.raises(ValueError, match="Balance for even segments must be between -1 and 1."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=10, balance_even=-2)
        mp3_merger.merge()
    with pytest.raises(ValueError, match="Balance for even segments must be between -1 and 1."):
        mp3_merger = MP3Merger(["test.mp3", "test2.mp3"], duration=10, balance_even=2)
        mp3_merger.merge()

    # 8. test when total voice duration is greater than the silence duration
    # generate 6 11 second files of sine waves
    for i in range(6):
        with open(f"merge_test{i}.mp3", "wb") as f:
            Sine(440).to_audio_segment(duration=11000).export(f, format="mp3")
    # check value error is raised and includes text: Total duration of input MP3 files exceeds the target duration.
    with pytest.raises(ValueError, match="Total duration of input MP3 files exceeds the target duration."):
        mp3_merger = MP3Merger([f"merge_test{i}.mp3" for i in range(6)], duration=60)
        mp3_merger.merge()
    # delete test files
    for i in range(6):
        os.remove(f"merge_test{i}.mp3")


    # Delete test files
    os.remove("test.mp3")
    os.remove("test2.mp3")

# Uncomment the line below to run tests directly if this script is executed
# pytest.main(["-v"])
