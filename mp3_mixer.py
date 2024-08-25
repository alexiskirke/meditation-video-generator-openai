import os
import random
from pydub import AudioSegment
import numpy as np


class MP3Mixer:
    """
    MP3Mixer class to mix an MP3 file with binaural beats and ambient sounds.
    Initialize the MP3Mixer with the path to the input MP3 file and the working directory.

    :param mp3_file: Path to the input MP3 file.
    :param working_dir: Working directory to store the output files.
    :param sounds_dir: Directory containing the ambient sound files (a random file is selected for each mix).
    :param binaural: Flag to enable binaural beats (if False, ambient sound will be mixed).
    :param binaural_fade_out_duration: Duration of the fade-out for binaural beats.
    :param start_beat_freq: Starting frequency of the binaural beats.
    :param end_beat_freq: Ending frequency of the binaural beats.
    :param base_freq: Base frequency for the binaural beats.
    :param sample_rate: Sample rate for the audio files.
    :param num_samples_to_chop: Number of samples to chop from the start of the ambient file (in case of introductions).
    :param fade_in_time: Fade-in time for the ambient file.
    :param fade_out_time: Fade-out time for the ambient file.
    :param power_ratio: Power ratio (higher means louder voice) between the voice and the binaural beats or the ambient sound.
    """
    def __init__(self, mp3_file: str, working_dir: str = "",
                 sounds_dir: str = "ambient_files",
                 binaural: bool = False,
                 binaural_fade_out_duration: float = 4,
                 start_beat_freq: float = 5,
                 end_beat_freq: float = 0.5,
                 base_freq: float = 110.0,
                 sample_rate: int = 44100,
                 num_samples_to_chop: int = 30000,
                 fade_in_time: float = 4,
                 fade_out_time: float = 4,
                 ambient_file_gain_db: int = -20,
                 power_ratio=None
                 ):
        self.binaural_fade_out_duration = binaural_fade_out_duration
        self.mp3_file = mp3_file
        self.working_dir = working_dir
        self.binaural = binaural
        self.output_file = os.path.join(self.working_dir, "output.mp3")
        self.start_beat_freq = start_beat_freq
        self.end_beat_freq = end_beat_freq
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        self.sounds_dir = sounds_dir
        self.num_samples_to_chop = num_samples_to_chop
        self.fade_in_time = fade_in_time
        self.fade_out_time = fade_out_time
        self.ambient_file_gain_db = ambient_file_gain_db
        self.power_ratio = power_ratio

    @staticmethod
    def calculate_average_power(audio_segment: AudioSegment) -> np.ndarray:
        # Convert the audio segment into an array of samples
        samples = np.array(audio_segment.get_array_of_samples())
        # Determine the number of channels in the audio (e.g., 1 for mono, 2 for stereo)
        channels = audio_segment.channels
        # Reshape the samples array to separate the channels.
        # If there are multiple channels, the array will have a shape like (n_samples, n_channels).
        samples = samples.reshape((-1, channels))
        # Convert the samples to float64 type to prevent any issues with integer overflow
        # when performing mathematical operations. This is especially important when
        # dealing with audio data, as the samples are typically integers (e.g., 16-bit or 32-bit).
        samples = np.asarray(samples, dtype=np.float64)
        # Calculate the power (mean squared value) for each channel separately.
        # This involves squaring each sample value (which gives us the power),
        # and then taking the mean across all samples in each channel.
        power_per_channel = np.mean(samples ** 2, axis=0)
        # Compute the average power across all channels.
        # If the audio is stereo, this will be the mean of the power of the left and right channels.
        average_power = np.mean(power_per_channel)
        return average_power

    def adjust_power_overlay_and_normalise(self, stereo_audio_segment_1: AudioSegment,
                                           stereo_audio_segment_2: AudioSegment) -> AudioSegment:
        """
        Overlay and normalize two stereo audio segments with a given power ratio.

        :param stereo_audio_segment_1: First stereo audio segment.
        :param stereo_audio_segment_2: Second stereo audio segment.
        :return: Normalized stereo audio segment after overlaying and adjusting power.
        """
        # Calculate the average power of the first stereo audio segment.
        power_1 = self.calculate_average_power(stereo_audio_segment_1)  # speech
        # Calculate the average power of the second stereo audio segment.
        power_2 = self.calculate_average_power(stereo_audio_segment_2) # (music / binaural)
        """A larger self.power_ratio increases the voice loudness. But it is a ratio
        so it will also reduce the music / binaural loudness. 
        The power of an audio signal is proportional to the square of its amplitude (volume).
        The square root is applied because power is proportional to the square of the amplitude. To scale the amplitude to achieve the desired power, you need to take the square root of the ratio of the powers.
        Without the square root, you'd be adjusting power directly rather than amplitude, which would result in an incorrect adjustment.
        """
        scaling_factor = np.sqrt((power_2 * self.power_ratio) / power_1)
        # Step 4: Adjust the gain of the second (music / binaural) audio segment based on the scaling factor.
        # The scaling factor is converted to a decibel value (dB), which is applied as a gain reduction
        # to the second segment. The negative sign ensures that the power ratio is maintained.
        # apply_gain() applies a gain adjustment in decibels to the audio segment.
        # so because power scaling factor is based on power, we need to convert it to dB
        # using the formula: 10 * log10(scaling_factor)
        # the -ve is to reduce the gain of the second (music / binaural) audio segment
        adjusted_audio_segment_2 = stereo_audio_segment_2.apply_gain(-10 * np.log10(scaling_factor))
        combined = stereo_audio_segment_1.overlay(adjusted_audio_segment_2)
        final_stereo_audio = combined.normalize()
        return final_stereo_audio

    def overlay_ambient(self, spoken_file_a: str, ambient_file_b: str) -> AudioSegment:
        if not os.path.exists(spoken_file_a):
            raise FileNotFoundError(f"Spoken audio file {spoken_file_a} not found.")
        if not os.path.exists(ambient_file_b):
            raise FileNotFoundError(f"Ambient audio file {ambient_file_b} not found.")
        spoken_audio_a = AudioSegment.from_file(spoken_file_a)
        ambient_audio_b = AudioSegment.from_file(ambient_file_b)
        spoken_duration_a = len(spoken_audio_a)
        ambient_audio_b = ambient_audio_b[self.num_samples_to_chop:]
        max_start_time = len(ambient_audio_b) - spoken_duration_a
        if max_start_time < 0:
            raise ValueError("Overlay_ambient: Ambient audio file is not long enough to extract the desired segment.")
        start_time = random.randint(0, max_start_time)
        ambient_segment_b = ambient_audio_b[start_time:start_time + spoken_duration_a]
        if self.fade_in_time > 0:
            ambient_segment_b = ambient_segment_b.fade_in(self.fade_in_time * 1000)
        elif self.fade_in_time < 0:
            raise ValueError("Overlay_ambient: Fade-in time must be greater than 0.")
        if self.fade_out_time > 0:
            ambient_segment_b = ambient_segment_b.fade_out(self.fade_out_time * 1000)
        elif self.fade_out_time < 0:
            raise ValueError("Overlay_ambient: Fade-out time must be greater than 0.")
        if not self.power_ratio:
            self.power_ratio = 7500  # 5000 # increasing this reduces the ambient volume
        overlaid_audio = self.adjust_power_overlay_and_normalise(spoken_audio_a,
                                                          ambient_segment_b)
        return overlaid_audio

    def generate_binaural_beats(self, duration: int) -> AudioSegment:
        if duration <= 0:
            raise ValueError("generate_binaural_beats: Binaural duration must be greater than 0.")
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        beat_freq = np.linspace(self.start_beat_freq, self.end_beat_freq, len(t))
        left_wave = np.sin(2 * np.pi * self.base_freq * t)
        right_wave = np.sin(2 * np.pi * (self.base_freq + beat_freq) * t)
        """What makes this complicated is that certain beat frequency sweeps seem to lead to
        the beat going in the opposite direction at some point. This is not what we want.
        So first we calculate the rate of maxima close to one and check it is moving in one direction."""
        wave_diff = left_wave - right_wave
        # find the number of peaks > 1- epsilon in wave_diff for each 0.5 second interval
        num_maxima = []
        for i in range(0, len(wave_diff), int(self.sample_rate / 2)):
            num_maxima.append(len(np.where(wave_diff[i:i + int(self.sample_rate / 2)] > 1-1e-3)[0]))
        num_maxima = np.array(num_maxima) # this is the rate of maxima per 0.5 seconds
        # calculate coefficient of variation across num_maxima
        cv = 100*np.std(num_maxima) / np.mean(num_maxima)
        # if this rate of maxmima varies too much, there is a cross over point where the beat goes in the opposite direction
        if cv >= 5:
            # find the global minimum of num_maxima
            min_idx = np.argmin(num_maxima) # this is the cross over point
            # get estimated time index
            est_time_idx = min_idx * int(self.sample_rate / 2)
            # convert to seconds
            est_time = est_time_idx / self.sample_rate
            # what would the beat frequency be at this time?
            est_beat_freq = beat_freq[est_time_idx]
            # now set the end beat frequency to this
            # rounding up or down depending what way we're going.
            if self.start_beat_freq > self.end_beat_freq:
                # round up the estimated beat frequency to the nearest integer
                self.end_beat_freq = np.ceil(est_beat_freq)
            else: # round down
                # round down the estimated beat frequency to the nearest integer
                self.start_beat_freq = np.floor(est_beat_freq)
        # reconfigure for the new end beat frequency
        beat_freq = np.linspace(self.start_beat_freq, self.end_beat_freq, len(t))
        left_wave = np.sin(2 * np.pi * self.base_freq * t)
        right_wave = np.sin(2 * np.pi * (self.base_freq + beat_freq) * t)
        stereo_wave = np.vstack((left_wave, right_wave)).T.flatten()
        audio_array = (stereo_wave * 32767).astype(np.int16)
        binaural_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=2
        )
        binaural_segment = binaural_segment.fade_out(self.binaural_fade_out_duration)
        return binaural_segment

    def mix_audio(self) -> str:
        if not self.mp3_file:
            raise ValueError("Error: mp3_file is an empty string.")
        if not os.path.exists(self.mp3_file):
            raise FileNotFoundError(f"Error: {self.mp3_file} does not exist.")
        if not self.mp3_file.endswith(".mp3"):
            raise ValueError(f"Error: {self.mp3_file} is not an MP3 file.")
        if self.power_ratio and self.power_ratio < 0:
            raise ValueError(f"Power ratio must be positive, not {self.power_ratio}.")
        input_audio = AudioSegment.from_mp3(self.mp3_file)
        duration = input_audio.duration_seconds
        if self.binaural:
            if self.start_beat_freq <= 0:
                raise ValueError("Error: start_beat_freq must be greater than 0.")
            if self.end_beat_freq <= 0:
                raise ValueError("Error: end_beat_freq must be greater than 0.")
            if self.base_freq <= 0:
                raise ValueError("Error: base_freq must be greater than 0.")
            if self.binaural_fade_out_duration < 0:
                raise ValueError("Error: binaural_fade_out_duration must be greater than or equal to 0.")
            binaural_segment = self.generate_binaural_beats(duration)
            if not self.power_ratio:
                self.power_ratio = 450000 # 300000 # increasing this reduces the binaural beat volume
            mixed_audio = self.adjust_power_overlay_and_normalise(input_audio, binaural_segment)
        else:
            my_dir = os.path.dirname(os.path.realpath(__file__))
            sounds_dir = os.path.join(my_dir, self.sounds_dir)
            if not os.path.exists(sounds_dir):
                raise FileNotFoundError(f"Error: {sounds_dir} does not exist.")
            ambient_files = os.listdir(sounds_dir)
            if not any(file.endswith(".mp3") for file in ambient_files):
                raise FileNotFoundError(f"Error: {sounds_dir} does not contain any MP3 files.")
            ambient_file = ""
            while not ambient_file.endswith(".mp3"):
                ambient_file = os.path.join(self.sounds_dir, random.choice(ambient_files))
                print(f"Selected ambient file: {ambient_file}")
                ambient_file = os.path.join(my_dir, ambient_file)
            mixed_audio = self.overlay_ambient(self.mp3_file, ambient_file)
        mixed_audio.export(self.output_file, format="mp3")
        return self.output_file
