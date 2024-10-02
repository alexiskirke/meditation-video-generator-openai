import os
from pydub import AudioSegment
from typing import List
from pydub.silence import split_on_silence
import random


class MP3Merger:
    """
    MP3Merger class to merge multiple MP3 files in order into a single MP3 file of
    the specified duration, leaving silence between the segments.
    Initialize the MP3Merger with a list of MP3 files and a target duration.

    :param mp3_files: List of paths to the MP3 files to combine.
    :param duration: Total required duration of the output MP3 file in seconds.
    :param front_buffer: Duration of silence to add at the beginning of the output file in seconds.
    :param rear_buffer: Duration of silence to add at the end of the output file in seconds.
    :param balance_even: Balance factor for even voice segments (-1 to 1)
    :param balance_odd: Balance factor for odd voice segments (-1 to 1)
    """
    def __init__(self, mp3_files: List[str],
                 duration: float,
                 front_buffer: float = 3,
                 rear_buffer: float = 3,
                 balance_even: float = 0,
                 balance_odd: float = 0,
                 min_silence_len: int = 800,
                 silence_thresh: int = -50,
                 silence_add: int = 8000,
                 spread_out: bool = False
                 ):
        """
        MP3Merger class to merge multiple MP3 files in order into a single MP3 file of
        the specified duration, leaving silence between the segments.
        Initialize the MP3Merger with a list of MP3 files and a target duration.

        :param mp3_files: List of paths to the MP3 files.
        :param duration: Total duration of the output MP3 file in seconds.
        :param front_buffer: Duration of silence to add at the beginning of the output file in seconds.
        :param rear_buffer: Duration of silence to add at the end of the output file in seconds.
        :param fade_time: Duration of the fade-in and fade-out effects in seconds.
        :param balance_even: The balance for even segments (0.0 is centered, -1.0 is left, 1.0 is right).
        :param balance_odd: The balance for odd segments (0.0 is centered, -1.0 is left, 1.0 is right)
        :param min_silence_len: Minimum length of silence between words (ms).
        :param silence_thresh: Silence threshold in dB.
        :param silence_add: Amount of silence to add to the beginning and end of each segment (ms).
        :param spread_out: Boolean indicating if phrases should be spread out with silence.
        """

        self.duration = duration
        self.output_file = "merged_output.mp3"
        self.front_buffer = front_buffer * 1000  # Convert seconds to milliseconds
        self.rear_buffer = rear_buffer * 1000  # Convert seconds to milliseconds
        self.balance_even = balance_even
        self.balance_odd = balance_odd
        self.mp3_files = mp3_files
        self.silence_add = silence_add
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.spread_out = spread_out

    def merge(self) -> str:
        """
        Merges the input MP3 files into a single MP3 file of the specified duration.
        If the total duration of the input MP3 files is less than the target duration,
        silences are inserted between the MP3 files to achieve the target duration.
        """
        if len(self.mp3_files) == 1:
            raise ValueError("At least two MP3 files are required.")
        if self.mp3_files is None or len(self.mp3_files) == 0:
            raise ValueError("No MP3 files provided.")
        if not all(file.endswith(".mp3") for file in self.mp3_files):
            raise ValueError("All input files must be in MP3 format.")
        if not all(os.path.exists(file) for file in self.mp3_files):
            raise ValueError("All input files must exist.")
        if self.duration <= 0:
            raise ValueError("Duration must be greater than zero.")
        if self.front_buffer < 0:
            raise ValueError("Front buffer must be positive.")
        if self.rear_buffer < 0:
            raise ValueError("Rear buffer must be positive.")
        if self.balance_even < -1 or self.balance_even > 1:
            raise ValueError("Balance for even segments must be between -1 and 1.")
        if self.balance_odd < -1 or self.balance_odd > 1:
            raise ValueError("Balance for odd segments must be between -1 and 1.")
        # Adds silences into spoken word files if spread_out is True
        if self.spread_out:
            self.spread_out_all_files()
        # Load all MP3 files and calculate their total duration
        segments = [AudioSegment.from_mp3(file) for file in self.mp3_files]
        total_duration = sum(segment.duration_seconds for segment in segments)
        # Ensure the target duration is at least as large as the total duration of all MP3 files
        if total_duration > self.duration:
            raise ValueError(
                f"Total duration of segments: {total_duration} seconds, target duration: {self.duration} seconds. Total duration of input MP3 files exceeds the target duration.")
        # Calculate the total silence duration needed
        total_silence_duration = self.duration - total_duration
        # not len(segments) + 1
        silence_duration = total_silence_duration / len(segments) if len(segments) > 1 else total_silence_duration
        silence_segment = AudioSegment.silent(duration=silence_duration * 1000)  # Convert seconds to milliseconds
        # Apply stereo balance to each segment
        for i, segment in enumerate(segments):
            if i % 2 == 0:
                segments[i] = segment.pan(self.balance_even)  # Apply even balance
            else:
                segments[i] = segment.pan(self.balance_odd)  # Apply odd balance
        # Create the merged audio segment
        merged_segment = AudioSegment.silent(duration=self.front_buffer) + segments[0]
        for segment in segments[1:]:
            merged_segment += silence_segment + segment
        merged_segment += AudioSegment.silent(duration=silence_duration * 1000) + AudioSegment.silent(
            duration=self.rear_buffer)
        # Export the final merged audio segment to an MP3 file
        merged_segment.export(self.output_file, format="mp3")
        return self.output_file

    def spread_out_phrases(self, filename: str) -> tuple[int, int]:
        """
        Loads an MP3 file and spreads out the phrases by adding silence between them.

        :param filename: Path to the input MP3 file.
        :return: Tuple containing the old and new file lengths in milliseconds.
        """
        # Load the audio file
        audio = AudioSegment.from_mp3(filename)
        old_file_length = len(audio)

        # Split the audio into chunks (words) based on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,  # Minimum length of silence between words (ms)
            silence_thresh=self.silence_thresh  # Consider anything quieter than -50 dBFS as silence
        )
        # Create a new audio segment with silence
        random_dev = self.silence_add // 3  # Add some randomness to the silence duration
        # Spread out the words by adding silence between them
        spread_audio = chunks[0]
        for chunk in chunks[1:]:
            silence = AudioSegment.silent(
                duration=self.silence_add -
                         random_dev + random.randint(0, random_dev * 2))  # Add some randomness to the silence duration
            spread_audio += silence + chunk
        # Rename filename with _original added
        base_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = f"{base_name}_original.mp3"
        # Rename the original file
        os.rename(filename, new_filename)
        new_file_length = len(spread_audio)
        # Export the new audio file
        spread_audio.export(filename, format="mp3")
        return old_file_length, new_file_length

    def spread_out_all_files(self):
        """
        Spreads out the phrases in all MP3 files.
        """
        for file in self.mp3_files:
            self.spread_out_phrases(file)
