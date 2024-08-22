import os
from moviepy.editor import VideoFileClip

def extract_audio_clips(video_path, output_dir, clip_duration=0.5):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio

        audio_clip_duration = int(audio_clip.duration)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        i = 0
        while i < audio_clip_duration:
            start_time = i
            end_time = i + clip_duration
            subclip = audio_clip.subclip(start_time, end_time)

            clip_filename = f"{os.path.basename(video_path).split('.')[0]}_{int(i*2):04d}.wav"
            clip_filepath = os.path.join(output_dir, clip_filename)

            subclip.write_audiofile(clip_filepath, codec='pcm_s16le')
            i += clip_duration

        video_clip.reader.close()
        audio_clip.reader.close_proc()
    except IOError as e:
        print(f"An error occurred: {e}")