import ffmpeg
import os
from fractions import Fraction
import uuid

INPUT_DIR = "/mnt/e/star_wars_gan_data"
OUTPUT_DIR = "/mnt/e/star_wars_gan_data/frames"
START_CUT = 60  # seconds
END_CUT = 120  # seconds
FPS = 1

files = []
for file in os.listdir(INPUT_DIR):
    if file.endswith(".mp4"):
        files.append(f"{INPUT_DIR}/{file}")

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


class VideoData:
    def __init__(self, width, height, fps, frames):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = frames

    def get(file_name):
        probe = ffmpeg.probe(file_name)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        return VideoData(
            video_stream["width"],
            video_stream["height"],
            float(Fraction(video_stream["r_frame_rate"])),
            int(video_stream["nb_frames"]),
        )


def process(file_name):
    tag = os.path.splitext(os.path.basename(file_name))[0]
    tag = tag.replace(" ", "_")
    vd = VideoData.get(file_name)
    start_cut_frames = int(vd.fps * START_CUT)
    end_cut_frames = int(vd.fps * END_CUT)
    out_filename = f"{OUTPUT_DIR}/%d_{tag}.jpg"

    ffmpeg.input(file_name).trim(
        start_frame=start_cut_frames, end_frame=vd.frames - end_cut_frames
    ).filter("fps", fps="%.4f" % FPS).output(out_filename, **{"qscale:v": 2}).run(
        quiet=True
    )


def postprocess():
    for file in os.listdir(OUTPUT_DIR):
        h = uuid.uuid4().hex
        os.rename(f"{OUTPUT_DIR}/{file}", f"{OUTPUT_DIR}/{h}_{file}")


for f in files[::-1]:
    print(f"Processing {f}...")
    process(f)

print("Postprocessing...")
postprocess()
