from tqdm import tqdm
import ffmpeg
import os
from fractions import Fraction
import uuid

INPUT_DIR = "/home/euirim/Downloads/SWTCW"
OUTPUT_DIR = "./data/full"
START_CUT = 20  # seconds
END_CUT = 60  # seconds
END_EXCEPTIONS = {"S07e12": 165, "S07e07": 195}
START_EXCEPTIONS = {}
FPS = 1

files = []
for root, _, filenames in os.walk(INPUT_DIR):
    for filename in filenames:
        if filename.endswith(".mp4"):
            files.append(os.path.join(root, filename))

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


def process(file_name, start=START_CUT, end=END_CUT):
    tag = os.path.splitext(os.path.basename(file_name))[0]
    tag = tag.replace(" ", "_")
    vd = VideoData.get(file_name)
    start_cut_frames = int(vd.fps * start)
    end_cut_frames = int(vd.fps * end)
    out_filename = f"{OUTPUT_DIR}/%d_{tag}.jpg"

    ffmpeg.input(file_name).trim(
        start_frame=start_cut_frames, end_frame=vd.frames - end_cut_frames
    ).filter("fps", fps="%.4f" % FPS).output(out_filename, **{"qscale:v": 2}).run(
        quiet=True
    )


def postprocess():
    for file in tqdm(os.listdir(OUTPUT_DIR)):
        h = uuid.uuid4().hex
        os.rename(f"{OUTPUT_DIR}/{file}", f"{OUTPUT_DIR}/{h}_{file}")


for f in tqdm(files[::-1]):
    start, end = START_CUT, END_CUT
    for exc, tmp in END_EXCEPTIONS.items():
        if exc.lower() in f.lower():
            end = tmp
            break
    for exc, tmp in START_EXCEPTIONS.items():
        if exc.lower() in f.lower():
            start = tmp
            break

    process(f, start=start, end=end)

print("Postprocessing...")
postprocess()
