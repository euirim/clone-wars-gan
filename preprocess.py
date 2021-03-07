from tqdm import tqdm
import ffmpeg
import os
from fractions import Fraction
import uuid

INPUT_DIR = "/home/euirim/Downloads/SWTCW"  # Input directory to read episodes from
OUTPUT_DIR = "./data/full"  # Output directory to write frames to
START_CUT = 20  # number of seconds to cut from a beginning of an episode
END_CUT = 60  # number of seconds to cut from the end of an episode
# Exceptions to cuts from the end for some episodes
END_EXCEPTIONS = {"S07e12": 165, "S07e07": 195}
START_EXCEPTIONS = {}  # Exceptions to cuts from the start for some episodes
FPS = 1  # Frames per second to sample videos from

# Recursively read .mp4 files from a directory (corresponding to episodes)
files = []
for root, _, filenames in os.walk(INPUT_DIR):
    for filename in filenames:
        if filename.endswith(".mp4"):
            files.append(os.path.join(root, filename))

# Create the output directory
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# Data structure to encapsulate some useful information about a video
class VideoData:
    def __init__(self, width, height, fps, frames):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = frames

    # Generate a `VideoData` object from a video file
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


# Preprocesses the video
def process(file_name, start=START_CUT, end=END_CUT):
    # Generate the `VideoData` object for the file name and the file name template for the output frames.
    tag = os.path.splitext(os.path.basename(file_name))[0]
    tag = tag.replace(" ", "_")
    vd = VideoData.get(file_name)
    start_cut_frames = int(vd.fps * start)
    end_cut_frames = int(vd.fps * end)
    out_filename = f"{OUTPUT_DIR}/%d_{tag}.jpg"

    # Trim the video to remove some seconds of video from the start and end.
    ffmpeg.input(file_name).trim(
        start_frame=start_cut_frames, end_frame=vd.frames - end_cut_frames
    ).filter("fps", fps="%.4f" % FPS).output(out_filename, **{"qscale:v": 2}).run(
        quiet=True
    )


# Randomize the filenames of the outputted frames
def postprocess():
    for file in tqdm(os.listdir(OUTPUT_DIR)):
        h = uuid.uuid4().hex
        os.rename(f"{OUTPUT_DIR}/{file}", f"{OUTPUT_DIR}/{h}_{file}")


for f in tqdm(files[::-1]):
    # Compute how much of the video to cut from the start and end
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
