import logging
import subprocess

logger = logging.getLogger(__name__)


def normalize_audio(input_path: str, output_path: str) -> bool:
    try:
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-af",
                "loudnorm=I=-16:LRA=11:TP=-2",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "5",
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return True
    except Exception as err:
        logger.exception(f"Error normalizing audio '{input_path}'", err)
        return False
