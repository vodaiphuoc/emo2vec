import subprocess
def is_ffmpeg_installed():
    try:
        output = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        return "ffmpeg version" in output.decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


use_ffmpeg = False
if is_ffmpeg_installed():
    use_ffmpeg = True
else:
    print(
        "Notice: ffmpeg is not installed. torchaudio is used to load audio\n"
        "If you want to use ffmpeg backend to load audio, please install it by:"
        "\n\tsudo apt install ffmpeg # ubuntu"
        "\n\t# brew install ffmpeg # mac"
    )



try:
    from pydub import AudioSegment
except:
    raise ImportError("cannot import AudioSegment from pydub")


from .emotion2vec_ft import E2VftModel

from .load_utils import (
    download_repo_from_hf,
    load_pretrained_model,
    get_pretrain_config
)