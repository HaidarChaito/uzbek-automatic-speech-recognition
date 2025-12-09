import os

from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class GoogleCloudProjectIdNotSet(Exception):
    """Raised when the Google Cloud Project ID is not set in the environment."""

    pass


def get_google_project_id():
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise GoogleCloudProjectIdNotSet(
            "Environment variable 'GOOGLE_CLOUD_PROJECT' is not set."
        )
    return project_id


def transcribe(audio_path: str, region: str = "eu") -> cloud_speech.RecognizeResponse:
    """Transcribes an audio file using Google Speech-to-Text V2.
    Args:
        audio_path (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
        region (str): Server region to be requested to
            Avoid rate limiter - 300 request per minute per region
            Learn more: https://docs.cloud.google.com/speech-to-text/docs/quotas
    Returns:
        cloud_speech.RecognizeResponse: The response from the Speech-to-Text API containing
        the transcription results.
    """
    project_id = get_google_project_id()

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{region}-speech.googleapis.com",
        )
    )

    # Reads a file as bytes
    with open(audio_path, "rb") as f:
        audio_content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["uz-UZ"],
        model="chirp_3",  # by default punctuation is enabled
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{region}/recognizers/_",
        config=config,
        content=audio_content,
    )

    # Transcribes the audio into text
    response = client.recognize(request=request)

    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
)
def transcribe_with_retry(audio_path: str, region: str = "eu"):
    return transcribe(audio_path, region)
