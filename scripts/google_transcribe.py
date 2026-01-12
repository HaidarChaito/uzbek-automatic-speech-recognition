import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    MethodNotImplemented,
)
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm


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


def transcribe_batch_parallel(
    audio_paths: List[str], regions: List[str] = ["eu", "us"], max_workers: int = 10
) -> List[Dict[str, Any]]:
    """Transcribes multiple audio files in parallel using multiple regions.

    Args:
        audio_paths: List of paths to audio files
        regions: List of regions to distribute work across (default: ["eu", "us"] - seems only these are available)
        max_workers: Number of parallel workers (default: 10, recommended: 8, 10, 12 with 2 regions)
            Avoid rate limiter - 300 request per minute per region
            Learn more: https://docs.cloud.google.com/speech-to-text/docs/quotas

    Returns:
        List of dictionaries containing transcription results
    """
    results = []

    # Distribute files across regions in one by one alternatively (round-robin fashion)
    tasks = []
    for idx, audio_path in enumerate(audio_paths):
        region = regions[idx % len(regions)]
        tasks.append((audio_path, region))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(transcribe_with_retry, audio_path, region): (
                audio_path,
                region,
            )
            for audio_path, region in tasks
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_task), desc="Transcribing", unit="file") as pbar:
            for future in as_completed(future_to_task):
                audio_path, region = future_to_task[future]

                result = {
                    "path": audio_path,
                    "predicted_sentence": None,
                    "error_message": None,
                    "error_type": None,
                }

                try:
                    response = future.result()
                    if response and response.results:
                        result["predicted_sentence"] = (
                            response.results[0].alternatives[0].transcript
                        )
                except GoogleCloudProjectIdNotSet as err:
                    print(f"\nConfiguration error: {err}")
                    break
                except (FileNotFoundError, MethodNotImplemented) as err:
                    print(f"\nError: {type(err).__name__}: {err}")
                    break
                except Exception as err:
                    result["error_message"] = str(err)
                    result["error_type"] = type(err).__name__

                results.append(result)
                pbar.update(1)

    return results
