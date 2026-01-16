import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable

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


def get_google_project_ids():
    """Get list of Google Cloud project IDs from environment variables.

    Supports multiple projects via:
    - GOOGLE_CLOUD_PROJECTS: comma-separated list (e.g., "project1,project2,project3")
    - GOOGLE_CLOUD_PROJECT: single project (backward compatible)

    Returns:
        List[str]: List of project IDs
    """
    load_dotenv()

    # Try to get multiple projects first
    projects_str = os.getenv("GOOGLE_CLOUD_PROJECTS")
    if projects_str:
        project_ids = [p.strip() for p in projects_str.split(",") if p.strip()]
        if project_ids:
            return project_ids

    # Fall back to single project (backward compatible)
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return [project_id]

    raise GoogleCloudProjectIdNotSet(
        "Environment variable 'GOOGLE_CLOUD_PROJECTS' or 'GOOGLE_CLOUD_PROJECT' is not set."
    )


def transcribe(
    audio_path: str, region: str = "eu", project_id: str = None
) -> cloud_speech.RecognizeResponse:
    """Transcribes an audio file using Google Speech-to-Text V2.

    Args:
        audio_path (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
        region (str): Server region to be requested to
            Avoid rate limiter - 300 request per minute per region
            Learn more: https://docs.cloud.google.com/speech-to-text/docs/quotas
        project_id (str): Google Cloud project ID. If None, will use default from env.

    Returns:
        cloud_speech.RecognizeResponse: The response from the Speech-to-Text API containing
        the transcription results.
    """
    if project_id is None:
        project_id = get_google_project_ids()[0]

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
def transcribe_with_retry(audio_path: str, region: str = "eu", project_id: str = None):
    return transcribe(audio_path, region, project_id)


def transcribe_batch_parallel(
    audio_paths: List[str],
    regions: List[str] = ["eu", "us"],
    max_workers: int = 10,
    checkpoint_callback: Optional[Callable[[List[Dict]], None]] = None,
    checkpoint_interval: int = 100,
) -> List[Dict[str, Any]]:
    """Transcribes multiple audio files in parallel using multiple regions and projects.

    Args:
        audio_paths: List of paths to audio files
        regions: List of regions to distribute work across (default: ["eu", "us"])
        max_workers: Number of parallel workers (default: 10)
            With multiple projects: can increase proportionally
            - 2 regions × 1 project = 10-12 workers
            - 2 regions × 2 projects = 18-20 workers (Optimal)
            - 2 regions × 3 projects = 26-30 workers

            Note: 2 projects is the "sweet spot." Using 3+ projects often hits
            diminishing returns due to server-side latency.

            Rate limit: 300 requests per minute per region per project
            Learn more: https://docs.cloud.google.com/speech-to-text/docs/quotas
        checkpoint_callback: Optional callback function that receives the current results list
                           Called every checkpoint_interval files
        checkpoint_interval: How often to call checkpoint_callback (default: 50 files)

    Returns:
        List of dictionaries containing transcription results
    """
    # Get available projects
    project_ids = get_google_project_ids()
    print(f"Using {len(project_ids)} project(s) across {len(regions)} region(s)")
    print(f"Total combinations: {len(project_ids) * len(regions)}")

    results = []
    files_since_checkpoint = 0

    # Distribute files across projects and regions in round-robin fashion
    # This ensures even distribution across all project-region combinations
    tasks = []
    project_region_combinations = [
        (project, region) for project in project_ids for region in regions
    ]

    for idx, audio_path in enumerate(audio_paths):
        project_id, region = project_region_combinations[
            idx % len(project_region_combinations)
        ]
        tasks.append((audio_path, region, project_id))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(transcribe_with_retry, audio_path, region, project_id): (
                audio_path,
                region,
                project_id,
            )
            for audio_path, region, project_id in tasks
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_task), desc="Transcribing", unit="file") as pbar:
            for future in as_completed(future_to_task):
                audio_path, region, project_id = future_to_task[future]

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
                files_since_checkpoint += 1
                pbar.update(1)

                # Checkpoint saving
                if (
                    checkpoint_callback is not None
                    and files_since_checkpoint >= checkpoint_interval
                ):
                    try:
                        checkpoint_callback(results)
                        files_since_checkpoint = 0
                    except Exception as checkpoint_err:
                        print(f"\n⚠️  Warning: Checkpoint save failed: {checkpoint_err}")
                        print("Continuing transcription...")

    # Final checkpoint save if callback is provided
    if checkpoint_callback is not None and files_since_checkpoint > 0:
        try:
            checkpoint_callback(results)
        except Exception as checkpoint_err:
            print(f"\n⚠️  Warning: Final checkpoint save failed: {checkpoint_err}")

    return results
