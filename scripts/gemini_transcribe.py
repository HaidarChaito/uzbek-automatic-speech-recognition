import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
)
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm


class GoogleAPIKeyNotSet(Exception):
    """Raised when the Google API Key is not set in the environment."""

    pass


class RateLimiter:
    """Thread-safe rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 150):
        self.rpm = requests_per_minute
        self.window_seconds = 60.0
        self.lock = threading.Lock()
        self.request_times = []

    def acquire(self):
        """Wait if necessary to respect rate limit."""
        while True:
            with self.lock:
                current_time = time.time()

                # Remove requests older than the window
                cutoff = current_time - self.window_seconds
                self.request_times = [t for t in self.request_times if t > cutoff]

                # If under limit, record this request and proceed
                if len(self.request_times) < self.rpm:
                    self.request_times.append(current_time)
                    return

                # Calculate how long to wait for the oldest request to expire
                oldest = min(self.request_times)
                wait_time = oldest + self.window_seconds - current_time + 0.05

            # Wait outside the lock
            if wait_time > 0:
                time.sleep(wait_time)


class PerKeyRateLimiter:
    """Manages separate rate limiters for each API key."""

    def __init__(self, api_keys: List[str], requests_per_minute_per_key: int = 150):
        self.limiters = {
            key: RateLimiter(requests_per_minute_per_key) for key in api_keys
        }

    def acquire(self, api_key: str):
        """Acquire rate limit slot for specific API key."""
        self.limiters[api_key].acquire()


def get_google_api_keys() -> List[str]:
    """Get list of Google API keys from environment variables.

    Supports multiple API keys via:
    - GOOGLE_API_KEYS: comma-separated list (e.g., "key1,key2,key3")
    - GOOGLE_API_KEY: single key (backward compatible)

    Returns:
        List[str]: List of API keys
    """
    load_dotenv()

    # Try to get multiple API keys first
    keys_str = os.getenv("GOOGLE_API_KEYS")
    if keys_str:
        api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if api_keys:
            return api_keys

    # Fall back to single API key (backward compatible)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return [api_key]

    raise GoogleAPIKeyNotSet(
        "Environment variable 'GOOGLE_API_KEYS' or 'GOOGLE_API_KEY' is not set. "
        "Get your API key from https://aistudio.google.com/apikey"
    )


def transcribe(
    audio_path: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    language: str = "uz-UZ",
    rate_limiter: PerKeyRateLimiter = None,
) -> Dict[str, Any]:
    """Transcribes an audio file using Google AI Studio (Gemini API).

    Args:
        audio_path (str): Path to the local audio file to be transcribed.
            Supported formats: WAV, MP3, AIFF, AAC, OGG, FLAC
        api_key (str): Google API key.
        model_name (str): Gemini model to use.
        language (str): Language code (e.g., "uz-UZ", "en-US")
        rate_limiter (PerKeyRateLimiter): Optional rate limiter instance

    Returns:
        Dict[str, Any]: Dictionary containing:
            - transcript: The transcribed text
            - model: Model used
            - language: Language code used
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Apply rate limiting for this specific API key
    if rate_limiter:
        rate_limiter.acquire(api_key)

    # Create Google AI Studio client
    client = genai.Client(api_key=api_key)

    # Read audio file as bytes
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Determine MIME type based on file extension
    ext = Path(audio_path).suffix.lower()
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".aiff": "audio/aiff",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
    }
    mime_type = mime_types.get(ext, "audio/wav")

    # Create Part object with inline audio data
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

    # Create the prompt for transcription
    prompt = """Transcribe this audio content into Uzbek language using Latin characters only. 

    Maintain the natural speech patterns including:
    - Keep Russian and English words exactly as spoken, don't translate them
    - Conversational grammar (even if imperfect)
    - Incomplete sentences or speech corrections
    - Colloquial expressions

    After completing the transcription:
    1. Validate for any transcription errors
    2. Fix obvious mistranslations while preserving the original speech style
    3. Ensure consistency with terminology and names used in the previous segment
    4. Return ONLY the final Uzbek transcription using Latin characters
    5. If it mostly contains music, or it's a sung song, return an empty string.

    Note: This is for ASR train data
    """

    # Generate the transcription
    response = client.models.generate_content(
        model=model_name, contents=[prompt, audio_part]
    )

    return {
        "transcript": response.text.strip(),
        "model": model_name,
        "language": language,
    }


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
)
def transcribe_with_retry(
    audio_path: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    language: str = "uz-UZ",
    rate_limiter: PerKeyRateLimiter = None,
):
    """Transcribe with automatic retry on rate limits or service issues."""
    return transcribe(audio_path, api_key, model_name, language, rate_limiter)


def transcribe_batch_parallel(
    audio_paths: List[str],
    model_name: str = "gemini-2.0-flash",
    language: str = "uz-UZ",
    max_workers: int = None,
    requests_per_minute_per_key: int = 150,
    checkpoint_callback: Optional[Callable[[List[Dict]], None]] = None,
    checkpoint_interval: int = 100,
) -> List[Dict[str, Any]]:
    """Transcribes multiple audio files in parallel using Google AI Studio.

    Args:
        audio_paths: List of paths to audio files
        model_name: Gemini model to use (default: "gemini-2.0-flash")
        language: Language code (default: "uz-UZ")
        max_workers: Number of parallel workers. If None, auto-calculated based on
                    number of API keys and expected request duration.
                    Rule of thumb: (num_keys * rpm_per_key / 60) * avg_request_seconds * 1.5
        requests_per_minute_per_key: Rate limit per API key (default: 150)
        checkpoint_callback: Optional callback for saving progress
        checkpoint_interval: How often to call checkpoint_callback (default: 100 files)

    Returns:
        List of dictionaries containing transcription results
    """
    # Get available API keys
    api_keys = get_google_api_keys()
    num_keys = len(api_keys)

    # Calculate effective rate limit
    total_rpm = requests_per_minute_per_key * num_keys

    # Auto-calculate workers if not specified
    # Assuming ~2-3 seconds per request, we need enough workers to keep the pipeline full
    if max_workers is None:
        # Target: enough workers to saturate the rate limit
        # With 2.5s avg request time and X RPM, we need X/60 * 2.5 workers minimum
        # Add 50% buffer for variance
        avg_request_time = 2.5  # seconds
        max_workers = max(10, int((total_rpm / 60) * avg_request_time * 1.5))
        # Cap at reasonable maximum
        max_workers = min(max_workers, 100)

    print(f"=" * 60)
    print(f"Google AI Studio Batch Transcription")
    print(f"=" * 60)
    print(f"API keys: {num_keys}")
    print(f"Rate limit per key: {requests_per_minute_per_key} RPM")
    print(f"Total effective rate: {total_rpm} RPM (~{total_rpm/60:.1f} req/sec)")
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    print(f"Workers: {max_workers}")
    print(f"Files to process: {len(audio_paths)}")

    # Estimate time
    estimated_minutes = len(audio_paths) / total_rpm
    print(
        f"Estimated time: {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)"
    )
    print(f"=" * 60)

    # Create per-key rate limiter
    rate_limiter = PerKeyRateLimiter(api_keys, requests_per_minute_per_key)

    results = []
    files_since_checkpoint = 0

    # Distribute files across API keys in round-robin fashion
    tasks = []
    for idx, audio_path in enumerate(audio_paths):
        api_key = api_keys[idx % num_keys]
        api_key_index = idx % num_keys
        tasks.append((audio_path, api_key, api_key_index))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                transcribe_with_retry,
                audio_path,
                api_key,
                model_name,
                language,
                rate_limiter,
            ): (audio_path, api_key_index)
            for audio_path, api_key, api_key_index in tasks
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_task), desc="Transcribing", unit="file") as pbar:
            for future in as_completed(future_to_task):
                audio_path, api_key_index = future_to_task[future]

                result = {
                    "path": audio_path,
                    "predicted_sentence": None,
                    "model": model_name,
                    "language": language,
                    "api_key_index": api_key_index,
                    "error_message": None,
                    "error_type": None,
                }

                try:
                    response = future.result()
                    if response and "transcript" in response:
                        result["predicted_sentence"] = response["transcript"]
                except GoogleAPIKeyNotSet as err:
                    print(f"\nConfiguration error: {err}")
                    break
                except FileNotFoundError as err:
                    result["error_message"] = str(err)
                    result["error_type"] = type(err).__name__
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

    # Final checkpoint save
    if checkpoint_callback is not None and files_since_checkpoint > 0:
        try:
            checkpoint_callback(results)
        except Exception as checkpoint_err:
            print(f"\n⚠️  Warning: Final checkpoint save failed: {checkpoint_err}")

    return results


# Example usage
if __name__ == "__main__":
    DATASET_DIR = "../datasets/news_youtube_uzbek_speech_dataset/data"

    sampled2_path = os.path.join(DATASET_DIR, "sampled2.csv")
    sampled2_df = pd.read_csv(sampled2_path, index_col="id")

    # 1. Extract the ID (part before the underscore)
    prefix = sampled2_df["path"].str.split("_", n=1).str[0]

    # 2. Identify all rows where the prefix appears more than once
    is_duplicate = prefix.duplicated(keep=False)

    # 3. Filter the original dataframe
    chunked_df = sampled2_df[is_duplicate]
    chunked_df = chunked_df[chunked_df["path"].str.len() > 0]

    # Configuration
    CHECKPOINT_INTERVAL = 250
    CHECKPOINT_PATH = "transcription_checkpoint.csv"
    chunked_audio_dir = os.path.join(DATASET_DIR, "chunked_audio")
    transcription_path = os.path.join(
        DATASET_DIR, "google_ai_studio_transcriptions.csv"
    )

    audio_paths = (
        chunked_df["path"]
        .apply(lambda filename: os.path.join(chunked_audio_dir, filename))
        .tolist()
    )

    # Check for existing checkpoint
    existing_results = []
    processed_paths = set()

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found existing checkpoint: {CHECKPOINT_PATH}")
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH, index_col="id")
        checkpoint_df = checkpoint_df[checkpoint_df["error_message"].isna()]
        existing_results = checkpoint_df.to_dict("records")
        processed_paths = set(checkpoint_df["path"].apply(lambda x: Path(x).name))
        print(f"Resuming with {len(existing_results)} already processed")

        audio_paths = [
            path for path in audio_paths if Path(path).name not in processed_paths
        ]
        print(f"Remaining files: {len(audio_paths)}")

    def save_checkpoint(results_list, is_final=False):
        """Save current results to checkpoint file"""
        df = pd.DataFrame(results_list)
        df.to_csv(CHECKPOINT_PATH, index_label="id")
        status = "FINAL" if is_final else "CHECKPOINT"
        print(f"\n[{status}] Saved {len(results_list)} results")

    try:
        all_results = existing_results.copy()

        if audio_paths:
            results = transcribe_batch_parallel(
                audio_paths=audio_paths,
                model_name="gemini-2.5-pro",
                language="uz-UZ",
                max_workers=50,  # Increased significantly
                requests_per_minute_per_key=150,
                checkpoint_callback=save_checkpoint,
                checkpoint_interval=CHECKPOINT_INTERVAL,
            )
            all_results.extend(results)

        save_checkpoint(all_results, is_final=True)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted!")
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
        raise
    except Exception as err:
        print(f"\n❌ Error: {type(err).__name__}: {err}")
        raise

    # Process final results
    predicted_transcriptions = []
    failed_count = 0
    success_count = 0

    for result in all_results:
        transcription_entry = {
            "path": Path(result["path"]).name,
            "predicted_sentence": result["predicted_sentence"],
            "error_message": result.get("error_message"),
            "error_type": result.get("error_type"),
        }
        predicted_transcriptions.append(transcription_entry)

        if result["predicted_sentence"] is not None:
            success_count += 1
        else:
            failed_count += 1

    predicted_transcripts_df = pd.DataFrame(predicted_transcriptions)
    predicted_transcripts_df.to_csv(transcription_path, index_label="id")
    print(f"\n✓ Saved to: {transcription_path}")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\n{'=' * 60}")
    print(f"Complete! ✓ {success_count} | ✗ {failed_count}")
    print(f"Success rate: {success_count / len(all_results) * 100:.1f}%")
