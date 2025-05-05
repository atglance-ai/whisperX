import os
import warnings
import time
from typing import List, Union, Optional, NamedTuple

import ctranslate2
import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator
import copy
from collections import Counter, defaultdict

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .vad import load_vad_model, merge_chunks
from .types import TranscriptionResult, SingleSegment

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(self, features: np.ndarray, tokenizer: faster_whisper.tokenizer.Tokenizer, options: faster_whisper.transcribe.TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        results = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                beam_size=options.beam_size,
                return_scores=True,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
            )

        # tokens_batch = [x.sequences_ids[0] for x in result]
        avg_logprobs = []
        tokens_batch = []
        for result in results:
            tokens = result.sequences_ids[0]
            tokens_batch.append(tokens)

            # Calculate average log probability.
            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)
            avg_logprobs.append(avg_logprob)

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return {'text': text, 'avg_logprob': avg_logprobs}

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            model,
            vad,
            vad_params: dict,
            options : NamedTuple,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            language : Optional[str] = None,
            suppress_numerals: bool = False,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        # Language detection
        self.default_language = 'sv'
        self.language_probability_threshold = 0.95
        self.language_of_segment = None

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        # outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        # return {'text': outputs}
        model_outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return model_outputs

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, task=None, chunk_size=30, print_progress = False, combined_progress=False
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                self.model.model.is_multilingual, task=task,
                                                                language=language)
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                    self.model.model.is_multilingual, task=task,
                                                                    language=language)
                
        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            print(f"Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = self.options._replace(suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            avg_logprob = out['avg_logprob']
            if batch_size in [0, 1, None]:
                text = text[0]
                avg_logprob = avg_logprob[0]
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3),
                    "avg_logprob": avg_logprob,
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = self.options._replace(suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language}
    
    def transcribe_multilang(
        self,
        audio: Union[str, np.ndarray],
        *,
        batch_size: int | None = None,
        num_workers: int = 0,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        # ---- tuning knobs ------------------------------------------------------
        lang_prob_threshold: float = 0.95,
        min_seg_for_lang: float = 10.0,
        max_language_groups: int = 5,
        fallback_language: str | None = None,
        verbose: bool = False,
    ):
        """
        Multilingual transcription with per‑language batching.
        Returns dict{"segments", "language", "languages"}  (no 'text').
        """
        # ------------------------------ I/O & VAD --------------------------------
        if isinstance(audio, str):
            audio = load_audio(audio)

        vad_segments = self.vad_model({
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": SAMPLE_RATE
        })
        vad_segments = merge_chunks(vad_segments,
                                    chunk_size,
                                    onset=self._vad_params["vad_onset"],
                                    offset=self._vad_params["vad_offset"])

        if not vad_segments:
            return {"segments": [], "language": None, "languages": []}

        # ---------------------- language detection / tagging ---------------------
        confident: list[dict] = []
        uncertain: list[dict] = []
        if verbose:
            print(f"Detecting language for {len(vad_segments)} segments...")

        for seg in vad_segments:
            seg_len = seg['end'] - seg['start']
            seg["length"] = seg_len

            # skip detection for short segments – automatically uncertain
            if seg_len < min_seg_for_lang:
                seg["language_prob"] = 0.0
                seg["language"] = None
                uncertain.append(seg)
                continue

            lang, prob = self.detect_segment_language(audio,
                                                    seg['start'],
                                                    seg['end'],
                                                    SAMPLE_RATE,
                                                    return_all=False)

            seg["language"] = lang
            seg["language_prob"] = prob

            if prob >= lang_prob_threshold:
                confident.append(seg)
            else:
                uncertain.append(seg)

        # ------------------- if nothing confident → fallback ---------------------
        if not confident:
            if verbose:
                print("No confident language segments; using plain transcribe().")
            res = self.transcribe(audio,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                chunk_size=chunk_size,
                                print_progress=print_progress,
                                combined_progress=combined_progress,
                                language=fallback_language)
            res["languages"] = [res["language"]]
            return res

        # -------------------- select top‑N language groups -----------------------
        lang_counts = Counter(s["language"] for s in confident)
        lang_pools = [l for l, _ in lang_counts.most_common(max_language_groups)]
        majority_lang = lang_pools[0]
        if verbose:
            counts_per_lang = {lang: lang_counts[lang] for lang in lang_pools}
            print(f"Language pools: {counts_per_lang}")

        # -------------- build pools, map rare langs to majority ------------------
        segments_by_lang: dict[str, list] = defaultdict(list)

        for seg in confident:
            key = seg["language"] if seg["language"] in lang_pools else majority_lang
            segments_by_lang[key].append(seg)

        # duplicate every uncertain segment into *all* pools
        for seg in uncertain:
            for l in lang_pools:
                segments_by_lang[l].append(copy.deepcopy(seg))

        # --------------------- per‑language transcription loop -------------------
        base_tokenizer = self.tokenizer
        base_options = self.options
        results_by_start: dict[float, tuple] = {}  # start_time → (segment, logp)

        for lang, segs in segments_by_lang.items():
            if verbose:
                print(f"Transcribing {lang}...")
            tokenizer = faster_whisper.tokenizer.Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task="transcribe",
                language=lang,
            )
            opts = copy.deepcopy(base_options)
            if self.suppress_numerals:
                opts = opts._replace(
                    suppress_tokens=list(set(opts.suppress_tokens +
                                            find_numeral_symbol_tokens(tokenizer)))
                )

            self.tokenizer = tokenizer
            self.options   = opts

            def seg_iter():
                for s in segs:
                    f1 = int(s['start'] * SAMPLE_RATE)
                    f2 = int(s['end'] * SAMPLE_RATE)
                    yield {'inputs': audio[f1:f2]}

            bs = batch_size or self._batch_size
            total = len(segs)

            for idx, out in enumerate(self.__call__(seg_iter(),
                                                    batch_size=bs,
                                                    num_workers=num_workers)):
                if print_progress:
                    base_p = ((idx + 1) / total) * 100
                    pct = base_p / 2 if combined_progress else base_p
                    print(f"[{lang}] {pct:.2f}%")

                text        = out['text'][0] if bs in [0, 1, None] else out['text']
                avg_logprob = out['avg_logprob'][0] if bs in [0, 1, None] else out['avg_logprob']

                seg = segs[idx]             # iterator order ≡ seg list order
                seg_out = {
                    "text": text,
                    "start": round(seg['start'], 3),
                    "end":   round(seg['end'], 3),
                    "avg_logprob": avg_logprob,
                    "language": lang
                }

                key = seg_out["start"]
                if key not in results_by_start or avg_logprob > results_by_start[key][1]:
                    results_by_start[key] = (seg_out, avg_logprob)

            self.tokenizer = base_tokenizer
            self.options   = base_options

        # ----------------------------- final merge -------------------------------
        final_segments = [v[0] for v in sorted(results_by_start.values(),
                                            key=lambda x: x[0]["start"])]

        return {
            "segments":  final_segments,
            "language":  majority_lang,
            "languages": sorted(set(lang_pools)),
        }


    def detect_language(self, audio: np.ndarray):
        start_time = time.time()
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate. [language detection]")
        
        language_of_segment = []

        total_segments = audio.shape[0] // N_SAMPLES
        # Determine the number of segments to check based on the given criteria
        if total_segments < 10:
            num_segments = total_segments
        else:
            # Use either the first 50 segments or the first third of the segments, whichever is smaller
            num_segments = min(50, max(10, total_segments // 3))
        
        model_n_mels = self.model.feat_kwargs.get("feature_size")

        for i in range(num_segments):
            segment = log_mel_spectrogram(audio[i * N_SAMPLES: (i + 1) * N_SAMPLES],
                                          n_mels=model_n_mels if model_n_mels is not None else 80,
                                          padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
            encoder_output = self.model.encode(segment)
            results = self.model.model.detect_language(encoder_output)
            language_token, language_probability = results[0][0]
            language = language_token[2:-2]
            language_of_segment.append((language, language_probability))
        
        if not language_of_segment:
            print(
                f"Warning: No language detected from the audio {num_segments} segments. [language detection]"
                f"Returning default language {self.default_language}. {language_of_segment} [language detection]"
            )
            return self.default_language
        elif all(prob < self.language_probability_threshold for lang, prob in language_of_segment):
            print(
                f"Warning: No language detected above threshold for {num_segments} segments. [language detection] "
                f"Returning language of first segment {language_of_segment[0][0]}. {language_of_segment} [language detection]"
            )
            return language_of_segment[0][0]
        
        self.language_of_segment = language_of_segment
        
        # Determine the most common language across all checked segments
        language_counts = {}
        for language, language_probability in language_of_segment:
            if language_probability < self.language_probability_threshold:
                continue
            if language in language_counts:
                language_counts[language] += 1
            else:
                language_counts[language] = 1

        sorted_languages = sorted(language_counts.items(), key=lambda item: item[1], reverse=True)
        most_common_language = sorted_languages[0][0]
        num_detected = sorted_languages[0][1]
        
        # Determine the second and third most common languages if available
        second_most_common_language = sorted_languages[1] if len(sorted_languages) > 1 else (None, 0)
        
        # Print the results of the language detection process
        print(
            f"Using most common language: {most_common_language}, detected in {num_detected}/{num_segments} segments of audio. [language detection]\n"
            f"Second most common: {second_most_common_language[0]}, detected in {second_most_common_language[1]}/{num_segments} segments. [language detection]\n"
            f"Total Inference time: {time.time() - start_time:.2f}s. [language detection]\n"
            f"Detected languages and probabilities per segment: {language_of_segment} [language detection]"
        )
        
        
        # Return the most common language detected
        return most_common_language
    
    def detect_segment_language(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
        sample_rate: int,
        return_all: bool = False,
    ):
        """
        Detect the language spoken between `start_time` and `end_time` (in seconds).

        Parameters
        ----------
        audio : np.ndarray
            1-D PCM float32 array (full file).
        start_time, end_time : float
            Segment boundaries in seconds.
        sample_rate : int
            Sample-rate of `audio`.
        return_all : bool, optional
            • False  → return (best_lang, best_prob)  
            • True   → return list[(lang, prob), …] sorted by `prob` desc.

        Returns
        -------
        best_lang : str
            ISO-639-1 language code (e.g. "en", "sv").
        best_prob : float
            Probability of `best_lang`.  
            **OR**, if `return_all=True`, a full ranked list.
        """
        # ---- 1. slice & pad / truncate to 30s window ----
        start_idx = int(start_time * sample_rate)
        end_idx   = int(end_time   * sample_rate)
        segment   = audio[start_idx:end_idx]

        if segment.size == 0:
            raise ValueError("Empty segment passed to detect_segment_language")

        # if segment.shape[0] < N_SAMPLES:
        #     # pad to 30 s so mel extractor sees the expected number of frames
        #     segment = np.pad(segment, (0, N_SAMPLES - segment.shape[0]))

        # ---- 2. log‑mel & encoder ----
        n_mels = self.model.feat_kwargs.get("feature_size") or 80
        mel    = log_mel_spectrogram(segment, n_mels=n_mels, padding=0)
        enc    = self.model.encode(mel)

        # ---- 3. language probs ----
        lang_scores = self.model.model.detect_language(enc)[0]  # list[(token, prob)]
        # convert tokens like "<|en|>" → "en"
        lang_scores = [(tok[2:-2], prob) for tok, prob in lang_scores]

        if return_all:
            return lang_scores      # already sorted by prob desc
        else:
            best_lang, best_prob = lang_scores[0]
            return best_lang, best_prob


def load_model(whisper_arch,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               language : Optional[str] = None,
               vad_model=None,
               vad_options=None,
               model : Optional[WhisperModel] = None,
               task="transcribe",
               download_root=None,
               threads=4):
    '''Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        model: Optional[WhisperModel] - The WhisperModel instance to use.
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    '''
    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         cpu_threads=threads)
    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        vad_model = vad_model
    else:
        vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
