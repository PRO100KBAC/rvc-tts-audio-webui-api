import asyncio
import os
import time
import edge_tts
import gradio as gr
import librosa
import torch
import yt_dlp
import soundfile as sf
import subprocess
import multiprocessing
import numba as nb
import numpy as np
if __name__ == '__main__':
    from pydub import AudioSegment
    from spleeter.separator import Separator
    from fairseq import checkpoint_utils
    from config import Config
    from lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )
    from rmvpe import RMVPE
    from vc_infer_pipeline import VC
    separator = Separator('spleeter:2stems')
    limitation = os.getenv("SYSTEM") == "spaces"
    config = Config()
    target_sample_rate = 48000
    edge_output_filename = "edge_output.mp3"
    tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
    tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
    model_root = "weights"
    models = [
        d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
    ]
    if len(models) == 0:
        raise ValueError("No model found in `weights` folder")
    models.sort()
def model_data(model_name):
    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")
    pth_path = pth_files[0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")
    return tgt_sr, net_g, vc, version, index_file, if_f0
def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
if __name__ == '__main__':
    print("Loading hubert model...")
    hubert_model = load_hubert()
    print("Hubert model loaded.")
    print("Loading rmvpe model...")
    rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
    print("rmvpe model loaded.")
@nb.jit(nopython=True, parallel=True, fastmath=True)
def process_audio(audio, tgt_sr, filter_radius, rms_mix_rate):
    for i in nb.prange(len(audio)):
        audio[i] = audio[i] * 2.0
    return audio
def tts_process(
    model_name,
    f0_key,
    volume,
    vc_gain,
    is_normal,
    index_rate,
    protect,
    tts_text,
    tts_voice,
    speed,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    audio_format,
    bitrate,
):
    voice_path = 'tts_voice.' + audio_format
    print(f"Model: {model_name}")
    print(f"Key: {f0_key}, Index: {index_rate}, Protect: {protect}")
    tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
    print(" ")
    print("text:")
    print(tts_text)
    print(f"voice: {tts_voice}")
    t0 = time.time()
    if speed >= 0:
        speed_str = f"+{speed}%"
    else:
        speed_str = f"{speed}%"
    asyncio.run(
        edge_tts.Communicate(
            tts_text, "-".join(tts_voice.split("-")[:-1]), rate=speed_str
        ).save(edge_output_filename)
    )
    print("selected TTS")
    t1 = time.time()
    edge_time = t1 - t0
    audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"Audio duration: {duration}s")
    if limitation and duration >= 30:
        print("Error: Audio too long")
        return (
            f"Audio should be less than 30 seconds in this huggingface space, but got {duration}s.",
            edge_output_filename,
            None,
        )
    f0_key = int(f0_key)
    if not hubert_model:
        load_hubert()
    vc.model_rmvpe = rmvpe_model
    times = [0, 0, 0]
    audio = process_audio(audio, tgt_sr, filter_radius, rms_mix_rate)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,
        audio,
        edge_output_filename,
        times,
        f0_key,
        "rmvpe",
        index_file,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        None,
        volume,
    )
    if tgt_sr != resample_sr >= 16000:
        tgt_sr = resample_sr
    info = f"Successfully! Time of processing: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
    print(info)
    audio_data_bytes = np.array(audio_opt, dtype=np.int16).tobytes()
    audio_segment = AudioSegment(
        audio_data_bytes,
        frame_rate=tgt_sr,
        sample_width=2,
        channels=1
    )
    if is_normal:
        audio_segment = audio_segment.normalize()
    audio_segment.export(voice_path, format=audio_format, bitrate=str(bitrate) + "k")
    voice_audio = AudioSegment.from_file(voice_path)
    resampled_voice_audio = voice_audio.set_channels(2).set_frame_rate(target_sample_rate)
    resampled_voice_audio = resampled_voice_audio + (vc_gain)
    resampled_voice_audio.export(voice_path, format=audio_format)
    return (
        info,
        edge_output_filename,
        voice_path,
    )
def audio_file_process(
        model_name,
        f0_key,
        volume,
        mus_gain,
        vc_gain,
        is_normal,
        index_rate,
        protect,
        mode,
        youtube_url,
        audio_file,
        sound_speed,
        is_ff,
        is_spleeter,
        ff_start_multi,
        ff_max_length,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        audio_format,
        bitrate,
):
    global separator
    if __name__ == '__main__':
        temp_audio_file = 'temp_audio.'+audio_format
        voice_path = 'voice.'+audio_format
        music_path = 'music.'+audio_format
        combined_file = 'combined.'+audio_format
        vc_output_path = 'vc_output.'+audio_format
        audio_path = 'input.'+audio_format
        if mode == 'youtube-url':
            youtube = youtube_url.split('&')
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_format,
                    'preferredquality': str(bitrate),
                }],
                'outtmpl': 'input',
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube[0]])
        else:
            audio_path = audio_file.name
        if is_ff:
            print("using ffmpeg")
            audio_clip, audio_clip_sample_rate = sf.read(audio_path)
            command = ["ffmpeg", "-y", "-i", audio_path, "-ss", str((len(audio_clip) / audio_clip_sample_rate) / ff_start_multi), "-to",
                    str(len(audio_clip) / audio_clip_sample_rate), "-t", str(ff_max_length), temp_audio_file]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            temp_audio_file = audio_path
        if is_spleeter:
            multiprocessing.freeze_support()
            print('using spleeter')
            audio_data, sample_rate = sf.read(temp_audio_file)
            results = separator.separate(audio_data)
            sf.write(voice_path, results['vocals'], sample_rate)
            sf.write(music_path, results['accompaniment'], sample_rate)
            music_audio = AudioSegment.from_file(music_path)
            resampled_music_audio = music_audio.set_channels(2).set_frame_rate(target_sample_rate)
            resampled_music_audio = resampled_music_audio + (mus_gain)
            resampled_music_audio.export(music_path, format=audio_format)
            voice_audio = AudioSegment.from_file(voice_path)
            resampled_voice_audio = voice_audio.set_channels(2).set_frame_rate(target_sample_rate)
            resampled_voice_audio = resampled_voice_audio + (vc_gain)
            resampled_voice_audio.export(voice_path, format=audio_format)
        print(f"Model: {model_name}")
        print(f"Key: {f0_key}, Index: {index_rate}, Protect: {protect}, slow_speed: {sound_speed}, ffmpeg: {is_ff}, start_cut_mulit: {ff_start_multi}, duration: {ff_max_length}, spleeter: {True}")
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        if is_spleeter:
            print("selected Audio File: ", voice_path)
            audio, sr = librosa.load(voice_path, sr=(sound_speed * 160), mono=True)
            duration = len(audio) / sr
            print(f"Estimate work time: {duration}s")
        else:
            print("selected Audio File: ", temp_audio_file)
            audio, sr = librosa.load(temp_audio_file, sr=(sound_speed * 160), mono=True)
            duration = len(audio) / sr
            print(f"Estimate work time: {duration}s")
        f0_key = int(f0_key)
        if not hubert_model:
            load_hubert()
        vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio = process_audio(audio, tgt_sr, filter_radius, rms_mix_rate)
        if is_spleeter:
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                voice_path,
                times,
                f0_key,
                "rmvpe",
                index_file,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                None,
                volume,
            )
        else:
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                temp_audio_file,
                times,
                f0_key,
                "rmvpe",
                index_file,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                None,
                volume,
            )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Successfully! Time of processing: npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        audio_data_bytes = np.array(audio_opt, dtype=np.int16).tobytes()
        audio_segment = AudioSegment(
            audio_data_bytes,
            frame_rate=tgt_sr,
            sample_width=2,
            channels=1
        )
        if is_spleeter:
            audio_segment = audio_segment.set_channels(2).set_frame_rate(target_sample_rate)
            audio_segment.export(vc_output_path, format=audio_format, bitrate=str(bitrate) + "k")
            audio1 = AudioSegment.from_file(vc_output_path)
            audio2 = AudioSegment.from_file(music_path)
            if len(audio1) < len(audio2):
                audio1 = audio1 + AudioSegment.silent(duration=len(audio2) - len(audio1))
            else:
                audio2 = audio2 + AudioSegment.silent(duration=len(audio1) - len(audio2))
            combined_audio = audio1.overlay(audio2)
            if is_normal:
                combined_audio = combined_audio.normalize()
            combined_audio.export(combined_file, format=audio_format)
        else:
            if is_normal:
                audio_segment = audio_segment.normalize()
            audio_segment.export(vc_output_path, format=audio_format, bitrate=str(bitrate) + "k")

        return (
            info,
            temp_audio_file,
            voice_path,
            music_path,
            vc_output_path,
            combined_file,
        )
if __name__ == '__main__':
    tts = gr.Interface(
        fn=tts_process,
        inputs=[
            gr.Dropdown(
                label="Model (RVC v1 and RVC v2 supported)",
                choices=models,
                value=models[0],
            ),
            gr.Slider(
                label="Voice tone (the choice depends on the selected model)",
                value=0,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="RVC output volume (%)",
                value=100,
                minimum=0,
                maximum=500,
                step=0.1,
            ),
            gr.Slider(
                label="Gain RVC voice (db)",
                value=0,
                minimum=-100,
                maximum=100,
                step=1,
            ),
            gr.Checkbox(
                label="Normalize",
                value=True,
            ),
            gr.Slider(
                label="Extract tone from model (0 - half | 1 - complete)",
                value=1,
                minimum=0,
                maximum=1,
                step=0.1,
            ),
            gr.Slider(
                label="Retention (the lower the value, the more accurately it determines the tone, there is an inverse relationship with tone enhancement)",
                value=0.33,
                minimum=0,
                maximum=1,
                step=0.1,
            ),
            gr.Textbox(
                label="Enter text",
                value="test segment for dubbing tts",
            ),
            gr.Dropdown(
                label="Edge-tts voice selection",
                choices=tts_voices,
                value="en-GB-RyanNeural-Male",
            ),
            gr.Slider(
                label="Speech speed (%)",
                value=0,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="Filter",
                value=3,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="Resampling",
                value=0,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="Mix",
                value=0.25,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Radio(
                ["wav", "mp3"],
                value="mp3",
                label="Select format",
            ),
            gr.Slider(
                label="Bitrate",
                value=128,
                minimum=64,
                maximum=320,
                step=64,
            ),
        ],
        outputs=[
            gr.Textbox(label="Console"),
            gr.Audio(label="EDGE-TTS"),
            gr.Audio(label="Result"),
        ],
    )
    audio_file = gr.Interface(
        fn=audio_file_process,
        inputs=[
            gr.Dropdown(
                label="Model (RVC v1 and RVC v2 supported)",
                choices=models,
                value=models[0],
            ),
            gr.Slider(
                label="Tone (the choice depends on the selected model)",
                value=0,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="RVC output volume (%)",
                value=100,
                minimum=0,
                maximum=500,
                step=0.1,
            ),
            gr.Slider(
                label="Gain music (db)",
                value=0,
                minimum=-100,
                maximum=100,
                step=1,
            ),
            gr.Slider(
                label="Gain RVC voice (db)",
                value=0,
                minimum=-100,
                maximum=100,
                step=1,
            ),
            gr.Checkbox(
                label="Normalize",
                value=True,
            ),
            gr.Slider(
                label="Extract tone from model (0 - half | 1 - complete)",
                value=1,
                minimum=0,
                maximum=1,
                step=0.1,
            ),
            gr.Slider(
                label="Retention (the lower the value, the more accurately it determines the tone, there is an inverse relationship with tone enhancement)",
                value=0.33,
                minimum=0,
                maximum=1,
                step=0.1,
            ),
            gr.Radio(
                ["youtube-url", "audiofile"],
                value="audiofile",
                label="Select mode",
            ),
            gr.Textbox(
                label="youtube link to video",
            ),
            gr.File(
                label="Audio file (mp3, wav)",
            ),
            gr.Slider(
                label="Slow down playback (%)",
                value=100,
                minimum=50,
                maximum=150,
                step=0.1,
            ),
            gr.Checkbox(
                label="ffmpeg",
                value=True,
            ),
            gr.Checkbox(
                label="spleeter",
                value=True,
            ),
            gr.Slider(
                label="start of track trim multiplier (x > then ss <)",
                value=10,
                minimum=0.1,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="Track length (s)",
                value=90,
                minimum=1,
                maximum=360,
                step=0.1,
            ),
            gr.Slider(
                label="Filter",
                value=3,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="Resampling",
                value=0,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Slider(
                label="Mix",
                value=0,
                minimum=-100,
                maximum=100,
                step=0.1,
            ),
            gr.Radio(
                ["wav", "mp3"],
                value="mp3",
                label="Select format",
            ),
            gr.Slider(
                label="Bitrate",
                value=128,
                minimum=64,
                maximum=320,
                step=64,
            ),
        ],
        outputs=[
            gr.Textbox(label="Console"),
            gr.Audio(label="Original"),
            gr.Audio(label="Original-voice"),
            gr.Audio(label="Original-music"),
            gr.Audio(label="RVC-voice"),
            gr.Audio(label="Result"),
        ],
    )
    tabbed_interface = gr.TabbedInterface(
        [tts, audio_file], ["TTS-RVC", "AUDIO-RVC"]
    )
    tabbed_interface.launch()