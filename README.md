# Retrieval-Voice-Conversion EdgeTTS & Audio WebUI+API

With this project, you can generate text to speech using any RVC model (including using your own trained ones). 

Re-voice a speech or re-sing a song by separating the voice and instrumental with a splitter 2stems. 

Changing the length and the indentation at the beginning( to skip silence or intro). 

Also included bunch of settings.

There is a built-in API, now you can use requests like a `rvctts` or `rvcaudio` to generate text to speech or sing songs!

>They say that if you set all the settings perfectly, you will get a masterpiece, but this is not so easy to achieve...

## TTS-RVC

This is a text-to-speech [rvc](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) models, using [edge-tts](https://github.com/rany2/edge-tts).

>Has a wide range of settings

![tts](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/3e4dbffe-a7aa-4dfb-91dd-9f77cde607a1)

## AUDIO-RVC

This is a audio [rvc](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) models, using [spleeter](https://github.com/deezer/spleeter), [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).

>The settings are the same as in TTS, but audio file specific settings have been added.

![audio](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/3c93be52-44f5-4ae6-bc97-b28a5612c351)

## Examples

[tts_voice.webm](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/858c4b77-1486-415d-92ba-28016db3ea50)

[combined.webm](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/1e2bed14-6e14-45ba-a864-a57a45612623)

## Locate RVC models

You can place your RVC models in `weights/` directory as follows:

```bash
weights
├── mymodel
│   ├── mymodel.pth
│   └── mymodel.index
└── testmodel
    ├── testmodel.pth
    └── testmodel.index
...
```

Each model directory should contain exactly one `.pth` file and at most one `.index` file. Directory names are used as model names.

Non-ASCII characters in path names gave faiss errors (like `weights/デル/mymodel.pth`), so please avoid them.

## Using built-in API

For using built-in API you can send POST requests `rvctts` or `rvcaudio` to `127.0.0.1:7850` with some required parameters
```bash
POST requests
├── /rvctts
│   ├── text - text to generate speech
│   ├── model - RVC model name from weights
│   ├── volume - RVC output volume
│   ├── gain - RVC output gain
│   ├── edgetts_model - EdgeTTS model name from EdgeTTS
│   ├── edgetts_speed - speed of speech
│   ├── transpose - voice tone of RVC model
│   ├── indexrate - extract tone from model
│   ├── protect - retention from model
│   └── res - output audiofile(0 - EdgeTTS speech | 1 - result with rvc processing)
└── /rvcaudio
    ├── model - RVC model name from weights
    ├── volume - RVC output volume
    ├── gain_vc - RVC output gain
    ├── gain_mus - music gain
    ├── url - youtube url video
    ├── slow - slow down playback
    ├── transpose - voice tone of RVC model
    ├── indexrate - extract tone from model
    ├── protect - retention from model
    ├── res - output audiofile(0 - original | 1 - original-voice | 2 - original-music | 3 - rvc-voice | 4 - result with rvc processing and music)
    ├── multi - start of track trim multiplier (x > then ss <)
    └── duration - track length
```

## Requirements

>Python 3.10

>NVIDIA CUDA or AMD ROCM(linux only) or CPU(so slow)

## Install guide for Windows

**№1 Clone or download project**

`git clone https://github.com/PRO100KBAC/rvc-tts-audio-webui-api.git`

**№2 Just open the fix.bat file**

>wait until process of installing redist vsbuildtools finished

**№3 You need to manually install .pt files from huggingface and put to the root of the project:**

>[rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true)

>[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true)

**№4 Install ffmpeg and add to the system environment variable**

>[ffmpeg](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z)

>Right click on start->system->advanced system settings->environment variables->

>System variables->Path->Edit->New->Put path to the ffmpeg folder

**№5 Open the install.bat file**

When during installation you will need to select Nvidia GPU cu118 & cu121 or CPU(so slow)

>after installation is complete, it starts automatically

## Install guide for Linux

**№1 Clone or download project**

`git clone https://github.com/PRO100KBAC/rvc-tts-audio-webui-api.git`

**№2 You need to manually install .pt files from huggingface and put to the root of the project:**

>[rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true)

>[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true)

**№3 Install ffmpeg**

`sudo apt install ffmpeg`

>check if ffmpeg is successfully installed by the command `ffmpeg -version`

**№4 Open the install.sh file**

When during installation you will need to select Nvidia GPU cu118 & cu121, AMD ROCM or CPU(so slow)

>after installation is complete, it starts automatically

## Launch

>Windows - run.bat

>Linux - run.sh
