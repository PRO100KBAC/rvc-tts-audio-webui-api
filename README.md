# Retrieval-Voice-Conversion EdgeTTS & Audio WebUI



## TTS-RVC

>This is a text-to-speech [rvc](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) models, using [edge-tts](https://github.com/rany2/edge-tts).

>Has a wide range of settings

![tts](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/d092102b-eeac-4c7e-bade-1e4cf7a4bbed)

## AUDIO-RVC

>This is a audio [rvc](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) models, using [spleeter](https://github.com/deezer/spleeter), [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).

>The settings are the same as in TTS, but audio file specific settings have been added.

![audio](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/0d80d1b7-e08d-42b6-98c6-a8deadae783d)

## Examples

[tts voice result](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/67986bd4-7550-43be-bff1-0f7b548c1b94)

[combined audio result](https://github.com/PRO100KBAC/rvc-tts-audio-webui-api/assets/98932626/f488e212-b029-449e-bfe8-f6f7d64eb953)

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

## Requirements

>Python 3.10

>NVIDIA or AMD ROCM or CPU(so slow)

## Install guide for Windows

**№1 Just open the fix.bat file**

>wait until process of installing redist vsbuildtools finished

**№2 You need to manually install .pt files from huggingface and put to the root of the project:**

>[rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true)

>[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true)

**№3 Install ffmpeg and add to the system environment variable**

>[ffmpeg](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z)

>Right click on start->system->advanced system settings->environment variables->

>System variables->Path->Edit->New->Put path to the ffmpeg folder

**№4 Open the install.bat file**

>after installation is complete, it starts automatically

## Install guide for Linux

>Coming soon

## Launch

>Windows - run.bat

>~~Linux - run.sh~~
