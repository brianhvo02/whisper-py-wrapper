# Whisper Wrapper for NodeJS

I made a wrapper in TypeScript for the [openai/whisper](https://github.com/openai/whisper) model written in Python for use in NodeJS projects. You'll need the same dependencies as they do:
```bash
pip install -U openai-whisper

# ffmpeg

# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

All options available on the Python command line are available to use in this wrapper.

## Examples
```typescript
import Whisper from 'whisper'
import { readFile } from 'fs/promises';

// Run Whisper just like you would from the command line.
await Whisper.nativeRun('file.mp3', { language: 'en' });

// Run Whisper in a temporary directory with a Buffer as an input.
const file = readFile('file.mp3');
await Whisper.run(file, { language: 'en' });

// Run Whisper with a configuration file.

// config.json in root directory.
await Whisper.run(file);
await Whisper.run(file, 'config.json');
```