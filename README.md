
<div style="background-color: white; padding: 10px; display: inline-block;">
    <img src="https://raw.githubusercontent.com/veitlab/moove/main/assets/logo_white_bg.png" alt="Moove Logo" width="250">
</div>

# Moove

Moove (Marking Online using Only the Onsets of Vocal Elements) is a novel tool for real-time syllable segmentation and classification of birdsong, designed to enable closed-loop experiments in vocal learning research. Designed to study the learned vocalisations of Bengalese finches, Moove identifies target syllables in a bird's song and provides feedback in real time. Moove provides an out-of-the-box, neural network-based approach to reliably target vocal syllables before their end, enabling a reinforcement protocol where a specific syllable can be targeted with aversive white noise or an alternative feedback stimulus if adjusted.

Moove uses a two-stage architecture: a convolutional-based encoder that segments syllables in the audio signal and a CNN classifier that assigns each detected syllable segment a label, identifying its type based on the initial part of its structure. This design allows Moove to operate at a lower audio chunk duration than other tools, enabling faster and more accurate syllable recognition with minimal latency. Moove includes a GUI for creating training datasets using unsupervised methods and training the networks, as well as a recording script for real-time syllable targeting.

## Installation

### Quick start (recommended)

The fastest way to get Moove running. Requires Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/veitlab/moove.git
cd moove
uv sync
uv run moovegui        # GUI for datasets & training
uv run moovetaf        # real-time recording & targeting
```

If you do not have `uv` yet:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### With pip

```bash
pip install moove
```

or for a specific version:
```bash
pip install moove==1.1.0
```

After installing, a default configuration file (`moove_config.ini`) will be available at `~/.moove/`. This configuration file should be adjusted to fit your experiment setup.

> **Note for existing users (upgrading from v1.0.x):** Version 1.1.0 includes substantial changes (GUI migration from tkinter to PyQt6, updated dependencies, new test suite). If you need the previous version for compatibility reasons:
> ```bash
> pip install moove==1.0.3
> ```

### PortAudio Installation

Moove uses the `sounddevice` library, which depends on PortAudio. On most systems, PortAudio is already available or bundled. If needed, follow the steps below to install it:

#### Windows
PortAudio is bundled with `sounddevice` on Windows -- no extra installation needed.

#### macOS
Install PortAudio via Homebrew before installing Moove:
```bash
brew install portaudio
```

#### Linux (Debian / Ubuntu)
Install the PortAudio development library:
```bash
sudo apt update
sudo apt install python3-dev gcc portaudio19-dev
```

### Enabling ASIO Support (Windows)

ASIO provides the lowest latency, which is critical for Moove's real-time targeting capabilities.

**Modern method** (sounddevice >= 0.5): set the environment variable before starting Moove:
```powershell
$env:SD_ENABLE_ASIO = "1"
moovegui
```

**Legacy method**: replace the default PortAudio DLL manually:
- Locate `libportaudio64bit.dll` in your `sounddevice` installation's `_sounddevice_data/portaudio-binaries/` folder
- If `libportaudio64bit-ASIO.dll` exists alongside it: delete the non-ASIO DLL and rename the ASIO version
- If only the non-ASIO DLL exists: download the ASIO-enabled binary from https://github.com/spatialaudio/portaudio-binaries and replace the existing file

## Configuration

### Default Configuration Location

By default, Moove stores its configuration file (`moove_config.ini`) in `~/.moove/`. This configuration file should be adjusted to fit your experiment setup.

### Custom Configuration Directory

To store the configuration in a different location, use the `MOOVE_CONFIG_DIR` environment variable:

**Windows:**
```cmd
set MOOVE_CONFIG_DIR=D:\moove_config
moovegui
```

**Linux/macOS:**
```bash
export MOOVE_CONFIG_DIR="/path/to/config"
moovegui
```

## Usage

Once installed, Moove offers two main entry points:

- `moovegui`: Opens the GUI for creating labeled datasets and training the segmentation and classification networks.
- `moovetaf`: Starts the recording and targeting application, enabling real-time targeting of specific syllables.

To start, simply type `moovegui` or `moovetaf` in the terminal. When using uv:
```bash
uv run moovegui
uv run moovetaf
```

### Requirements

- Python 3.10 -- 3.12 (3.11 recommended)
- Audio hardware: a microphone and speaker setup is required for online targeting experiments

### Troubleshooting

**PortAudio / sounddevice errors:**

Ensure PortAudio is installed on your system (see Installation above). On Linux: `sudo apt install portaudio19-dev`. On macOS: `brew install portaudio`.

**GUI not starting:**

Make sure you have a display server available. On headless Linux systems, PyQt6 requires an X11 or Wayland session.

**NumPy compatibility:**

Moove works with both NumPy 1.x and 2.x. If you encounter unexpected errors with NumPy 2.x, try `pip install "numpy<2"`.

### Workflow Overview

1. **Baseline Recordings** -- Begin with baseline recordings using MooveTaf. In the configuration file, set `realtime_classification` to `False` and configure `dB_threshold` for bout detection.

2. **Manual Segmentation** -- In MooveGUI, use the Resegmentation Window to manually segment recorded songs.

3. **Train the Segmentation Network** -- With the segmentation data, open the Training Window to train the segmentation network. Then return to the Resegmentation Window for automated re-segmentation.

4. **Label Creation** -- Use the Cluster Window in the GUI to label syllable segments. Clusters can be automatically labeled and then manually adjusted.

5. **Train the Classification Network** -- With labeled syllables, use the Training Window to train the classification network.

6. **Real-Time Targeting** -- Update the configuration file with the trained model names, set `realtime_classification` to `True`, and specify the target syllable. MooveTaf is now ready for real-time targeting experiments.

A detailed guide is available in the `docs/` folder and can be built with Sphinx:
```bash
cd docs
sphinx-build -b html source build/html
open build/html/index.html
```

## Contact

For questions, issues, or feedback regarding Moove, please contact:

**Primary contact:**

Lena Veit: lena.veit@uni-tuebingen.de

Nils Riekers: nils@riekers.it

## License

Moove is licensed under the MIT License. You are free to use, modify, and distribute the software, provided that you retain the copyright notice and give appropriate credit to the original authors.

See [LICENSE](LICENSE) for details.
