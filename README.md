### Overview

This repository contains the code necessary to collect the data, run the analysis and render the figures/text for the paper at (TODO: add biorxiv link). 

Highlights:

- All of the analysis is in a single notebook at [analysis.ipynb](analysis.ipynb).
- Logic for collecting and preparing Open Targets data is in the module [analysis.py](analysis.py) (more specifically [here](https://github.com/related-sciences/clinical_advancement_paper/blob/5ad0e5cec1aa304209bb01bd47366442b013f306/analysis.py#L121-L816)).
- The most recently rendered version of the paper is at [paper/paper.pdf](paper/paper.pdf).
- The [xrml.py](xrml.py) module is a frozen, near-replica of https://github.com/related-sciences/xrml

### Reproduction

Here are commands necessary to reproduce this analysis:

```bash
mamba env create --file environment.yml
conda activate paper

echo 'GOOGLE_CLOUD_PROJECT="<YOUR_PROJECT_NAME>"' > .env
export DATA_DIR="<YOUR_DATA_DIR>"
export CLI="analysis.py"
export OT_VERSION="23.12"

# This will download and extract all necessary datasets from a particular OT release
./bin/run_py --output_mode=dev -- $CLI \
export_features --version="$OT_VERSION" --output-path=$DATA_DIR

# This will run the analysis.ipynb notebook, which will generate all figures and tables
./bin/run_py --output_mode=dev -- $CLI \
run_analysis --version="$OT_VERSION" --output-path=$DATA_DIR

# This will run the same notebook with multiple configurations as a part of the sensitivity analysis (takes an hour or two)
./bin/run_py --output_mode=dev -- $CLI \
run_analysis_configs --output-path=$DATA_DIR 2>&1 | tee /tmp/log.txt
```

To build the `paper.tex` file:

- Install [Latex Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) for VS Code
- Follow the [instructions](https://github.com/James-Yu/LaTeX-Workshop/wiki/Install#installation) for installing TeX Live, which is used by that extension
- Open `paper.text` and press the green play arrow at the top to build the tex project at the top, or make a change of some kind and save the file
  - Conveniently, the project will also rebuild if any of the figures/tables are changed on disk (i.e. if you run cells that do this in `analysis.ipynb`)
