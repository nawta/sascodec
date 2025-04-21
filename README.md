# SASCodec: Semantic Aware Speech Codec


## 🚧  Changes introduced in this fork
| Area | What was changed | Why |
|------|------------------|-----|
| **Dependencies** | Replaced `audiotools@git+…` with **`descript-audiotools @ git+https://github.com/descriptinc/audiotools.git@main`** | Starting with **pip 24** the resolver raises an error if the *requested name* (left of the `@`) and the `Name` field inside the package’s `METADATA` don’t match. The official distribution name is `descript-audiotools`, so the requirement had to be updated. |
| **Package versions** | Pinned to `torch==2.3.1+cu121`, `torchvision==0.18.1+cu121`, `torchaudio==2.3.1+cu121` | These versions align with the CUDA 12.1 toolchain baked into the base image `nvcr.io/nvidia/pytorch:24.05-py3`. |
| **Dockerfile** | Updated the `sed` replacements in the build step to reflect the new dependency names / CUDA 12.1 index | – |
| **setup.py** | Adjusted `install_requires` accordingly | – |
| **third_party** | (Bundling of VideoReTalking / face3d is **postponed** for now) | – |

## 🚧  このフォークで行った主な改変点
| 区分 | 変更内容 | 理由 |
|------|----------|------|
| **依存関係** | `audiotools@git+…` → **`descript-audiotools @ git+https://github.com/descriptinc/audiotools.git@main`** に書き換え | pip 24 以降は *要求名（egg / @ の左側）* と *パッケージメタデータ内の Name フィールド* が一致しないと `has inconsistent name` エラーで失敗するため。公式レポジトリの実際の distribution 名は `descript-audiotools` なので、これに合わせた（詳しくは ▶︎ 「pip 24 で起きる名前不一致問題」章を参照）。 |
| **パッケージバージョン** | PyTorch 系を <br>`torch==2.3.1+cu121`, `torchvision==0.18.1+cu121`, `torchaudio==2.3.1+cu121` に固定 | ベースイメージ `nvcr.io/nvidia/pytorch:24.05-py3` が CUDA 12.1 系のため整合を取った。 |
| **Dockerfile** | *requirements* への sed 置換を更新し、上記バージョン／CUDA 12.1 index を反映 | – |
| **setup.py** | `install_requires` の該当行を差し替え | – |
| **third_party** | （※今回は VideoReTalking/face3d の同梱は保留） | – |


## Getting Started

This is the repository for the SASCodec, which is applied in paper [TransVIP: Speech to Speech Translation System with Voice and Isochrony Preservation](https://arxiv.org/abs/2405.17809).

### Installation

```bash
pip install git+https://github.com/nethermanpro/sascodec.git
```

or install editable version.

```bash
git clone https://github.com/nethermanpro/sascodec.git
cd sascodec
pip install -e .
```

Pretrained checkpoint on CommonVoice 15 multi-lingual dataset is available [here](https://drive.google.com/file/d/1PlFqmWuG_OkXAqstbVKufSb1h88pOXuE/view?usp=sharing).

### Usage

```python
from sascodec import SASCodec
model = SASCodec.from_pretrained("/path/to/sascodec.pt")
wav = [...] # load some wav file, shape: (batch, 1, timesteps)
codes = model.encode(wav) # codes, shape: (batch, n_q, timesteps)
reconstructed_wav = model.decode(codes) # reconstructed_wav, shape: (batch, 1, timesteps)
```

### Training

Refer to the script [here](https://github.com/nethermanpro/transvip) for training the model.

## Acknowledgement

The SASCodec model and training script is based on DAC (<https://github.com/descriptinc/descript-audio-codec>) and speechtokenizer (<https://github.com/ZhangXInFD/SpeechTokenizer>). Again, we would like to thank the authors for their valuable contributions and for making their work available as open-source.

## Citation

To cite this repository

```bibtex
@article{le2024transvip,
  title={TransVIP: Speech to Speech Translation System with Voice and Isochrony Preservation},
  author={Le, Chenyang and Qian, Yao and Wang, Dongmei and Zhou, Long and Liu, Shujie and Wang, Xiaofei and Yousefi, Midia and Qian, Yanmin and Li, Jinyu and Zhao, Sheng and others},
  journal={Proceedings of the 38th International Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}
```
