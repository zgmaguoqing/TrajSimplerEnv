# FSD Evaluation on SimplerEnv

<div align="center">

**From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation**

[[🌐 Website](https://embodied-fsd.github.io)] [[📄 Paper](https://arxiv.org/pdf/2505.08548)] [[🤗 Models](https://huggingface.co/collections/IffYuan/fsd-683fa0d552e70f302fd04b34)] [[🎯 Datasets](https://huggingface.co/collections/IffYuan/fsd-683fa0d552e70f302fd04b34)] [[💬 Demo](#demo)]

</div>

---

This repository is used to evaluate the FSD performance on bridge tasks of [SimplerEnv](https://github.com/simpler-env/SimplerEnv). 


## 💿 Installation

**Clone this repo:**

```bash
git clone --recurse-submodules https://github.com/hilookas/SimplerEnv -b fsd
cd SimplerEnv
```

**Create an anaconda environment:**

```bash
conda create -n simpler_env python=3.10
conda activate simpler_env
```

**Install SimplerEnv:**

```bash
# Following the instructions <https://github.com/simpler-env/SimplerEnv#installation>

pip install numpy==1.24.4

pushd ManiSkill2_real2sim
pip install -e .
popd

pip install -e .
```

**Install GraspNet:**

```bash
# Following the instructions in GSNet/README.md

conda install openblas-devel -c anaconda
pushd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
popd

pushd GSNet

pushd pointnet2
python setup.py install
popd

pushd knn
python setup.py install
popd

popd

pushd graspnetAPI
pip install .
popd
```

## 🏃 Execution

You can run the evaluation using:

```bash
bash scripts/fsd_bridge.sh
```

## 🙏 Acknowledgments

We sincerely thank the following open-source projects and research works:

- [SimplerEnv-SOFAR](https://github.com/Zhangwenyao1/SimplerEnv-SOFAR)
- [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim)
- [graspnetAPI](https://github.com/graspnet/graspnetAPI)
- [graspness_unofficial](https://github.com/graspnet/graspness_unofficial) / [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d)

## 📚 Citation

If you use FSD in your research, please cite our paper:
```
@misc{yuan2025seeingdoingbridgingreasoning,
      title={From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation}, 
      author={Yifu Yuan and Haiqin Cui and Yibin Chen and Zibin Dong and Fei Ni and Longxin Kou and Jinyi Liu and Pengyi Li and Yan Zheng and Jianye Hao},
      year={2025},
      eprint={2505.08548},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.08548}, 
}
```
