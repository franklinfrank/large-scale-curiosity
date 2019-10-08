**Status:** Archive (code is provided as-is, no updates expected)

## Large-Scale Study of Curiosity-Driven Learning ##
#### [[Project Website]](https://pathak22.github.io/large-scale-curiosity/) [[Demo Video]](https://youtu.be/l1FqtAHfJLI)

[Yuri Burda*](https://sites.google.com/site/yburda/), [Harri Edwards*](https://github.com/harri-edwards/), [Deepak Pathak*](https://people.eecs.berkeley.edu/~pathak/), <br/>[Amos Storkey](http://homepages.inf.ed.ac.uk/amos/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)<br/>
(&#42; alphabetical ordering, equal contribution)

University of California, Berkeley<br/>
OpenAI<br/>
University of Edinburgh

<a href="https://pathak22.github.io/large-scale-curiosity/">
<img src="https://pathak22.github.io/large-scale-curiosity/resources/teaser.jpg" width="500">
</img></a>

This is a TensorFlow based implementation for our [paper on large-scale study of curiosity-driven learning](https://pathak22.github.io/large-scale-curiosity/) across
54 environments. Curiosity is a type of intrinsic reward function which uses prediction error as reward signal. In this paper, We perform the first large-scale study of purely curiosity-driven learning, i.e. without any extrinsic rewards, across 54 standard benchmark environments. We further investigate the effect of using different feature spaces for computing prediction error and show that random features are sufficient for many popular RL game benchmarks, but learned features appear to generalize better (e.g. to novel game levels in Super Mario Bros.). If you find this work useful in your research, please cite:

    @inproceedings{largeScaleCuriosity2018,
        Author = {Burda, Yuri and Edwards, Harri and
                  Pathak, Deepak and Storkey, Amos and
                  Darrell, Trevor and Efros, Alexei A.},
        Title = {Large-Scale Study of Curiosity-Driven Learning},
        Booktitle = {arXiv:1808.04355},
        Year = {2018}
    }

### Installation and Usage
You will first need to install Deepmind Lab. Afterwards, you can install python dependencies by running
```bash
pip install -r requirements.txt
```
The following command should train the agent that achieves the baseline results on staticmaze01:
```bash
python run.py --exp_name randfeats_staticmaze01_usenews_extcoeff_1_intcoeff_.01_layernorm0_nepochs8_nsteps_per_seg512_envs_per_process32
--use_news 1 --ext_coeff 1.0 --int_coeff 0.01 --layernorm 0 --feat_learning none --nepochs 8 --nsteps_per_seg 512
--envs_per_process 32
```
To use more than one gpu/machine, use MPI (e.g. `mpiexec -n 8 python run.py` should use 1024 parallel environments to collect experience instead of the default 128 on an 8 gpu machine).

### Generating Video
Install ffmpeg, then run the following command:
```bash
ffmpeg -r 15 -start_number 0 -i images/{exp_name}_{ep_num}_0_%d_.png -vcodec mpeg4 {video_name}.mp4
```
### Data for plots in paper

[Data for Figure-2](https://www.dropbox.com/s/ufr7o8g9omb9zpl/experiments.tar.gz): contains raw game score data along with the plotting code to generate Figure-2 in the paper.

### Other helpful pointers
- [Paper](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf)
- [Project Website](https://pathak22.github.io/large-scale-curiosity/)
- [Demo Video](https://youtu.be/l1FqtAHfJLI)
