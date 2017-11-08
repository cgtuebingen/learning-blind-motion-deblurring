# Learning Blind Motion Deblurring

TensorFlow implementation of multi-frame blind deconvolution:

**Learning Blind Motion Deblurring**<br>
Patrick Wieschollek, Michael Hirsch, Bernhard Schölkopf, Hendrik P.A. Lensch<br>
*ICCV 2017*

[Download results from the paper](https://github.com/cgtuebingen/learning-blind-motion-deblurring/releases). We propose to use the [saccade-viewer](http://image-viewer.com) to compare images qualitatively.

![results](https://user-images.githubusercontent.com/6756603/28306964-93f64ce2-6ba1-11e7-8cdc-4f112d9d6059.jpg)


## Prerequisites
### 1. Get YouTube videos

The first step is to gather videos from some arbitrary sources. We use YouTube to get some videos with diverse content and recording equipment. To download these videos, we use the python-tool `youtube-dl`.

```bash
pip install youtube-dl --user
```

Some examples are given in `download_videos.sh`. Note, you can use whatever mp4 video you want to use for this task. In fact, for this re-implementation we use some other videos, which also work well.

###  2. Generate Synthetic Motion Blur

Now, we use optical flow to synthetically add motion blur. We used the most simple OpticalFlow method, wich provides reasonable results (we average frames anyway):

```bash
cd synthblur
mkdir build  && cd build
cmake ..
make all
```

To convert a video `input.mp4` into  a blurry version, run

```bash
./synthblur/build/convert "input.mp4"
```

This gives you multiple outputs:
- 'input.mp4_blurry.mp4'
- 'input.mp4_sharp.mp4'
- 'input.mp4_flow.mp4'

Adding blur from synthetic camera shake is done on-the-fly (see `psf.py`).

### 3. Building a Database
For performance reasons we randomly sample frames from all videos beforehand and store 5+5 consecutive frames (sharp+blurry) into an LMDB file (for training/validation/testing). 

I use

```bash
#!/bin/bash
for i in `seq 1 30`; do
    python data_sampler.py --pattern '/graphics/scratch/wieschol/YouTubeDataset/train/*_blurry.mp4' --lmdb /graphics/scratch/wieschol/YouTubeDataset/train$i.lmdb --num 5000
done

for i in `seq 1 10`; do
    python data_sampler.py --pattern '/graphics/scratch/wieschol/YouTubeDataset/val/*_blurry.mp4' --lmdb /graphics/scratch/wieschol/YouTubeDataset/val$i.lmdb --num 5000
done

```

To visualize the training examples just run

```bash
python data_provider.py --lmdb /graphics/scratch/wieschol/YouTubeDataset/train1.lmdb --show --num 5000
```


## Training

This re-implementation uses [TensorPack](https://github.com/ppwwyyxx/tensorpack) instead of the used custom library for the paper. Starting training is done by

```bash
python learning_blind_motion_deblurring.py --gpu 0,1 --data path/to/lmdb-files/
```

## Results
See the [release section](https://github.com/cgtuebingen/learning-blind-motion-deblurring/releases) for full-resolution images produced by our approach.

## Further experiments
We further tried a convLSTM/convGRU and a multi-scale approach (instead of the simple test from the paper). These script are available in `additional_scripts`.

## Notes
I re-trained a slightly larger model in TensorPack just for testing the TensorPack library some months ago. It seems to have similar performance (although it is not compatible with this GitHub project).
Find the inference code/weights [here](http://files.patwie.com/suppmat/patwie_iccv17_deblurring_model.tar.gz).
