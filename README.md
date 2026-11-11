# Can Notes Alone Be Jazzy?

Robert Wienroeder, ISMIR 2026 late breaking demo.

audio demo: https://row56.github.io/cannotesalonebejazzy/

paper: [Wienroeder_CanNotesAloneBeJazzy.pdf](Wienroeder_CanNotesAloneBeJazzy.pdf)

## what this is

a melodic VAE trained on jazz and classical melodies. monophonic, 16th grid, no harmony, no swing, no sound. after training i take the difference between the jazz and classical centroids in the latent space and use it as a jazz axis. moving a melody along that axis and decoding it makes it more or less jazzy.

the first model separated jazz and classical at AUC 0.99, but it was mostly separating transcribed performances from sheet music. so the final model is trained on both formats for both genres, and the test sets cover both genres in both formats. all numbers are in the paper.

## running it

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/base.yaml
```
