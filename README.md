# Image-Based Search (in progress)

Finding similar images can be useful in some cases, for example one can use it to retrieve mountain photos from a ton of photos in his gallery. In this project, we aim to achieve this search by using a nearest neighbour approach over image features produced by neural networks.

## Motivation
Neural networks learn to extract features from data without any explicit knowledge. Not only these features are relevant to solve given problems that they are trained for, the features might also be useful for other tasks. In this case, we use three pretrained networks, namely [VGG16][vgg16], [ResNet152][resnet], and [DenseNet][densenet]. We aim to use these features as a representation of each image. We hope that visually similar images would have a similar representation. In other words, these images lay closely in this feature space. 

<div align="center">
<img src="https://i.imgur.com/as9lJ7i.png"/><br>
<b>Fig. 1: Project Overview</b>
</div>

With this representation, it enables us to perform nearest neighbor search. We use [Annoy][annoy], performing approximate nearest neightbor search.

## Analysis Tool
An interactive analysis can be found at [Something][tool].

## Experiment 1: Visually Similar Artworks
In this first part, we randomly take 5000 artworks from [MoMa's collection][dataset]. The goal is to explore how the images of these artworks are projected onto the feature space.

<div align="center">
<img src="https://i.imgur.com/Jz9snGi.png"/><br>
<b>Fig. 2: Visually Similar Artworks</b>
</div>

TODO: Some analysis

## Experiment 2: Recovery Perturbed Artworks
<div align="center">
<img src="https://i.imgur.com/KOtYz4V.png"/><br>
<b>Fig. 3: Perturbation Profiles</b></br>
</div>

<div align="center">
<img src="https://i.imgur.com/jDW2Y4z.png"/><br>
<b>Fig. 4: Recovery Perturbed Artworks</b><br>
</div>

<div align="center">
<img src="https://i.imgur.com/9Jzl8gS.png"/><br>
<b>Fig. 5: Precision, Recall, and F1-score</b><br>
</div>


## Development
Please refer to `DEVELOPMENT.md`.

## Acknowledgement
- This project is inspired by Dilpreet Singh's [Searching for Visually Similar Artworks
][idea].
- Data is from [MoMa's collection][dataset].


[idea]: http://ai.sensilab.monash.edu/2018/09/17/similarity-search-engine/
[dataset]: https://github.com/MuseumofModernArt/collection
[tool]: http://pat.chormai.org/artwork-similarity-vis-tool/
[annoy]: https://github.com/spotify/annoy
[vgg16]: x
[resnet]: x
[densenet]: x
