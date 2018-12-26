# Image-Based Search

Finding similar images can be useful in some cases, for example one can use it to retrieve mountain photos from a ton of photos in his gallery. In this project, we aim to achieve this search by using a nearest neighbour approach over image features produced by neural networks.

## Motivation
Neural networks learn to extract features from data without any explicit knowledge. Not only these features are relevant to solve given problems that they are trained for, the features might also be useful for other tasks. In this case, we use three pretrained networks, namely VGG16, ResNet152, and DenseNet. We aim to use these features as a representation of each image. We hope that visually similar images would have a similar representation. In other words, these images lay closely in this feature space. 

<div align="center">
<img src="https://i.imgur.com/as9lJ7i.png"/><br>
<b>Fig. 1: Project Overview</b>
</div>

With this representation, it enables us to perform nearest neighbor search. We use [Annoy][annoy], performing approximate nearest neightbor search.

## Experimental Setting

## Results
An interactive analysis can be found at [Something][tool].

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
