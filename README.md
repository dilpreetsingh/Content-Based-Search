# Visually Similar Image Search

Finding similar images can be useful in many cases, for example one can use it to retrieve mountain photos from a ton of photos in his gallery. In this project, we aim to achieve this search by using a nearest neighbour approach over image features produced by pretrained neural networks.

## Motivation
Neural networks learn to extract features from data without any explicit knowledge. Not only these features are relevant to solve given problems that they are trained for, the features might also be useful for other tasks. In this case, we use three pretrained networks, namely [VGG16][vgg16], [ResNet152][resnet], and [DenseNet][densenet]. We aim to use these features as a representation of each image. We hope that visually similar images would have a similar representation. In other words, these images lay closely in this feature space. 

<div align="center">
<img src="https://i.imgur.com/as9lJ7i.png"/><br>
<b>Fig. 1: Project Overview</b>
</div>

With this representation, it enables us to perform nearest neighbor search. We use [Annoy][annoy], performing approximate nearest neightbor search.

## Analysis Tool
We have developed [a website][tool] that provides an interface for exploring results from our experiments in an informative and reproducible way. If you are interested in self-exploring and digging into the results, please give it a try 😎.


## Experiment 1: Visually Similar Artworks
In this first part, we randomly take 5000 artworks from [MoMa's collection][dataset]. The goal is to explore how the images of these artworks are projected onto the feature space.

<div align="center">
<img src="https://i.imgur.com/Jz9snGi.png"/><br>
<b>Fig. 2: Visually Similar Artworks</b>
</div>

From Fig. 2, we can see that the nearest neighbours in these feature spaces are somehow related to the given images. For example, if we look at the artwork `Pettibon with Strings`,  similar artworks from ResNet152 and Densenet contains faces. Please explore [our analysis tool][tool] for more examples.


## Experiment 2: Recovery Perturbed Artworks
The goal of this experiment is to verify whether the neural network features of close images are also more or less the same. In other word, these images are semantically the same for us. As shown in Fig. 3, we use five profiles to perturb 1000 original images from  [MoMa's collection][dataset], producing close images for the experiment. 

<div align="center">
<img src="https://i.imgur.com/KOtYz4V.png" width="600"/><br>
<b>Fig. 3: Perturbation Profiles</b></br>
</div>

Therefore, if the representation of an image and its perturbed versions are similar, we should be able to recover those perturbed images when performing nearest neighbour search. 


<div align="center">
<img src="https://i.imgur.com/jDW2Y4z.png"/><br>
<b>Fig. 4: Recovery Perturbed Artworks</b><br>
</div>

As shown in Fig. 4, we can see that the two original images and their perturbed versions are proximiately close in the feature spaces, particular the feature space of VGG16. For these two examples, VGG16's feature space allows us to recover 4/5 corrupted versions while the feature spaces of the other networks yield the ratio of 3/5.

With this setting, we can also quantitatively measure the performance of the results by looking at precision, recall, and f1-score. 

- Precision: `no. correctly returned samples / no. returned samples`
- Recall: `no. correctly returned samples / no. all relevant samples in data`
- f1-score: `2*(precision*recall)/(precision + recall)`
 
In this case, the `no. correctly returned samples` is simply the number of an image's perturbed versions being returned. `no. all relevant samples in data` is 5 because we have five perturbation profiles, and `no. returned samples` is `k` whose values are `1, 3, 5`.

<div align="center">
<img src="https://i.imgur.com/N3MOOdB.png"/><br>
<b>Fig. 5: Averaged Precision, Recall, and F1-score</b><br>
</div>

From Fig. 5, we can see that VGG16 performs quite good on average and better than the other architectures for this purpose of study. Their large variation seems to suggest that there are some cases that `ResNet152` and `DenseNet` can embed close images to near locations in the feature spaces. This might be a good further analysis. 

## Future Work
- Try with more samples. Maybe 10,000 artworks?
- Use features from autoencoders (vanila, VAE)
- Train autoencoders with the following scheme: `perturbed image -> autoencoder -> original image`.

## Development
Please refer to `DEVELOPMENT.md`.

## Acknowledgements
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
