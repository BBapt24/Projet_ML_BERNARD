# Projet_ML_BERNARD

## *Introduction*

Depuis leur introduction en 2017, les réseaux de neurones de type transformers semblent avoir pris le pas sur tous les autres types de réseaux et ce, quel que soit le type de problème à résoudre. C'est pourquoi, dans le cadre de mon projet de Machine Learning, je propose de s'atteler à un problème classique de classification d'image en étudiant deux architectures de réseaux face à la base de données CIFAR-10 (*CIFAR-10 se compose de 60000 images couleur 32x32 réparties en 10 classes, avec 6000 images par classe. Il y a 50000 images d'apprentissage et 10000 images de test.*). 


La première architecture sera une version simple de ResNet, un type de réseaux qui a été conçu pour résoudre ce type de problème.
La seconde architecture sera un modèle de Transformers dont je détaillerai l'architecture et la création par la suite.

## *I- ResNet*

ResNet (Residual Networks) est une architecture de réseau neuronal profond qui a introduit le concept de bloc résiduel, permettant de résoudre le problème de la disparition du gradient dans les réseaux profonds. L'idée clé derrière ResNet est d'introduire des connexions résiduelles (skip connections) entre les couches, ce qui permet au modèle d'apprendre les résidus plutôt que les fonctions directes. 

Introduit en 2015, cette architecture est donc très pratique pour réaliser des tâches habituellement confiées à des réseaux de convolution avec une meilleure convergence que la plupart des modèles le précédant. C'est pourquoi on l'utilise pour de la classification d'images ici.


![image](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/675134cb-5e3d-46f4-a30e-cd60ce28cb3f)
**Figure 1 - Descriptif des différentes architectures de ResNet [1]**

Dans le cadre de ce projet, j'ai décidé d'implémenter le modèle ResNet 18 couches à l'aide de pytorch puis de l'entrainer sur la base CIFAR-10. Pour ce faire, je me suis inspiré du travail d'un tutoriel donné en annexe [3].

*1. Entraînement du modèle*


*2. Résultats & commentaires*


## *II- Transformers*


*1. Entraînement du modèle*


*2. Résultats & commentaires*



## *Conclusion*

Finalement, les résultats obtenus à l'issu de cette expérience sont contre-intuitifs. En effet, malgré l'évidente efficacité du modèle de Vision Transformer, on peut discuter de l'intérêt de leur mise en place lors d'exemples comme celui-ci où des méthodes moins coûteuses et mieux documentées fonctionnent déjà. 

Ainsi, ce travail m'a permis de constater que les réseaux convolutionnels tels que ResNet restent une solution efficace et simple d'accès pour de la classification d'image sur ce type de base de données. Que ce soit sur la taille des modèles, le temps de travail pour que celui-ci ait une efficacité satisfaisante ou bien la simple documentation pour comprendre les concepts mis en oeuvre dans l'architecture utilisée, il me paraît évident de privilégier une approche plus simple comme celle effectuée lors du grand I (si des résultats extrêmements précis ne sont pas nécessaires. Il faut rappeler que sur la plupart des benchmarks les ViT sont plus efficaces que les ConvNets, mais au prix de quels efforts ?).




## *Bibliographie*

[1] https://arxiv.org/pdf/1512.03385.pdf (papier de recherche introduisant le modèle ResNet).

[2] https://arxiv.org/pdf/2010.11929v2.pdf (papier de recherche introduisant les Vision Transformers).

[3] https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448 (tutoriel d'implémentation de ResNet sous Pytorch)

[4]


