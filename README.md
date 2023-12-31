# Projet_ML_BERNARD

## *Introduction*

Depuis leur introduction en 2017, les réseaux de neurones de type transformers semblent avoir pris le pas sur tous les autres types de réseaux et ce, quel que soit le type de problème à résoudre. C'est pourquoi, dans le cadre de mon projet de Machine Learning, je propose de s'atteler à un problème classique de classification d'image en étudiant deux architectures de réseaux face à la base de données CIFAR-10 (*CIFAR-10 se compose de 60000 images couleur 32x32 réparties en 10 classes, avec 6000 images par classe. Il y a 50000 images d'apprentissage et 10000 images de test.*). 

![image](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/986b1371-1640-457a-9947-660e6d783e51)

**Figure 1 - Illustration de CIFAR-10**

La première architecture sera une version simple de ResNet, un type de réseaux qui a été conçu pour résoudre ce type de problème.
La seconde architecture sera un modèle de Transformers dont je détaillerai l'architecture et la création par la suite.

## *I- ResNet*

ResNet (Residual Networks) est une architecture de réseau neuronal profond qui a introduit le concept de bloc résiduel, permettant de résoudre le problème de la disparition du gradient dans les réseaux profonds. L'idée clé derrière ResNet est d'introduire des connexions résiduelles (skip connections) entre les couches, ce qui permet au modèle d'apprendre les résidus plutôt que les fonctions directes. 

Introduit en 2015, cette architecture est donc très pratique pour réaliser des tâches habituellement confiées à des réseaux de convolution avec une meilleure convergence que la plupart des modèles le précédant. C'est pourquoi on l'utilise pour de la classification d'images ici.


![image](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/675134cb-5e3d-46f4-a30e-cd60ce28cb3f)

**Figure 2 - Descriptif des différentes architectures de ResNet [1]**



Dans le cadre de ce projet, j'ai décidé d'implémenter le modèle ResNet 18 couches à l'aide de pytorch puis de l'entrainer sur la base CIFAR-10. Pour ce faire, je me suis inspiré du travail d'un tutoriel donné en annexe [3].

*1. Entraînement du modèle*

En premier lieu, il est important de déterminer les hyperparamètres du modèle. Le paramètre le plus important pour ce type de réseau est le Learning Rate, c'est pourquoi on se propose d'utiliser une fonction learning rate finder qui permettra de déterminer la valeur de ce dernier.

![image](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/8862dc5f-dfe4-4efe-98ec-cca90fb131d4)

**Figure 3 - Loss en fonction du Learning Rate**

Après analyse de la courbe, on trouve donc que le learning rate optimal est compris entre 10-3 et 10-4 car c'est là que la pente de la loss s'infléchit le plus.
Essayons donc de jouer sur ce learning rate afin d'avoir une meilleure accuracy en le conservant dans la gamme [10-3; 10-4].


*2. Résultats & commentaires*

Après avoir joué avec le learning rate (et avoir fait varier d'autres hyperparamètres, comme la batch size, qui n'améliorent pas l'accuracy du modèle de manière significative), on trouve un learning rate optimal à 2.10-4 et une accuracy optimal pour ce modèle d'environ 90%. 

![image](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/db43b6a0-c2dc-4685-93c6-827a1b48e237)

**Figure 4 - Résultats après premier entrainement**


![image](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/4d719745-21d6-48e9-aad8-362802b9d3b2)

**Figure 5 - Résultats améliorés avec modification du learning rate**

L'accuracy est faible mais le modèle est peu lourd et vite entraîné. On a donc un compromis viable pour de la classification d'image si on prend un modèle plus élaboré de ResNet qui permettrait d'obtenir une meilleure accuracy par exemple.

## *II- Transformers*

Dans cette partie, nous allons utiliser un certain type de Transformer appelé : Vision Transformers (ViT). Il s'agit d'une architecture qui utilise des mécanismes d'auto-attention pour traiter les images. Les ViT consistent en une série de blocs transformers et d'un MLP en sortie afin de classer les catégories d'image dans ce cas par exemple.


![vision-transformer-vit](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/e1f67346-cbb3-46dd-80d1-cc8429071e0a)

**Figure 7 - Architecture et fonctionnement global d'un ViT [3]**

Dans le cadre de ce projet, j'ai décidé d'implémenter un modèle de ViT à 12 couches de Transformers à l'aide de pytorch puis de l'entrainer sur la base CIFAR-10. Pour ce faire, je me suis inspiré du travail d'un tutoriel donné en annexe [4]. Malheureusement, j'ai eu énormément de difficultés à réaliser cette implémentation et ressort finalement avec peu de résultats probants. Je propose donc de décrire l'état de mon travail sur ce modèle mais suis pe confiant quant à la pertinence des résultats avancés.


*1. Entraînement du modèle*

La première difficulté rencontrée avec ce modèle est que le temps d'entraînement est bien supérieur à son prédecesseur. Là où 80 epochs donnaient un entrainement de 5 à 10 minutes pour le modèle précédent, le transformer a une taille telle que pour une trentaine d'epochs il faut compter 15 minutes d'entrainement au moins. Je n'ai donc pas pu appliquer les méthodes précédentes pour trouver un learning rate acceptable ni pour les autres hyperparamètres mais ai dû fonctionner à l'aide de la documentation que j'ai pu trouver et à tâtons en changeant les paramètres du système.



*2. Résultats & commentaires*

J'ai eu beaucoup de difficultés à implémenter ce second modèle de réseaux de neurones. En effet, du fait du nombre élevé d'hyperparamètres différents, de la multiplicité des utilisations des ViT, je me suis un peu perdu dans les papiers de recherches et n'ait pas réussi à en tirer un modèle satisfaisant pour cette expérience.

Après avoir optimisé manuellement les hyperparamètres, je trouve des résultats bien loins de ceux auquels je m'attendais : l'accuracy n'est pas si bonne que cela en comparaison avec le modèle ResNet alors que je m'attendais à une accuracy de 99%.



![image](https://github.com/BBapt24/Projet_ML_BERNARD/assets/150921474/da88d700-64ab-48e4-90f1-daee4ff2234e)


**Figure  8 - Résultats avec les meilleurs hyperparamètres trouvés**


Enfin, lors de l'étape d'analyse de mes résultats j'ai trouvé plusieurs sources selon lesquelles les ViT ne sont pas si efficaces sur des "petites" bases d'apprentissages telles que CIFAR-10 [5]. De là à en conclure que les résultats décevant que j'obtiens avec mon modèle de ViT sont dus à celà, je pense que ce serait excessif. En effet, je remarque bien que le modèle que j'utilise est encore améliorable et je pense que ce résultat pourrait être différent après plus d'heures de travail à l'optimiser.


## *Conclusion*

Finalement, les résultats obtenus à l'issu de cette expérience sont contre-intuitifs. En effet, malgré l'évidente efficacité du modèle de Vision Transformer, on peut discuter de l'intérêt de leur mise en place lors d'exemples comme celui-ci où des méthodes moins coûteuses et mieux documentées fonctionnent déjà. 

Ainsi, ce travail m'a permis de constater que les réseaux convolutionnels tels que ResNet restent une solution efficace et simple d'accès pour de la classification d'image sur ce type de base de données. Que ce soit sur la taille des modèles, le temps de travail pour que celui-ci ait une efficacité satisfaisante ou bien la simple documentation pour comprendre les concepts mis en oeuvre dans l'architecture utilisée, il me paraît évident de privilégier une approche plus simple comme celle effectuée lors du grand I (si des résultats extrêmements précis ne sont pas nécessaires. Il faut rappeler que sur la plupart des benchmarks les ViT sont plus efficaces que les ConvNets, mais au prix de quels efforts ?).




## *Bibliographie*

[1] https://arxiv.org/pdf/1512.03385.pdf (papier de recherche introduisant le modèle ResNet).

[2] https://arxiv.org/pdf/2010.11929v2.pdf (papier de recherche introduisant les Vision Transformers).

[3] https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448 (tutoriel d'implémentation de ResNet sous Pytorch)

[4] https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c (tutoriel d'implémentation de ViT sous Pytorch)

[5] https://arxiv.org/pdf/2302.03751.pdf (papier expliquant que les ViT ont des lacunes sur des bases telles que CIFAR-10 par exemple)


