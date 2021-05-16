# New Super Mario Mario 3D Q

[[_TOC_]]

Projet de Machine Learning pour l'apprentissage de Super Mario Bros par des méthodes de renforcement.

## L'apprentissage par renforcement

L'apprentissage par renforcement est une méthode de Machine Learning très différente des approches classiques. En effet, là où les apprentissages (supervisés ou non) ont besoin de données d'entraînement, les algorithmes de Reinforcement Learning apprennent tout seul à partir de leur environnement.

L'un des exemples les plus connus est l'algorithme AlphaGo développé par Google DeepMind qui a appris de lui-même à jouer au jeu de go qui est un jeu très complexe. En 2016 AlphaGo a même été capable de surpasser de loin Lee Sedol considéré alors comme étant le meilleur joueur au monde du jeu de go (les matchs opposant Lee Sedol à AlphaGo sont visibles [ici](https://www.youtube.com/watch?v=vFr3K2DORc8)).

Les algorithmes de renforcement peuvent donc devenir très performants sur des jeux dont les règles sont très compliquées. On pourrait donc facilement imaginer son application à des problèmes réels.

Comme énoncé précédemment, l'apprentissage par renforcement consiste à apprendre à un agent à se comporter dans un environnement. Il sera récompensé s’il fait une bonne action et pénalisé dans le cas contraire. Ce mode de fonctionnement est très proche de ce que nous faisons dans la vie de tous les jours. Les données d'entraînement proviennent donc de l'environnement. Cet environnement peut être réel ou simulé. Par exemple, dans le cas d'AlphaGo, l'environnement d'entraînement a totalement été recréé virtuellement. Un exemple commun d'apprentissage dans un environnement réel est celui des voitures autonomes (un exemple très parlant d'apprentissage par renforcement dans un environnement réel pour apprendre à conduire à une voiture est visible [ici](https://www.youtube.com/watch?v=eRwTbRtnT1I)). Les environnements virtuels sont en général plus pratiques, car l'apprentissage y est plus aisé et plus rapide, mais ils ne reproduisent pas forcément tous aspects du monde réel. Enfin, le but de l'apprentissage par renforcement n'est pas de minimiser une fonction d'erreur comme dans les méthodes de Machine Learning classiques, mais plutôt de maximiser le nombre de récompenses actuelles et futures.

L'agent est donc plongé dans un environnement et va être amené à prendre des décisions en fonction de cet environnement. À chaque fois que l'agent prend une décision, l'environnement va lui renvoyer un état (le nouvel état de l'environnement après que l'agent ai effectué son action) ainsi qu'une récompense. Cette récompense peut être positive (si l'action est bénéfique), négative (si l'action est néfaste), ou neutre (si l'action n'a pas de répercutions).

Les interractions entre l'agent et l'environnement peuvent être résumées de la sorte :

![Interractions agent-environnement](img/example/agent_environ_interactions.png)

### Exemple simple

Voici un exemple très simpliste du principe de l'apprentissage par renforcement.

On considère un agent : Maro, et un environnement composé d'un Goombass (ennemi) et d'une pièce (objectif à atteindre).

Toute ressemblance avec des personnages ou des situations existantes ou ayant existé (notamment dans l'univers Nintendo) ne saurait être que fortuite.

Voici l'état initial :

![État initial](img/example/initial_state.png)

L'agent peut effectuer 3 actions distinctes :

- attendre
- avancer vers la droite
- sauter

On définit les gains comme suit :

- -1 si Maro touche le Goomba
- 1 si Maro atteint la pièce
- 0 sinon

L'objectif de l'agent est donc d'atteindre la pièce sans toucher le Goombass.

Dans un premier temps, on peut imaginer que l'agent va effectuer l'action "attendre" et donc ne pas bouger de son état initial, et donc obtenir un gain égal à **0**.

On considère ensuite que l'agent va avancer vers la droite :

|   Étape 1                                         |   Étape 2                                        |
|   :---------------------------------------------: |   :--------------------------------------------: |
|   ![Mario running 1](img/example/running_1.png)   |   ![Mario running 2](img/example/running_2.png)  |

Maro va alors rencontrer le Goombass, et perdre une vie. L'environnement va donc lui renvoyer une récompense de **-1**.

Enfin, après avoir essayé les deux actions précédentes et n'ayant pas eu de retours fructueux, l'agent va essayer la troisième action possible, c’est-à-dire "sauter" :

|   Étape 1                                         |   Étape 2                                        |
|   :---------------------------------------------: |   :--------------------------------------------: |
|   ![Mario jumping 1](img/example/jumping_1.png)   |   ![Mario jumping 2](img/example/jumping_2.png)  |

Cette fois-ci, Mario a réussi à atteindre la pièce, il reçoit donc une récompense égale à **1**.

À la suite de ces trois expériences, l'agent va privilégier l'action "sauter" car elle permet de maximiser le gain.

### Exploration vs exploitation

Les algorithmes d'apprentissage par renforcement possèdent deux phases différentes : la phase d'exploration et la phase d'exploitation.

Durant l'exploration, l'agent va prendre des actions aléatoires et noter les gains qu'il reçoit en effectuant ces différentes actions (ce qui s'apparente à la phase d'apprentissage).

Lors de l'exploitation, l'agent va effectuer les actions maximisant son gain à partir de ce qu'il a appris de ses explorations.

Cependant, il faut réussir à trouver un juste milieu entre l'explortion et l'exploitation. En effet, un algorithme ne faisant pas assez d'exploration pourrait se cantonner à une solution sous optimale, tandis qu'un algorithme ne faisant pas assez d'exploitation n'atteindrait jamais la solution.

Pour simplifier les choses, reprenons l'exemple de Maro.

![Exploitation vs exploration 1](img/example/exploit_vs_explor_1.png)

Cette fois-ci, Maro peut obtenir deux gains positifs : un gain de 1 s’il va à gauche et un gain de 3 s’il va à droite.

Si on fixe un nombre d'actions d'exploration petit, on pourrait imaginer que Maro va trouver la pièce de gauche (pour laquelle il lui suffit de se déplacer à gauche), mais ne va jamais atteindre celles de droite (car elles nécessitent plusieurs actions). Ainsi, lors de la phase d'exploitation, Mario va se contenter d'aller à gauche, ne maximisant pas son gain.

|   Solution sous-optimale      |   Solution optimale                                        |
| :--------------------------------------------------: | :--------------------------------------------: |
| ![Sous-optimal](img/example/exploit_vs_explor_2.png) | ![Optimal](img/example/exploit_vs_explor_3.png) |

Ainsi, il existe plusieurs politiques d'exploration/exploitation.

La plus simple et aussi la moins efficace est la politique "greedy". L'agent va simplement choisir l'action maximisant son gain.

Une autre politique bien plus efficace est la politique nommée $`\epsilon`$-greedy. On fixe une valeur $`\epsilon`$ qui va représenter la proportion d'exploration. Par exemple, si on fixe $`\epsilon=0.9`$, on va faire 90% d'exploration et 10% d'exploitation.

Enfin, une autre politique très utilisée est nommée "decaying $`\epsilon`$-greedy". Le principe est exactement le même que pour la politique $`\epsilon`$-greedy, seulement, au bout d'un certain temps, on va faire diminuer petit à petit la valeur de $`\epsilon`$ pour arriver à un stade où l'on fait majoritairement de l'exploitation. Par exemple, on pourrait imaginer qu'on commence avec $`\epsilon=0.9`$ et qu'au bout du 100ᵉ épisode on commence à diminuer cette valeur de $`0.01`$ à chaque épisode jusqu'à ce que $`\epsilon=0.1`$.

### Value function

Pour savoir quelle action doit choisir l'agent en fonction de l'état de son environnement, il faut calculer la valeur de ces états. Cette valeur est calculée grâce à la value function.

La définition de cette value function dépend des algorithmes de Reinforcment Learning.

## L'algorithme Q-Learning

L'algorithme du Q-learning est l'un des algorithmes de renforcement les plus utilisé. Son nom vient de la fonction d'évaluation qui lui es associée : la Q-function.

### La Q-function

La Q-function mesure la qualité d'une action dans un état de l'environnement. Elle prend donc en paramètre l'état, mais aussi l'action que l'agent va effectuer.

Elle est définie de la sorte :

```math
Q(s_t, a_t)^{\pi} = \mathbb{E}[r_{t+1} + \gamma r_{t+2} + \gamma ^2 r_{t+3} + ...|s_t, a_t]
```

Avec :

- $`s_t`$ : l'état de l'environnement à l'instant $`t`$.
- $`a_t`$ : l'action choisie à l'instant $`t`$.
- $`r_{t}`$ : la récompense à l'instant $`t`$.
- $`\gamma \in [0, 1]`$ : un facteur représentant à quel point on va prêter de l'importance aux récompenses sur le long terme (si $`\gamma \approx 1`$ on accorde autant d'importance aux récompenses futures qu'aux récompenses actuelles).
- $`\pi`$ : veut dire que l'agent choisi l'action optimale.

On utilise l'espérance $`\mathbb{E}`$ afin de faire une moyenne, car l'environnement n'est pas forcément déterministe et peut varier au cours des expériences.

La valeur de la Q-function pour une action et un état donné est donc une moyenne des différentes récompenses possibles futures.

### Q-function récursive

On peut redéfinir l'équation précédente de manière récursive avec l'équation de Bellman :

```math
Q(s_t, a_t)^{\pi} = r + \gamma max_{a_{t+1}} Q(s_{t+1}, a_{t+1})^{\pi}
```

Avec :

- $`r`$ : la récompense obtenue en prenant l'action $`a_t`$ dans l'état $`s_t`$.
- $`\gamma`$ : de même que précédement.
- $`max_{a_{t+1}} Q(s_{t+1}, a_{t+1})`$ : la valeur maximale de la Q-function à l'état $`t+1`$ en fonction de l'action.

Cette équation est plus utile car définie de manière récursive.

### Update function

Au cours de l'apprentissage, on veut mettre à jour la valeur de cette Q-function. Pour cela, on utilise donc une fonction d'update qui est définie de la sorte :

```math
Q(s_t, a_t)_{new} = Q(s_t, a_t)_{old} + \alpha [r + \gamma max_{a_{t+1}}Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)_{old}]
```

Avec :

- $`max_{a_{t+1}}Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)_{old}`$ : la différence entre la valeur de la prochaine action (au temps $`t+1`$) et la valeur de l'action actuelle.
- $`\gamma`$ : de même que précédemment.
- $`r`$ : de même que précédemment.
- $`\alpha`$ : le learning rate (afin de pouvoir moyenner cette modification sur plusieurs expériences).

Ainsi, les valeurs de la Q-function sont mise à jour au cours des expériences de manière rétroactive. Une fois qu'un agent a atteint un gain, ce gain va se propager dans les états précédents.

L'algorithme de Q-learning est très pratique et très performant pour résoudre de simples problèmes d'apprentissage par renforcement (sur des jeux simples). En revanche, il n'est pas suffisant pour résoudre des problèmes plus compliqués (comme des jeux vidéos style 8-bits).

### Exemple du Q

On reprend l'exemple ci-dessous avec comme actions possibles "aller à droite", "sauter" et "attendre". Et avec des gains de 1 et -1 respectivement pour la pièce et le Goombass. On choisit arbitrairement $`\gamma=0.9`$

![Initial state](img/example/initial_state.png)

On peut alors imaginer, qu'au terme de la phase d'apprentissage, la valeur de l'action "sauter" sera très proche de 1, car elle permet d'obtenir un gain de 1. De même, la valeur de l'action "aller à droite" aura une valeur proche de -1.

L'action "attendre" en revanche aura une valeur de $`0 + \gamma \times 1 = 0.9`$. En effet, la valeur du gain de l'action "attendre" est de 0. Le gain potentiel maximum de la prochaine action est lui égal à 1 (correspondant à l'action sauter).

### Limitations du Q-Learning

La mise en oeuvre du Q-learning nécessite de construire un tableau associant les différentes transitions possibles ainsi que l'espérance des récompenses associées. En effet on estime la Q-function en actualisant le tableau à chaque itération à l'aide de l'équation de _Bellman_.

Cependant lorsque l'environnement devient trop complexe les nombre de transitions possibles augmente considérablement. Ainsi construire un tableau devient contre-productif puisqu'il serait trop lourd à stocker en mémoire.

C'est pour cela qu'une amélioration de la méthode a été proposée pour pouvoir s'appliquer à des environnement plus complexes.

## Le Deep Q-Learning

Dans l'[article suivant](https://arxiv.org/pdf/1312.5602.pdf) une méthode de deep learning est proposée pour pouvoir pallier au principal problème du Q-learning classique cité précédemment pour pouvoir faire jouer un agent à des jeux videos (environnement complexes).

### Principe général

L'idée présentée dans cette article se base sur la combinaison du principe du Q-learning avec la puissance des réseaux de neurones dans le but de faire jouer un agent à des jeux Atari.

Ainsi, l'objectif ici est de pouvoir approximer la Q-function avec un réseau de neurone.

### Problème à résoudre

En entrée du réseau nous avons l'état actuel du jeu et nous souhaitons qu'en sortie il nous soit renvoyé l'espérance des récompenses qu'il est possible d'obtenir par action que l'agent peut effectuer dans l'environnement. En somme nous voulons estimer le retour de la Q-function.

L'objectif de l'agent est donc d'intéragir avec un environnement (ici l'émulateur de jeu) en sélectionnant des actions pour maximiser les récompenses qu'il peut obtenir. Nous définissons $`Q^*(s_{t}, a_{t})`$ la valeur maximale des récompenses futures espérées en suivant n'importe quelle stratégie (en choisissant n'importe quelle action) après avoir pris connaissance de l'état de l'environnement $`s`$ pour y appliquer une action $`a`$. Ainsi :

```math
Q^*(s_{t}, a_{t}) = \max_\pi \mathbb{E}[R_t | s_t = s, a_t = a, \pi]
```

avec $`R_t`$ les récompenses futures.

Ainsi, nous nous basons sur l'équation de _Bellman_ pour construire notre objectif. Si la valeur optimale $`Q^*(s_{t+1}, a_{t+1})`$ à l'état $`s_{t+1}`$ (état suivant) est connue pour toutes les actions $`a_{t+1}`$ possibles alors la stratégie optimale est de sélectionner l'action $`a_{t+1}`$ qui maximise la valeur des récompenses espérées : $`r+\gamma Q^*(s_{t+1}, a_{t+1})`$ avec $`\gamma`$ le coefficient d'actualisation des récompenses futures.

```math
Q^*(s_{t}, a_{t}) = r + \gamma \max_{a_{t+1}} Q^*(s_{t+1}, a_{t+1})
```

Nous utilisons donc un réseau de neurone que nous appelerons Q-network pour servir d'estimateur de cette fonction. La fonction de perte est définie de la manière suivante :

```math
L_i (\theta_i) = (y_i - Q(s, a; \theta_i))^2
```

avec $`y_i`$ la *target* telle que :

```math
y_i = r + \gamma \max_{a_{t+1}} Q^*(s_{t+1}, a_{t+1}; \theta_{i-1})
```

et notre *feature*:

```math
x_i = Q(s, a; \theta_i)
```

avec $`\theta`$ faisant référence aux poids du réseau de neurones.

### Algorithme

L'algorithme présenté par *deepmind* illustrant la procédure d'apprentissage de l'agent est le suivant :

![algorithme](img/example/algorithme_deep_q_learning.png)

Ici, le principe d'**experience replay** est utilisé pour constituer une base de données de transitions. Lors de l'apprentissage à chaque pas effectué par l'environnement la transition entre l'état actuel et l'état suivant est stockée dans un buffer et à chaque pas d'apprentissage le modèle va apprendre sur un batch de transitions échantillonné depuis ce buffer. Cela permet d'éviter les dépendances fortes entre les données successives fournies au modèle pour apprendre.

### Limitations

Le principal problème du Deep Q-learning réside dans sa procédure d'apprentisage très instable. En effet, le même modèle d'estimation de la Q-function est utilisé pour, à la fois générer les *features* et les *targets* utilisées par le modèle pour apprendre. Cela signifie que la fonction de perte change à chaque itération. De plus, avec cette méthode le réseau a tendance à surestimer les valeurs estimées (à cause notamment de l'utilisation de l'opérateur **max**) ce qui peut mener l'agent à s'enfermer plus facilement dans des comportements optimaux localement.

## Le Double Deep Q-Learning

Afin de pallier aux problèmes engendrés par le deep Q-learning, on peut utiliser une amélioration de cette méthode : le Double Deep Q-learning. Son principe est détaillé dans l'article [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf).

### Le Double Q-Learning

Le Double Deep Q-Learning s'inspire d'une méthode appellé le Double Q-Learning. Le principe du Double Q-learning consiste simplement à utiliser deux Q-functions. Ainsi, à chaque épisode, les poids d'une des deux fonction (choisie de manière aléatoire) vont être mis à jour, créant deux jeux de poids notés $`\theta`$ et $`\theta'`$.

Les target seront donc de la forme :

```math
y_t^{DoubleQ} = R_{t+1} + \gamma Q\left(S_{t+1}, \max_a Q\left(S_{t+1}, a; \theta_t\right); \theta_t' \right)
```

On retrouve bien ici deux Q-functions :

- Une première fonction (celle associé au modèle $`\theta`$) permet de sélectionner l'action optimale pour un état donnée. On dit que c'est la fonction _online_.
- Une seconde fonction (celle associé au modèle $`\theta'`$) permet de calculer les récompenses attendues par l'utilisation de cette action pour le même état. On dit que c'est la fonction _target_.

Cette séparation permet d'obtenir une évaluation plus stable, en limitant la surévaluation des récompenses.

Cette méthode n'avait été essayé que pour des Q-functions et pas des Q-network.

### Application du Double Q-Learning pour le Double Deep Q-Learning

Le Double Deep Q-Learning utilise les avantages du Deep Q-Learning, tout en rendant l'apprentissage plus stable en utilisant le Double Q-Learning.

Le principe est ici le même que pour le Double Q-Learning. Au lieu d'utiliser un seul Q-network, on utilisera deux Q-network : un _online_ pour le choix de la meilleure action et un _target_ pour le calcul des valeurs de récompenses.

La principale différence avec le Double Q-Learning, est l'apprentissage des deux réseaux. Là ou dans le Double Q-Learning, les deux fonctions jouaient un rôle symétrique et interchangeable, le Double Deep Q-Learning sépare l'apprentissage des deux réseaux :

- Le réseau _online_ est mis à jour à chaque itération.
- Le réseau _target_ est une copie du réseau _online_, toutes les `$\tau$` itérations. Ce réseau correspond donc à une version plus ancienne du réseau _online_.

Cette séparation permet d'améliorer la stabilité de l'apprentissage du Q-network.

## Présentation de nos expérimentations

### Présentation de l'environnement

L'environnement utilisé est disponible à cette adresse : [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/)

Cette environnement implémente l'interface `Environnement` de la bibliothèque [gym](https://gym.openai.com/). Pour faire progresser l'environnement la fonction `step` est utilisée (on lui envoie en paramètre une action à réaliser). Cette fonction renvoie ainsi :

- **state** : le nouvel état du jeu, ici c'est une image du jeu en RGB.
- **reward** : les récompenses obtenues suite à l'action effectuée.
- **done** : booléen indiquant si le nouvel état est terminal ou non (si le niveau est terminé ou bien si mario a été tué).
- **infos** : un dictionnaire contenant des informations sur le déroulement du jeu telles que,
  - **coins** : les pièces collectées par l'agent.
  - **flag_get** : booléen indiquant si le drapeau ou la hache ont été atteint (si l'agent a complété le niveau).
  - **life** : le nombre de vies restantes de mario.
  - **score** : le score actuel de l'agent.
  - **stage** : l'identifiant du niveau.
  - **status** : le status de mario (si il est petit, grand ou en fleur de feu).
  - **time** : le temps ingame.
  - **world** : l'identifiant du monde.
  - **x_pos** : la position en x de mario sur le niveau.
  - **y_pos** : la position en y de mario sur le niveau.

#### La politique de récompense

Une politique de récompense par défaut est définie par l'environnement :

##### La vélocité

$`v`$ : la différence entre la position en x de mario à l'état initial et à l'état suivant selon la relation $`v = x_{t+1} - x_{t-1}`$.

Ainsi lorsque mario :

- va vers la droite (vers la fin du niveau) : $`v > 0`$
- va vers la gauche : $`v < 0`$
- ne bouge pas : $`v = 0`$

Cette composante de la fonction de récompense a pour objectif de stimuler le déplacement de l'agent vers la fin du niveau.

##### La composante temporelle

$`c`$ : la différence entre le temps ingame à l'état initial et à l'état suivant selon la relation $`c = c_{t+1} - c_{t-1}`$.

Cette composante de la fonction de récompense a pour objectif d'imposer une contrainte de temps à l'agent pour pouvoir finir le niveau. Ainsi, l'agent est encouragé à finir le niveau le plus rapidement possible.

##### La peine de mort

$`d`$ : la pénalité appliquée à l'agent s'il meurt.

Ainsi lorsque mario :

- est vivant : $`d = 0`$
- meurt : $`d = -15`$

##### Total

L'expression finale de la fonction de récompense est la suivante :

```math
r = v + c + d
```

### Architecture du modèle

Notre objectif est de pouvoir calculer l'espérance des récompense futures à partir d'un état du jeu.

Comme préconisé dans [insérer nom d'article](todo) nous fournirons à notre modèle 4 (nombre pouvant varier) images en nuances de gris de taille $`84 \times 84`$ empilées (images allant des temps $`t-3`$ à $`t`$).

Afin de pouvoir analyser ces images nous utilisons $3$ couches de convolution et nous complétons l'architecture avec $1$ couche dense.

Finalement nous ressortons un vecteur indiquant l'espérance des récompenses futures pour chaque action possible.

![model](img/example/model.png)

### Preprocessing

Nous utiliserons les wrappers recommandés par *deepmind* pour apprendre sur des jeux atari :

- MaxAndSkip
- WarpFrame
- FrameStack
- ScaledFloatFrame

### Deep Q Learning

### Double Deep Q Learning

Ici nous utiliserons en plus le wrapper : `ClipReward`.
La récompense totale obtenue par l'agent en cas de victoire sur le niveau 1-1 est environ égale à 300.

![training](img/example/training_average_morio_max_ep_99999775.png)

#### Nouvelle politique de récompense

Ici nous utiliserons en plus le wrapper : `CustomReward` à la place de `ClipReward`.
La récompense totale obtenue par l'agent en cas de victoire sur le niveau 1-1 est environ égale à 300.

![training](img/example/training_average_morio_custom_rewards.png)

### Critique sur le travail réalisé

### Graphiques annexes

#### Histogrammes de densité des récompenses

##### Avec ClipReward

**Epsilon** = 0.99999775

![training](img/example/density_hist_max_ep_99999775.png)

##### Avec CustomReward

**Epsilon** = 0.999999

![training](img/example/density_hist_custom_rewards.png)
