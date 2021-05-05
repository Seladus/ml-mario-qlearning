# New Super Mario Mario 3D Q

Projet de Machine Learning pour l'apprentissage de Super Mario Bros par des méthodes de renforcement.

## L'apprentissage par renforcement

L'apprentissage par renforcement est une méthode de Machine Learning très différente des approches classiques. En effet, là où les apprentissages supervisés ou non ont besoin de données d'entraînement, les algorithmes de Reinforcement Learning apprennent tout seul à partir de leur environnement.

L'un des exemples les plus connus est l'algorithme AlphaGo développé par Google DeepMind qui a appris de lui-même à jouer au jeu de go qui est un jeu très complexe. En 2016 AlphaGo a même été capable de surpasser de loin Lee Sedol considéré alors comme étant le meilleur joueur au monde du jeu de go (les matchs opposant Lee Sedol à AlphaGo sont visibles [ici](https://www.youtube.com/watch?v=vFr3K2DORc8)).

Les algorithmes de renforcement peuvent donc devenir très performants sur des jeux dont les règles sont très compliquées. On pourrait donc facilement imaginer son application à des problèmes réels.

Comme énoncé précédemment, l'apprentissage par renforcement consiste à apprendre à un agent à se comporter dans un environnement, il va être récompensé s’il fait une bonne action et pénalisé dans le cas contraire. Ce mode de fonctionnement est très proche de ce que nous faisons dans la vie de tous les jours. Les données d'entraînement proviennent donc de l'environnement. Cet environnement peut être réel ou simulé. Par exemple, dans le cas d'AlphaGo, l'environnement d'entraînement a totalement été recréé virtuellement. Un exemple commun d'apprentissage dans un environnement réel est celui des voitures autonomes (un exemple très parlant d'apprentissage par renforcement dans un environnement réel pour apprendre à conduire à une voiture est visible [ici](https://www.youtube.com/watch?v=eRwTbRtnT1I)). Les environnements virtuels sont en général plus pratiques, car l'apprentissage y est plus aisé et plus rapide, mais ils ne reproduisent pas forcément tous aspects du monde réel. Enfin, le but de l'apprentissage par renforcement n'est pas de minimiser une fonction d'erreur comme dans les méthodes de Machine Learning classiques, mais plutôt de maximiser le nombre de récompenses actuelles et futures.

L'agent est donc plongé dans un environnement et va être amené à prendre des décisions en fonction de cet environnement. À chaque fois que l'agent prend une décision, l'environnement va lui renvoyer un état (le nouvel état de l'environnement après que l'agent ai effectué son action) ainsi qu'une récompense. Cette récompense peut être positive (si l'action est bénéfique), négative (si l'action est néfaste), ou neutre (si l'action n'a pas de répercutions).

### Exemple simple

Voici un exemple très simpliste du principe de l'apprentissage par renforcement.

On considère un agent : Mario, et un environnement composé d'un Goomba (ennemi) et d'une pièce (objectif à atteindre).

Toute ressemblance avec des personnages ou des situations existantes ou ayant existé (notamment dans l'univers Nintendo) ne saurait être que fortuite.

Voici l'état initial :

![État initial](img/example/initial_state.png)

L'agent peut effectuer 3 actions distinctes :

- attendre
- avancer vers la droite
- sauter

On définit les gains comme suit :

- -1 si Mario touche le Goomba
- 1 si Mario atteint la pièce
- 0 sinon

L'objectif de l'agent est donc d'atteindre la pièce sans toucher le Goomba.

Dans un premier temps, on peut imaginer que l'agent va effectuer l'action "attendre" et donc ne pas bouger de son état initial, et donc obtenir un gain égal à **0**.

On considère ensuite que l'agent va avancer vers la droite :

|   Étape 1                                         |   Étape 2                                        |
|   :---------------------------------------------: |   :--------------------------------------------: |
|   ![Mario running 1](img/example/running_1.png)   |   ![Mario running 2](img/example/running_2.png)  |

Mario va alors rencontrer le Goomba, et perdre une vie. L'environnement va donc lui renvoyer une récompense de **-1**.

Enfin, après avoir essayé les deux actions précédentes et n'ayant pas eu de retour fructueux, l'agent va essayer la troisième action possible, c’est-à-dire "sauter" :

|   Étape 1                                         |   Étape 2                                        |
|   :---------------------------------------------: |   :--------------------------------------------: |
|   ![Mario jumping 1](img/example/jumping_1.png)   |   ![Mario jumping 2](img/example/jumping_2.png)  |

Cette fois-ci, Mario a réussi à atteindre la pièce, il reçoit donc une récompense égale à **1**.

À la suite de ces trois expériences, l'agent va privilégier l'action "sauter" car elle permet de maximiser le gain.

### Exploration vs exploitation

### Value function

## L'algorithme Q-Learning

## Le Deep Q-Learning

## Le Double Deep Q-Learning

## Présentation de nos expérimentations
