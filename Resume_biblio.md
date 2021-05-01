# Résumé de l'article *Playing Atari with Deep Reinforcement Learning*

> Résumé biblio : PDF 2-3 pages sur l'article qu'on a choisi : dequoi il parle ? Il faut un avis critique --> en quoi c'est pratique, en quoi c'est intéressant, qu'est ce que ça coûte ? difficile à manipuler ? difficile à apprendre ?

- [Résumé de l'article *Playing Atari with Deep Reinforcement Learning*](#résumé-de-larticle-playing-atari-with-deep-reinforcement-learning)
  - [Introduction](#introduction)
  - [Présentation de la méthode](#présentation-de-la-méthode)
    - [Deep Reinforcement Learning](#deep-reinforcement-learning)
    - [Avantages](#avantages)
      - [Avantages vis à vis du Deep Learning](#avantages-vis-à-vis-du-deep-learning)
      - [Avantages vis à vis du Q-Learning](#avantages-vis-à-vis-du-q-learning)
    - [Limites](#limites)
  - [Résultats des expériences](#résultats-des-expériences)

## Introduction

L'article [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) traite de l'apprentissage par renforcement sur des jeux Atari. Il a été publié en 2013 par des chercheurs de l'entreprise [DeepMind Technologies](https://deepmind.com/) :

- Volodymyr Mnih
- Koray Kavukcuoglu
- David Silver
- Alex Graves
- Ioannis Antonoglou
- Daan Wierstra
- Martin Riedmiller

Cet article présente donc les résultats d'un apprentissage par renforcement et plus particulièrement d'une variante de Q-Learning sur des jeux [Atari 2600](https://fr.wikipedia.org/wiki/Atari_2600). Ces expériences sont basées sur 7 jeux de l'environnement [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) qui est environnement simple permettant de développer des agents pour de l'intelligence artificielle ou du machine learning sur des jeux de l'Atari 2600.

Le principe de l'apprentissage par renfocement et du Q-learning sont décrits plus détails [ici](README.md).

## Présentation de la méthode

L'objectif est de développer un seul agent capable de jouer à différents jeux Atari. Les entrées du réseau sont les mêmes que celles qu'un joueur humain recevrait, c'est-à-dire l'entrée vidéo, les signaux de récompense et de victoire et l'ensemble des actions possibles. Ainsi, l'architecture du réseau ainsi que les hyper paramètres ne diffèrent pas d'un jeu à un autre.

Le réseau est donc conçu pour être au plus proche de l'être humain.

### Deep Reinforcement Learning

Ce projet a donc pour but d'utiliser de l'apprentissage par renfocrcement. Pour ce faire, les chercheurs ont connecté un algorithme d'apprentissage par renforcement à un réseau neuronal profond qui opère directement sur les images RVB en entrée et qui traite les données d'entraînement par la méthode du gradient stochastique.

La méthode présentée dans cet article est basé sur l'architecture [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon) qui est connue pour être très performante au jeu de Gammon.
Cependant, l'approche est légèrement différente. En effet, il est utilisé le principe d'*experience replay* (qui est très utilise dans Deep Q-Learning). Ainsi, plutôt que d'utiliser les $N$ dernières actions pour prédire la prochaine action, on stocke les expériences de l'agent dans un ensemble de données (que l'on va noter $D$) dans lequel on tirer au hasard les $N$ échantillons qui vont servir à prendre la décision.

L'agent suit une politique $\epsilon$-greedy, c'est à dire qu'il a une probabilité de $1-\epsilon$ de choisir la meilleure action (celle que maximise le gain) et une probabilité $\epsilon$ de choisir une action aléatoire parmis les actions possibles. Une action aléatoire est considérée comme une action *d'explration* tandis qu'une action qui vise à maximiser le gain est une action *d'exploitation*.

### Avantages

Cette méthode, selon ses auteurs, présente de nombreux avantages.  Elle présente notement des avantages par rapport aux autres méthodes de Deep Learning, mais aussi par rapport aux autres approches du Q-Learning.

#### Avantages vis à vis du Deep Learning

#### Avantages vis à vis du Q-Learning

### Limites

Qu'est-ce que ça coûte ?

Difficile à manipuler ?

Difficile à apprendre ?

## Résultats des expériences
