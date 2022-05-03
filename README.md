# Random Graphs

By [kevinchangjk](https://github.com/kevinchangjk)

This is the result of 3 months of research and study on random graphs, as part of an internship at the Nanyang Technological University, School of Physical and Mathematical Sciences, mentored by Dr Fedor Duzhin.

## Background

In mathematics, graphs are an arrangement of a set of nodes and a set of vertices between the nodes. A random graph is a graph that is generated through some sort of probabilistic method. There are many methods, algorithms, and thus models.

In this project, I looked into some of these models, to see is I could model a social network using one of these. I used a sample of Facebook's network as a representation of a social network, and wrote my programs in Python and Jupyter Notebook for reports.

## Work

To start with, I studied various models, writing out an implementation myself, for greater modifiability. With these models, I tried to generate random graphs of similar properties to the Facebook sample, though most didn't even come close.

Some of the models I studied include:

- Erdos-Renyi
- Watts-Strogatz
- Barabasi-Albert
- Bianconi-Barabasi
- Klemm-Eguiluz
- Preferential Block
- Static Block
- Random Geometric Graph
- Clustering Distribution

After studying these, I made an attempt at creating my own model to achieve the desired properties of high transitivity, high clustering coefficient, scale-free degree distribution, and low diameter. It resulted in a model taking inspiration from many of the models mentioned above, and I called it the "Chang-Chang" model, as it was in collaboration with another intern surnamed Chang.

The implementation of the models, and other properties and functions, are located in the `models` directory. The `Juypyter` directory contains my work on Jupyter Notebooks, and the `graph-data` directory contains generated graphs and relevant data for my reports.

## Reports

For a more streamlined way of understanding the work I did, it might be better to look at the `slides` directory, which contains some slides I had made for my final report at the end of the internship.
