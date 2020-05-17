# Habitat Supervised Learning Baselines

### Detailed report regarding this work can be accessed here: [Supervised Learning Benchmarks for Point Goal Navigation in Photo-realistic indoor cluttered](https://zubairirshad.com/portfolio/supervised-learning-benchmarks-for-embodied-visual-navigation-in-habitat/)

## Behavior Cloning

#### Installation

Install Habitat-api and Habitat-sim

Simlink to create the following structure
```
.
+- vln-instruction-generation
   +- config
   +- data
   |  +- datasets
   |  |  +- pointnav
   |  |     +- mp3d
   |  |        +- v1
   |  |           +- test
   |  |           +- train
   |  |           +- val
   |  +- scene_datasets
   |     +- mp3d
   |        +- 1LXtFkjw3qL
   |        +- 1pXnuDYAj8r
   |        +- ...
   |        +- zsNo4HB9uLZ
   +- semantic-path.py
```

#### Data Pre-processing
Run the following script to make data batches. Change source path and destination path in make_batches.py

```
python make_batches.py
```


#### DataLoader for training

```
expert.py #provides various functions to get optimal shortest path for any datapoint
```

#### Train 
```
python train.py
```

#### Hyperparameters 

![Hyperparameters](https://github.com/zubair-irshad/habitat_imitation_learning/blob/master/logger/hyperparameters.png)

#### Preliminary Results
![Resukts](https://github.com/zubair-irshad/habitat_imitation_learning/blob/master/logger/plots.jpg)


#### Future Work 
- Implement [imitation learning pipeline](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf)
- Evaluating approach in real-time simulation in habitat-api and tracking various metrics including [SPL](https://arxiv.org/abs/1807.06757)

