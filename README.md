# Final Project: Computing Fundamentals
## NeuralGen-HH for the Firefighter Problem

> Based on: https://github.com/jcobayliss/FFPHHS

Steps to setup the environment:
- Activate conda: `source ~/miniforge3/bin/activate`
- Create new virtual environment and activate it: `conda activate venv`
- Install dependencies specified in `requirements.txt` file.

Run the program:
1. User can either choose between the following two models to predict the next heuristic to apply: *Feed Forward Neural Network* and *Decision Tree*. To do so the user needs to change the `model_name` variable, in line number 440, to either `NN` or `DT`, respectively.
2. For training from scratch the models (with the Genetic Algorithm output), only delete either the `neuralgen.h5` or `decision_tree_model_joblib.pickle` files.
3. As we provide, the program runs only on one instance, but uncomment from the lines 472 up to 530, and comment from the lines 442 up to 470, to run with all the instances available (i.e., 360).
4. The results are exported in the corresponding csv file: either `results_neural_network.csv` or `results_decision_tree.csv`; the output generated by our Genetic Algorithm (i.e., the conditions and actions to train our AI models), are in the folder named `out`.
6. Our main file is `ffp.py`, so you can run it and see our Hyper-Heuristic working!