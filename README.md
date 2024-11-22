# GDVisualizer
## Install requirements
Install all essential packages: ```pip install -r reqs.txt```
## Tutorials
Run ```python main.py``` to run the app.

DearPyGui Controller:
- Optimizer: choose an optimizer to run the algorithm. 6 Options: 5 different optimizers and the option "all" will run all optimizers simuteniously.
- Learning rate: learning rate for all optimizers.
- Alpha: alpha parameter for RMSprop.
- Beta1: the first beta parameter for Adam and AdamW.
- Beta2: the second beta parameter for Adam and AdamW.
- Epsilon: epsilon parameter for all optimizers except SGD.
- Weight decay: weight decay rate for all optimizers.
- Momentum: momentum parameter for SGD and RMSprop.
- Coordinate size: the size of the mesh. Increase this number will make the mesh larger.
- Initial x, initial y: the starting point of the ball.
- FPS: Frame per second.
- Speed: the speed of the ball.

Interact with GL window:
- Press w to change the mode of visualizer: full mesh, triangle mesh, set of points.
- Use left mouse to rotate the viewer.
- Use right mouse to change the reference point.
- Scroll to zoom-in, zoom-out.