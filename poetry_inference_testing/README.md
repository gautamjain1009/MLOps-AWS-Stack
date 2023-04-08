
* Select an open source project with available pretrained weights. Some examples being, styleGAN, YOLO, SSD, MMPOSE, MMDET, ... etc.
* Convert the weights to onnx (if it is not readily available already)
* Create a poetry package for inference. Setup the poetry project.
* Write a singleton logger and inference implementations using the `__new__` keyword under the class def.
* Customise your project w.r.to modules that work for you.
* Write a main file that handles the flow of different modules.
* Make the codebase ready for local testing with unit tests (using pytest).