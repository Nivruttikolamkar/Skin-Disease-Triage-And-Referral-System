from roboflow import Roboflow

rf = Roboflow(api_key="9TRBRASkdZDRZaI9m11A")
project = rf.workspace("nivrutti-kolamkar").project("skin-lesion-classification-with-97.8-accuracy-im2or")

# Dataset version download karein (jaise version 1)
version = project.version(1)
dataset = version.download("folder",location="data") # ya "yolov8", etc.

print("Dataset successfully downloaded!")