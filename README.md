This repository is for the capstone project with zaka.ai academy.

In this project we used three models, one for gender detection, one for face swapping and one for voice cloning.

The gender model is build with YOLOv10 pre-trained model after fine tuning it.
You can see a script of testing it here, you just need to specify the image path when it requests it.


The face swapping model used SimSwap model after fine tuning it.

You can also see a script of the two models pipeline where the gender model suggests the set of videos related to the user's detected gender. After that the user will choose one video and the SimSwap (our trained model) will start working.

How to use:
First clone the repository and install all requirements, then check GUIDANCE to perform a good setup

We uploaded two files, "pipeline" for google colab interface, "Flask.app" for windowsbased interface.

NOTE:

