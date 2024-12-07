# Interactive-room
The aim of this project is to have a fully emerged experience where the user can see themselves in  any movie scene, and not only the face, but also the scenes lines will be spelled by the user's voice! It has a google colab to try it, we attached some movie scenes for each gender.

(It is only for use, it cannot be trained)


# Model
This project consists of three models:

1- The first one is the gender model where it will detect the user's gender and proposes movie scenes based on it.

2- The second model is the SimSwap model where it will detect the face, swap it and align it on the scene.

3- The voice model will have to clone the user's voice so the words of the actor will be said by the user's voice.

# Process
We firstly trained the gender model using Yolov10s pre-trained model, then we tested the SimSwap model but we considered training it for better result and finally we used Coquii TTS model for voice cloning.

# Guidance (SimSwap Setup)
To begin, we have to set up the SimSwap environement, since we already have our gender model within the repository. To use it, you have to first Follow the **Guidance** file and then keep up with the rest.

- You can see that we have a flaskapp, this is only for Gender and SimSwap model and does not include voice cloning.
+ We also provided a script to test our Gender model alone. And the Google Colab file is for the interface of the whole project.

# Usage
To start using this project simply do the following
1. Clone the repository
2. Follow the *guidance* file to setup the simswap
3. Use the notebook.

## Note
The google colab notebook is ready to use, only you should download it and run the cells, no need to do coding, you will have your gradio interface.

This project is not trainable.

# Result
We provided a result for Arabic scene, there are 4 by 4 arabic scenes for male and female to try them out.
