# Interactive-room
The aim of this project is to have a fully emerged experience where the user can see themselves in  any movie scene, and not only the face, but also the scenes lines will be spelled by the user's voice! It has a google colab to try it, we attached some movie scenes for each gender.

(It is only for use, it cannot be trained)


# Model
This project consists of three models:

1- The first one is the gender model where it will detect the user's gender and proposes movie scenes based on it.

2- The second model is the SimSwao model where it will detect the face, swap it and align it on the scene.

3- The voice model will have to clone the user's voice so the words of the actor will be said by the user's voice.

# Process
We firstly trained the gender model using Yolov10s pre-trained model, then we tested the SimSwap model but we considered training it for better result and finally we used Coquii TTS model for voice cloning.

# Guidance
To begin, we have to set up the SimSwap environement, since we already have our gender model. To use it, you have to first Follow the Guidance file and then keep up with the rest.
