# Multimodal Fractal AI Model

## Description

The **Multimodal Fractal AI Model** is an experimental deep learning framework designed to process and integrate multiple data modalities, including text, audio, and video. The architecture employs a fractal approach, utilizing branches and leaves to dynamically grow and prune based on resource usage and importance metrics. 

It is based on the idea that human brain / AI may store information as fractals. 

It is mostly broken code that most likely will never be done written by ChatGPT / Claude. 

## Features
- **Fractal Architecture**: Dynamic branching structure for processing different modalities.
- **Multimodal Integration**: Combines text, audio, and video inputs for enriched AI responses.
  (these have not been tried) 
- **Memory Management**: Adapts to available GPU memory and can prune or grow branches accordingly.
- **Interactive Training**: Train the model with custom prompts through a user-friendly interface.

## Installation
To run the Multimodal Fractal AI Model locally, follow these steps:

1. Clone the repository:
   
   git clone https://github.com/anttiluode/fractalai.git

   cd fractalai
   
Install the required dependencies:

pip install -r requirements.txt

Ensure you have the necessary software:

Python: Version 3.7 or higher
PyTorch: Make sure to install a version compatible with your CUDA setup.
Librosa: For audio processing
OpenCV: For video processing
Gradio: For building the interactive interface

Usage

Run the application:

python main.py

Access the web interface at http://127.0.0.1:7860.

Interact with the model:

Train the model with custom prompts.

Train it with LM Studio - start a server and click on start training. LM Studio will start talking with the model 
and perhaps it will learn something. Do not expect intelligent output. 

Ideally it would be able to handle video that is sent to it in very small resolution and audio. . Like a babby. 
You know. A lil android. Interesting is that it will grow and prune leaves and brances somehow. According to what 
ever mumbo jumbo claude and o1 model put in to the code.. 

Oh and if it was all working.. You might be able to generate creative dreams based on trained data.

Known Issues

It is mostly a feverish dream. Autistic obsession. 

License

This project is licensed under the MIT License. See the LICENSE file for details.
