# Fractal AI with Visualization and Interaction

## Description
This project implements an advanced Fractal AI system with dynamic growth, visualization, and interactive features. It combines concepts from fractal geometry, neural networks, and Hebbian learning to create a unique and evolving AI structure.

**Project Video Demonstration:**

[![Fractal AI Project Video](https://img.youtube.com/vi/M1mV1a5Te44/0.jpg)](https://www.youtube.com/watch?v=M1mV1a5Te44)

Project space at hugginface: 

[FractalAI](https://huggingface.co/spaces/Aluode/FractalAI)


## Features
- Dynamic fractal network growth
- 3D visualization of the fractal AI structure
- Hebbian learning for connection weight updates
- Interactive chat functionality
- Wikipedia integration for training
- Self-conversation mode
- State saving and loading
- Zoom functionality for detailed exploration

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/anttiluode/fractalai.git
   cd fractal-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```
python app.py
```

This will launch a Gradio interface in your default web browser, where you can interact with the Fractal AI system.

## Interface Options

- **Run Cycles**: Execute a specified number of thinking cycles
- **Train on Wikipedia**: Input a topic to train the AI on Wikipedia content
- **Chat**: Engage in a conversation with the AI
- **Self-Conversation**: Let the AI converse with itself
- **Zoom**: Explore the fractal structure in detail
- **Save/Load State**: Preserve or restore the AI's state

## AppV2.py: Advanced Fractal Neural Network Implementation

AppV2.py represents a significant upgrade from the initial version, introducing several sophisticated features and improvements:

### Key Enhancements in AppV2.py

1. **Fractal Neuron Structure**:
   - Implements a `FractalNeuron` class with advanced properties including position, connections, weights, and gradients.
   - Introduces a more complex activation function with bias.

2. **Adam Optimizer Integration**:
   - Incorporates an `AdamOptimizer` class for more efficient weight updates.
   - Includes momentum and adaptive learning rates for improved convergence.

3. **Enhanced FractalNeuralNetwork Class**:
   - Implements sophisticated methods for adding words, connecting neurons, and managing the network structure.
   - Introduces asynchronous learning from Wikipedia topics.

4. **Advanced Training Mechanisms**:
   - Implements backpropagation with dropout for improved generalization.
   - Includes a forward pass method with dropout support.
   - Introduces an attention mechanism for context-aware processing.

5. **Improved Data Handling**:
   - Implements methods to fetch and process training data from external APIs.
   - Includes tokenization and text preprocessing capabilities.

6. **Sophisticated Response Generation**:
   - Utilizes an attention-based mechanism for generating responses.
   - Implements temperature-controlled sampling for varied outputs.

7. **Asynchronous Operations**:
   - Utilizes `asyncio` and `aiohttp` for efficient, non-blocking Wikipedia content fetching.

8. **Advanced Visualization**:
   - Implements a 3D visualization of the network structure using matplotlib and networkx.

9. **Robust State Management**:
   - Includes methods for saving and loading the entire network state, including optimizer parameters and RNG state.

10. **Gradio Interface Enhancements**:
    - Provides a more comprehensive interface with tabs for initialization, training, visualization, chatting, and state management.
    - Includes detailed instructions and warnings for users.

### Key Differences from the Initial Version

- **Complexity**: AppV2 implements a much more complex and theoretically grounded neural network structure.
- **Learning Approach**: Introduces sophisticated learning methods including backpropagation and attention mechanisms.
- **Scalability**: Designed to handle larger, more complex networks with efficient training and optimization.
- **Interactivity**: Offers a more comprehensive and user-friendly interface for interacting with the network.
- **Visualization**: Provides advanced 3D visualization of the network structure.
- **Persistence**: Implements more robust methods for saving and loading network states.

## Contributors
- Antti Luode - Original concept and ideation
- ChatGPT - Assisted in code generation and problem-solving
- Claude (Anthropic) - Implemented core functionality and resolved issues

## Acknowledgements
Special thanks to Antti Luode for the innovative and ambitious idea behind this project. The collaboration between human creativity and AI assistance has made this unique project possible.

## License
This project is open-source and available under the MIT License.
