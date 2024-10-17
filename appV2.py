import numpy as np
import random
import requests
from bs4 import BeautifulSoup
import re
import json
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import time
from PIL import Image  # Added for image handling
import asyncio
import aiohttp
from tqdm import tqdm  # For progress visualization

# Helper functions for serialization
def convert_ndarray_to_list(obj):
    """
    Recursively convert all ndarray objects in a nested structure to lists.
    """
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def convert_list_to_ndarray(obj):
    """
    Recursively convert all lists in a nested structure back to ndarrays where appropriate.
    """
    if isinstance(obj, dict):
        return {k: convert_list_to_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Attempt to convert lists of numbers back to ndarrays
        try:
            return np.array(obj)
        except:
            return [convert_list_to_ndarray(item) for item in obj]
    else:
        return obj

class FractalNeuron:
    def __init__(self, word, position):
        """
        Initialize a neuron with a given word and position in the space.
        """
        self.word = word
        self.position = position
        self.connections = {}  # Connections to other neurons {word: neuron}
        self.activation = np.random.uniform(-0.1, 0.1)  # Random initial activation
        self.bias = np.random.uniform(-0.1, 0.1)        # Random bias
        self.gradient = 0.0
        self.weights = {}  # Weights of connections {word: weight}
        self.time_step = 0.01  # Small step size for Euler's method
        self.gradients = {}   # Gradients for each connection

    def activate(self, input_signal):
        """
        Update the neuron's activation based on the input signal.
        """
        # Ensure input_signal is a scalar
        if isinstance(input_signal, np.ndarray):
            input_signal = np.mean(input_signal)
        
        # Update activation using activation function with bias
        self.activation = np.tanh(input_signal + self.bias)
        
        # Ensure activation remains a scalar float
        if isinstance(self.activation, np.ndarray):
            self.activation = float(np.mean(self.activation))
        
        # Debugging
        print(f"Neuron '{self.word}' activation after update: {self.activation}")

    def connect(self, other_neuron, weight):
        """
        Establish a connection to another neuron with a specified weight.
        """
        self.connections[other_neuron.word] = other_neuron
        self.weights[other_neuron.word] = weight


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0001):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, network):
        """
        Update the network's weights using Adam optimization.
        """
        self.t += 1
        for word, neuron in network.neurons.items():
            for connected_word, weight in neuron.weights.items():
                grad = neuron.gradients.get(connected_word, 0.0) + self.weight_decay * weight
                if word not in self.m:
                    self.m[word] = {}
                if connected_word not in self.m[word]:
                    self.m[word][connected_word] = 0.0
                if word not in self.v:
                    self.v[word] = {}
                if connected_word not in self.v[word]:
                    self.v[word][connected_word] = 0.0
                # Update biased first moment estimate
                self.m[word][connected_word] = self.beta1 * self.m[word][connected_word] + (1 - self.beta1) * grad
                # Update biased second raw moment estimate
                self.v[word][connected_word] = self.beta2 * self.v[word][connected_word] + (1 - self.beta2) * (grad ** 2)
                # Compute bias-corrected first moment estimate
                m_hat = self.m[word][connected_word] / (1 - self.beta1 ** self.t)
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[word][connected_word] / (1 - self.beta2 ** self.t)
                # Update weights
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                neuron.weights[connected_word] += update


class FractalNeuralNetwork:
    def __init__(self, space_size=10, seed=None):
        """
        Initialize the Fractal Neural Network.
        """
        self.neurons = {}
        self.space_size = space_size
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = {}  # First moment vector (mean) for Adam optimizer
        self.v = {}  # Second moment vector (variance) for Adam optimizer
        self.t = 0   # Timestep for Adam optimizer
        self.rng = np.random.default_rng(seed)
        self.optimizer = AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                       beta2=self.beta2, epsilon=self.epsilon, weight_decay=0.0001)

    def tokenize_text(self, text):
        # Convert to lowercase and split on whitespace
        tokens = text.lower().split()
        # Optional: Remove any remaining punctuation
        tokens = [token.strip('.,!?:;()[]{}') for token in tokens]
        # Remove any empty tokens
        tokens = [token for token in tokens if token]
        return tokens

    def add_word(self, word):
        """
        Add a word as a neuron to the network if it doesn't already exist.
        """
        if word not in self.neurons:
            position = self.rng.random(3) * self.space_size
            self.neurons[word] = FractalNeuron(word, position)
            return f"Added word: '{word}'."
        else:
            return f"Word '{word}' already exists in the network."

    def connect_words(self, word1, word2):
        """
        Connect two words in the network with a randomly initialized weight.
        """
        if word1 not in self.neurons:
            return f"Word '{word1}' does not exist in the network."
        if word2 not in self.neurons:
            return f"Word '{word2}' does not exist in the network."
        weight = self.rng.normal()
        self.neurons[word1].connect(self.neurons[word2], weight)
        # Initialize optimizer moments for the new connection
        if word1 not in self.optimizer.m:
            self.optimizer.m[word1] = {}
        if word2 not in self.optimizer.m[word1]:
            self.optimizer.m[word1][word2] = 0.0
        if word1 not in self.optimizer.v:
            self.optimizer.v[word1] = {}
        if word2 not in self.optimizer.v[word1]:
            self.optimizer.v[word1][word2] = 0.0
        return f"Connected '{word1}' to '{word2}' with weight {weight:.4f}."

    async def fetch_wikipedia_content_async(self, session, topic):
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.text for p in paragraphs])
                    return topic, content
                else:
                    print(f"Failed to fetch {topic}: Status {response.status}")
                    return topic, None
        except Exception as e:
            print(f"Exception fetching {topic}: {e}")
            return topic, None

    async def learn_from_wikipedia_async(self, topics, concurrency=5):
        """
        Asynchronously learn from Wikipedia articles with controlled concurrency.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for topic in topics:
                task = asyncio.ensure_future(self.fetch_wikipedia_content_async(session, topic))
                tasks.append(task)
            responses = await asyncio.gather(*tasks)
            
            results = []
            for topic, content in responses:
                if content:
                    tokens = self.tokenize_text(content)
                    for token in tokens:
                        self.add_word(token)
                    for i in range(len(tokens) - 1):
                        self.connect_words(tokens[i], tokens[i + 1])
                    results.append(f"Learned from Wikipedia article: {topic}")
                else:
                    results.append(f"Failed to fetch content for: {topic}")
            return "\n".join(results)

    def fetch_training_data(self, num_sequences=100, seq_length=5):
        training_data = []
        for _ in range(num_sequences):
            if not self.neurons:
                break
            start_word = self.rng.choice(list(self.neurons.keys()))
            url = f"https://api.datamuse.com/words?rel_trg={start_word}&max={seq_length*2}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                related_words = response.json()
                if not related_words:
                    continue
                input_sequence = [start_word] + [self.tokenize_text(word['word'])[0] for word in related_words[:seq_length-1]]
                target_sequence = [min(float(word['score']) / 100000, 1.0) for word in related_words[:seq_length]]
                if len(input_sequence) == seq_length and len(target_sequence) == seq_length:
                    training_data.append((input_sequence, target_sequence))
            except requests.RequestException as e:
                print(f"Error fetching data for {start_word}: {e}")
        return training_data

    def backpropagate(self, input_sequence, target_sequence, optimizer, dropout_rate=0.2):
        """
        Perform backpropagation to update weights based on the error.
        """
        activations = self.forward_pass(input_sequence, dropout_rate)
        if not activations or not target_sequence:
            return 0.0  # Skip backpropagation for empty sequences

        # Ensure activations and target_sequence have the same shape
        min_length = min(len(activations), len(target_sequence))
        activations = activations[:min_length]
        target_sequence = target_sequence[:min_length]

        # Debugging: Print activations and target_sequence
        print(f"Activations: {activations}")
        print(f"Target Sequence: {target_sequence}")

        try:
            # Ensure both are flat lists of floats
            activations = [float(a) for a in activations]
            target_sequence = [float(t) for t in target_sequence]
            error = np.array(target_sequence, dtype=float) - np.array(activations, dtype=float)
        except (ValueError, TypeError) as e:
            print(f"Error computing error: {e}")
            print(f"Activations: {activations}")
            print(f"Target Sequence: {target_sequence}")
            return 0.0  # Skip this backpropagation step due to data inconsistency

        total_loss = 0.0

        for i, word in enumerate(input_sequence[:min_length]):
            if word in self.neurons:
                neuron = self.neurons[word]
                neuron.gradient = error[i] * (1 - neuron.activation ** 2)
                for connected_word in neuron.connections:
                    connected_neuron = self.neurons[connected_word]
                    gradient = neuron.gradient * connected_neuron.activation
                    neuron.gradients[connected_word] = gradient
        # Update weights using the optimizer
        optimizer.update(self)
        # Calculate loss
        loss = np.mean(error ** 2)
        return loss

    def forward_pass(self, input_sequence, dropout_rate=0.2):
        """
        Perform a forward pass through the network with the given input sequence.
        """
        activations = []
        for word in input_sequence:
            if word in self.neurons:
                neuron = self.neurons[word]
                # Calculate input_signal as sum of activations * weights
                input_signal = 0.0
                for connected_word in neuron.connections:
                    connected_neuron = self.neurons[connected_word]
                    act = connected_neuron.activation
                    input_signal += act * neuron.weights.get(connected_word, 0)
                neuron.activate(input_signal)
                # Apply dropout (during training)
                if random.random() <  dropout_rate:
                    neuron.activation = 0.0
                activations.append(neuron.activation)
            else:
                activations.append(0.0)
        return activations

    def attention(self, query, keys, values):
        """
        Compute attention weights and context vector.
        """
        attention_weights = np.dot(query, np.array(keys).T)
        attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
        context = np.dot(attention_weights, values)
        return context, attention_weights

    def generate_response(self, input_sequence, max_length=20, temperature=0.5):
        """
        Generate a response based on the input sequence.
        """
        response = []
        context = self.forward_pass(input_sequence)
        dropout_rate = 0.0  # No dropout during generation

        for _ in range(max_length):
            query = np.mean(context) if context else 0.0
            keys = [n.activation for n in self.neurons.values()]
            values = [n.position for n in self.neurons.values()]

            if not keys or not values:
                break  # Prevent errors if there are no neurons

            attended_context, _ = self.attention(query, keys, values)

            # Calculate distances and convert to probabilities
            distances = [np.linalg.norm(n.position - attended_context) for n in self.neurons.values()]
            probabilities = np.exp(-np.array(distances) / temperature)
            probabilities /= np.sum(probabilities)

            # Sample word based on probabilities, avoiding repetition
            try:
                next_word = self.rng.choice(list(self.neurons.keys()), p=probabilities)
            except ValueError as e:
                print(f"Error in sampling next_word: {e}")
                return "Unable to generate a response at this time."

            if response and next_word == response[-1]:
                continue  # Avoid immediate repetition

            response.append(next_word)
            context = self.forward_pass(response[-3:], dropout_rate=dropout_rate)  # Update context with recent words

            if next_word in ['.', '!', '?']:
                break

        return ' '.join(response)

    def train_with_api_data(self, num_sequences=100, seq_length=5, epochs=10, batch_size=32, learning_rate=0.001, dropout_rate=0.2, weight_decay=0.0001):
        """
        Train the network using data fetched from an API with adjustable parameters.
        """
        self.learning_rate = learning_rate  # Update learning rate
        self.optimizer.lr = learning_rate
        self.optimizer.weight_decay = weight_decay
        training_data = self.fetch_training_data(num_sequences, seq_length)
        if not training_data:
            return "No training data could be fetched. Please ensure the network has words and the API is accessible."
        for epoch in range(epochs):
            total_loss = 0
            valid_sequences = 0
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                for input_sequence, target_sequence in batch:
                    if len(input_sequence) != len(target_sequence):
                        print(f"Skipping sequence due to length mismatch: {len(input_sequence)} != {len(target_sequence)}")
                        continue
                    loss = self.backpropagate(input_sequence, target_sequence, self.optimizer, dropout_rate)
                    total_loss += loss
                    valid_sequences += 1
            average_loss = total_loss / valid_sequences if valid_sequences else 0
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.6f}, Valid Sequences: {valid_sequences}")
        return f"Training completed with {valid_sequences} valid sequences for {epochs} epochs"

    async def initialize_with_wikipedia_topics(self, topics):
        """
        Initialize the network with a predefined list of Wikipedia topics.
        """
        results = await self.learn_from_wikipedia_async(topics, concurrency=5)
        return results

    def fetch_wikipedia_content(self, topic):
        """
        Fetch content from a Wikipedia article based on the topic.
        """
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text for p in paragraphs])
            return content
        except requests.RequestException as e:
            print(f"Error fetching {topic}: {e}")
            return None

    def learn_from_wikipedia(self, topic):
        """
        Learn from a Wikipedia article by tokenizing and adding tokens to the network.
        """
        content = self.fetch_wikipedia_content(topic)
        if content:
            tokens = self.tokenize_text(content)
            for token in tokens:
                self.add_word(token)
            for i in range(len(tokens) - 1):
                self.connect_words(tokens[i], tokens[i + 1])
            return f"Learned from Wikipedia article: {topic}"
        else:
            return f"Failed to fetch content for: {topic}"

    def save_state(self, filename):
        """
        Save the current state of the network to a JSON file.
        """
        state = {
            'neurons': {
                word: {
                    'position': neuron.position.tolist(),
                    'connections': {w: weight for w, weight in neuron.weights.items()}
                }
                for word, neuron in self.neurons.items()
            },
            'space_size': self.space_size,
            'learning_rate': self.learning_rate,
            'optimizer': {
                'm': convert_ndarray_to_list(self.optimizer.m),
                'v': convert_ndarray_to_list(self.optimizer.v),
                't': self.optimizer.t
            },
            'rng_state': convert_ndarray_to_list(self.rng.bit_generator.state)  # Convert ndarrays to lists
        }
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=4)
            return f"State saved to {filename}"
        except Exception as e:
            return f"Failed to save state to {filename}: {e}"

    @staticmethod
    def load_state(filename):
        """
        Load the network state from a JSON file.
        """
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            network = FractalNeuralNetwork(state['space_size'])
            network.learning_rate = state['learning_rate']
            # Restore optimizer state
            network.optimizer.m = convert_list_to_ndarray(state['optimizer']['m'])
            network.optimizer.v = convert_list_to_ndarray(state['optimizer']['v'])
            network.optimizer.t = state['optimizer']['t']
            # Restore RNG state by converting lists back to ndarrays
            restored_rng_state = convert_list_to_ndarray(state['rng_state'])
            network.rng.bit_generator.state = restored_rng_state
            for word, data in state['neurons'].items():
                network.add_word(word)
                network.neurons[word].position = np.array(data['position'])
                for connected_word, weight in data['connections'].items():
                    network.connect_words(word, connected_word)
                    network.neurons[word].weights[connected_word] = weight
            return network
        except Exception as e:
            print(f"Failed to load state from {filename}: {e}")
            return None

    def visualize(self):
        """
        Visualize the network structure using a 3D plot.
        Returns a PIL Image compatible with Gradio.
        """
        if not self.neurons:
            return "The network is empty. Add words to visualize."

        G = nx.Graph()
        for word, neuron in self.neurons.items():
            G.add_node(word, pos=neuron.position)
        for word, neuron in self.neurons.items():
            for connected_word in neuron.connections:
                G.add_edge(word, connected_word)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        pos = nx.get_node_attributes(G, 'pos')

        # Extract positions
        xs = [pos[word][0] for word in G.nodes()]
        ys = [pos[word][1] for word in G.nodes()]
        zs = [pos[word][2] for word in G.nodes()]

        # Draw nodes
        ax.scatter(xs, ys, zs, c='r', s=20)

        # Draw edges
        for edge in G.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, c='gray', alpha=0.5)

        ax.set_xlim(0, self.space_size)
        ax.set_ylim(0, self.space_size)
        ax.set_zlim(0, self.space_size)
        plt.title("Fractal Neural Network Visualization")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        image = Image.open(buf)
        return image

    def chat(self, input_text, temperature=0.5):
        """
        Handle chat interactions by generating responses based on input text.
        """
        tokens = self.tokenize_text(input_text)
        if not tokens:
            return "I didn't understand that. Please try again."
        response = self.generate_response(tokens, temperature=temperature)
        # Optionally, train the network with the input and response to improve over time
        # Here, we train with the input tokens and the response activations
        response_tokens = self.tokenize_text(response)
        self.train_with_api_data(
            num_sequences=1,
            seq_length=len(tokens),
            epochs=1,
            batch_size=1,
            learning_rate=self.learning_rate
        )
        return response


def create_gradio_interface():
    """
    Create the Gradio interface for interacting with the Fractal Neural Network.
    """
    network = FractalNeuralNetwork(seed=42)  # Set a seed for reproducibility

    with gr.Blocks() as iface:
        gr.Markdown("# 🧠 Fractal Neural Network Interface")
        gr.Markdown("""
        **⚠️ Warning:** Training the model with extensive data and high epochs will take a significant amount of time and computational resources. Please ensure your system is equipped to handle the training process.
        """)

        with gr.Tab("Initialize with Wikipedia Topics"):
            gr.Markdown("### Initialize the Network with Comprehensive Wikipedia Topics")
            gr.Markdown("""
            **Instructions:**
            - Enter a list of Wikipedia topics separated by commas.
            - Example topics are pre-filled to guide you.
            - Click **"Start Initialization"** to begin the process.
            - **Note:** This may take several minutes depending on the number of topics and your internet connection.
            """)

            wiki_input = gr.Textbox(
                label="Wikipedia Topics",
                placeholder="Enter Wikipedia topics separated by commas...",
                lines=5,
                value="Artificial Intelligence, History of Computing, Biology, Physics, Chemistry, Mathematics, World History, Geography, Literature, Philosophy"
            )
            init_button = gr.Button("Start Initialization")
            init_output = gr.Textbox(label="Initialization Output", interactive=False, lines=10)

            async def handle_initialization(wiki_topics):
                # Split the input string into a list of topics
                topics = [topic.strip() for topic in wiki_topics.split(",") if topic.strip()]
                if not topics:
                    return "Please enter at least one valid Wikipedia topic."
                # Learn from the provided Wikipedia topics
                result = await network.initialize_with_wikipedia_topics(topics)
                # Save the state after initialization
                save_result = network.save_state("fnn_state.json")
                return f"{result}\n\n{save_result}"

            init_button.click(fn=handle_initialization, inputs=wiki_input, outputs=init_output)

        with gr.Tab("API Training"):
            gr.Markdown("### Configure and Start API-Based Training")
            gr.Markdown("""
            **Instructions:**
            - Adjust the training parameters below according to your requirements.
            - Higher values will result in longer training times and increased computational load.
            - Click **"Start Training"** to begin the API-based training process.
            """)

            with gr.Row():
                num_sequences_input = gr.Number(label="Number of Sequences", value=50000, precision=0, step=1000)
                seq_length_input = gr.Number(label="Sequence Length", value=15, precision=0, step=1)
            with gr.Row():
                epochs_input = gr.Number(label="Number of Epochs", value=100, precision=0, step=1)
                batch_size_input = gr.Number(label="Batch Size", value=500, precision=0, step=50)
            with gr.Row():
                learning_rate_input = gr.Number(label="Learning Rate", value=0.0005, precision=5, step=0.0001)
            train_button = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Output", interactive=False, lines=10)

            def handle_api_training(num_sequences, seq_length, epochs, batch_size, learning_rate):
                if not network.neurons:
                    return "The network has no words. Please initialize it with Wikipedia topics first."
                if num_sequences <= 0 or seq_length <= 0 or epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
                    return "All training parameters must be positive numbers."
                # Start training
                result = network.train_with_api_data(
                    num_sequences=int(num_sequences),
                    seq_length=int(seq_length),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate)
                )
                # Save the state after training
                save_result = network.save_state("fnn_state.json")
                return f"{result}\n\n{save_result}"

            train_button.click(
                fn=handle_api_training,
                inputs=[num_sequences_input, seq_length_input, epochs_input, batch_size_input, learning_rate_input],
                outputs=train_output
            )

        with gr.Tab("Visualization"):
            gr.Markdown("### Visualize the Fractal Neural Network")
            gr.Markdown("""
            **Instructions:**
            - Click **"Visualize Network"** to generate a 3D visualization of the network's structure.
            - Ensure the network has been initialized and trained before visualizing.
            """)

            visualize_button = gr.Button("Visualize Network")
            visualize_image = gr.Image(label="Network Visualization")

            def handle_visualize():
                if not network.neurons:
                    return "The network is empty. Add words to visualize."
                return network.visualize()

            visualize_button.click(fn=handle_visualize, inputs=None, outputs=visualize_image)

        with gr.Tab("Chat"):
            gr.Markdown("### Interact with the Fractal Neural Network")
            gr.Markdown("""
            **Instructions:**
            - Enter your message in the textbox below.
            - Adjust the **Temperature** slider to control the randomness of the response.
              - **Lower values (e.g., 0.2):** More deterministic and focused responses.
              - **Higher values (e.g., 0.8):** More creative and varied responses.
            - Click **"Chat"** to receive a generated response.
            """)

            with gr.Row():
                chat_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=2)
                chat_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Temperature")
            chat_button = gr.Button("Chat")
            chat_output = gr.Textbox(label="Response", interactive=False, lines=2)

            def handle_chat(input_text, temperature):
                if not input_text.strip():
                    return "Please enter a message to chat."
                response = network.chat(input_text, temperature=temperature)
                return response

            chat_button.click(fn=handle_chat, inputs=[chat_input, chat_temperature], outputs=chat_output)

        with gr.Tab("State Management"):
            gr.Markdown("### Save or Load the Network State")
            gr.Markdown("""
            **Instructions:**
            - **Save State:** Enter a filename and click **"Save State"** to save the current network configuration.
            - **Load State:** Enter a filename and click **"Load State"** to load a previously saved network configuration.
            - Ensure that the filenames are correctly specified and that the files exist when loading.
            """)

            with gr.Row():
                save_filename_input = gr.Textbox(label="Filename to Save State", value="fnn_state.json", placeholder="e.g., fnn_state.json")
                save_button = gr.Button("Save State")
            save_output = gr.Textbox(label="Save Output", interactive=False, lines=2)

            def handle_save(filename):
                if not filename.strip():
                    return "Please enter a valid filename."
                result = network.save_state(filename)
                return result

            save_button.click(fn=handle_save, inputs=save_filename_input, outputs=save_output)

            with gr.Row():
                load_filename_input = gr.Textbox(label="Filename to Load State", value="fnn_state.json", placeholder="e.g., fnn_state.json")
                load_button = gr.Button("Load State")
            load_output = gr.Textbox(label="Load Output", interactive=False, lines=2)

            def handle_load(filename):
                if not filename.strip():
                    return "Please enter a valid filename."
                loaded_network = FractalNeuralNetwork.load_state(filename)
                if loaded_network:
                    nonlocal network
                    network = loaded_network
                    return f"Loaded state from {filename}."
                else:
                    return f"Failed to load state from {filename}."

            load_button.click(fn=handle_load, inputs=load_filename_input, outputs=load_output)

    return iface


if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()
