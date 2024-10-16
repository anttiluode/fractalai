import numpy as np
import pickle
import gradio as gr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image
import random
import requests
from bs4 import BeautifulSoup
import time
import networkx as nx

# Constants
MAX_DEPTH = 15
MAX_CHILDREN = 5
SPACE_SIZE = 10
GROWTH_PROBABILITY = 0.2  # Increased from 0.1

class FractalNode:
    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position
        self.connections = {}
        self.activation = 0.0

    def activate(self, input_signal):
        self.activation = np.tanh(input_signal)

    def connect(self, other_node, weight):
        self.connections[other_node.id] = weight

class FractalNetwork:
    def __init__(self, initial_nodes=5, space_size=SPACE_SIZE):
        self.nodes = {}
        self.space_size = space_size
        self.graph = nx.Graph()
        self.cycle_count = 0
        self.memory = ""
        self.create_initial_nodes(initial_nodes)

    def create_initial_nodes(self, num_nodes):
        for i in range(num_nodes):
            position = np.random.rand(3) * self.space_size
            self.add_node(FractalNode(i, position))

    def add_node(self, node):
        self.nodes[node.id] = node
        self.graph.add_node(node.id, pos=node.position)

    def connect_nodes(self, node1, node2, weight):
        node1.connect(node2, weight)
        node2.connect(node1, weight)
        self.graph.add_edge(node1.id, node2.id, weight=weight)

    def grow(self):
        new_node_id = len(self.nodes)
        position = np.random.rand(3) * self.space_size
        new_node = FractalNode(new_node_id, position)
        self.add_node(new_node)

        for node in self.nodes.values():
            if node.id != new_node_id:
                distance = np.linalg.norm(np.array(new_node.position) - np.array(node.position))
                if distance < self.space_size * 0.2:
                    weight = np.random.rand()
                    self.connect_nodes(new_node, node, weight)

    def hebbian_learning(self):
        for node in self.nodes.values():
            for other_node_id, weight in list(node.connections.items()):
                other_node = self.nodes[other_node_id]
                delta_weight = 0.01 * node.activation * other_node.activation
                new_weight = np.clip(weight + delta_weight, 0, 1)  # Clip weight to [0, 1]
                node.connections[other_node_id] = new_weight
                other_node.connections[node.id] = new_weight
                self.graph[node.id][other_node_id]['weight'] = new_weight

    def process_input(self, input_text):
        input_signal = sum(ord(c) for c in input_text) / len(input_text) / 128
        for node in self.nodes.values():
            node.activate(input_signal)
        self.hebbian_learning()
        if random.random() < GROWTH_PROBABILITY:
            self.grow()

    def think(self):
        self.cycle_count += 1
        for node in self.nodes.values():
            node.activate(np.random.rand())
        self.hebbian_learning()
        if random.random() < GROWTH_PROBABILITY:
            self.grow()
        return f"Cycle {self.cycle_count}: {chr(int(np.mean([node.activation for node in self.nodes.values()]) * 26) + 97)}"

    def chat(self, input_text):
        self.memory += input_text + " "
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]
        self.process_input(input_text)
        response = ''.join(random.choice(self.memory) for _ in range(20))
        self.cycle_count += 1
        return f"Cycle {self.cycle_count}: {response}"

    def save_state(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def visualize(self, zoom=1.0):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        for edge in self.graph.edges():
            start = pos[edge[0]]
            end = pos[edge[1]]
            weight = self.graph[edge[0]][edge[1]]['weight']
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    color='b', alpha=min(weight, 1.0), linewidth=weight*3)
        
        for node_id, node_pos in pos.items():
            ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                       color='r', s=100*self.nodes[node_id].activation+50)
        
        center = self.space_size / 2
        ax.set_xlim(center - self.space_size/(2*zoom), center + self.space_size/(2*zoom))
        ax.set_ylim(center - self.space_size/(2*zoom), center + self.space_size/(2*zoom))
        ax.set_zlim(center - self.space_size/(2*zoom), center + self.space_size/(2*zoom))
        plt.title(f"Fractal Network - {len(self.nodes)} nodes")
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        image = Image.open(buf)
        return image

def fetch_wikipedia_content(topic):
    url = f"https://en.wikipedia.org/wiki/{topic}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        return content
    else:
        return None

def gradio_interface():
    network = FractalNetwork()
    zoom_level = 1.0

    def cycle_ai(num_cycles):
        nonlocal zoom_level
        thoughts = []
        for _ in range(num_cycles):
            thought = network.think()
            thoughts.append(thought)
        
        image = network.visualize(zoom_level)
        
        return "\n".join(thoughts), image

    def save_state(filename):
        if filename.strip() == "":
            return "Please enter a valid filename."
        try:
            network.save_state(filename)
            return f"Network state saved as {filename}"
        except Exception as e:
            return f"Error saving network state: {str(e)}"

    def load_state(file):
        if file is None:
            return "Please upload a file."
        try:
            loaded_network = FractalNetwork.load_state(file.name)
            nonlocal network
            network = loaded_network
            return f"Loaded network state from {file.name}"
        except Exception as e:
            return f"Error loading network state: {str(e)}"

    def recreate_network(initial_nodes):
        nonlocal network, zoom_level
        network = FractalNetwork(initial_nodes=initial_nodes)
        image = network.visualize(zoom_level)
        return f"Network recreated with {initial_nodes} initial nodes", image

    def train_on_wikipedia(topic):
        nonlocal zoom_level
        content = fetch_wikipedia_content(topic)
        if content:
            chunks = [content[i:i+500] for i in range(0, len(content), 500)]
            thoughts = []
            for chunk in chunks:
                network.process_input(chunk)
                thoughts.append(f"Processed chunk: {network.think()}")
            
            image = network.visualize(zoom_level)
            return "\n".join(thoughts), image
        else:
            return f"Could not retrieve content for topic: {topic}", None

    def chat_with_ai(input_text):
        nonlocal zoom_level
        response = network.chat(input_text)
        image = network.visualize(zoom_level)
        return response, image

    def self_conversation(num_cycles):
        nonlocal zoom_level
        thoughts = []
        for _ in range(num_cycles):
            thought = network.think()
            thoughts.append(thought)
            
            image = network.visualize(zoom_level)
            
            time.sleep(0.1)  # Add a small delay to make the process visible
            
            yield "\n".join(thoughts), image

    def update_zoom(zoom_factor):
        nonlocal zoom_level
        zoom_level *= zoom_factor
        image = network.visualize(zoom_level)
        return image

    with gr.Blocks() as demo:
        gr.Markdown("# Advanced Fractal AI with Visualization and Interaction")

        with gr.Row():
            num_cycles = gr.Number(label="Number of Cycles", value=1, precision=0)
            cycle_button = gr.Button("Run Cycles")
        
        output_text = gr.Textbox(label="AI Thoughts", lines=5)
        fractal_viz = gr.Image(label="Fractal Visualization")
        
        with gr.Row():
            zoom_in = gr.Button("Zoom In")
            zoom_out = gr.Button("Zoom Out")
        
        with gr.Row():
            save_name = gr.Textbox(label="Save filename:")
            save_btn = gr.Button("Save Network State")
        
        load_file = gr.File(label="Load Network State")
        
        initial_nodes_slider = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Initial Nodes")
        recreate_btn = gr.Button("Recreate Network")
        
        wiki_topic = gr.Textbox(label="Wikipedia Topic:")
        wiki_btn = gr.Button("Train on Wikipedia")

        chat_input = gr.Textbox(label="Chat with Fractal AI")
        chat_output = gr.Textbox(label="Fractal AI Response", lines=3)
        chat_button = gr.Button("Send")

        self_convo_cycles = gr.Number(label="Self-Conversation Cycles", value=10, precision=0)
        self_convo_button = gr.Button("Start Self-Conversation")

        # Connect components
        cycle_button.click(cycle_ai, inputs=[num_cycles], outputs=[output_text, fractal_viz])
        save_btn.click(save_state, inputs=[save_name], outputs=[output_text])
        load_file.change(load_state, inputs=[load_file], outputs=[output_text])
        recreate_btn.click(recreate_network, inputs=[initial_nodes_slider], outputs=[output_text, fractal_viz])
        wiki_btn.click(train_on_wikipedia, inputs=[wiki_topic], outputs=[output_text, fractal_viz])
        chat_button.click(chat_with_ai, inputs=[chat_input], outputs=[chat_output, fractal_viz])
        self_convo_button.click(self_conversation, inputs=[self_convo_cycles], outputs=[output_text, fractal_viz])
        zoom_in.click(update_zoom, inputs=[gr.State(1.2)], outputs=[fractal_viz])
        zoom_out.click(update_zoom, inputs=[gr.State(0.8)], outputs=[fractal_viz])

    return demo

# Launch the Gradio interface
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()