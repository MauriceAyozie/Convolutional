import tkinter as tk
import numpy as np
import os
import glob
from PIL import Image


# Helper Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# Load Data
def load_data(data_dir):
    categories = ['NORMAL', 'PNEUMONIA']
    images = []
    labels = []

    for label_idx, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        image_files = glob.glob(os.path.join(category_dir, '*.jpeg'))

        for image_file in image_files:
            img = Image.open(image_file).convert('L')  # Convert image to grayscale
            img = img.resize((128, 128))  # Resize to a fixed size
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array.flatten())  # Flatten the image
            labels.append(label_idx)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


train_dir = 'chest_xray/train'
test_dir = '/chest_xray/test'

x_train, y_train = load_data(train_dir)
x_test, y_test = load_data(test_dir)

# Initialize Parameters
input_size = 128 * 128  # Size of flattened image (128x128 grayscale image)
hidden_size = 128
output_size = 2  # Normal and Pneumonia

weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
bias_hidden = np.zeros(hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
bias_output = np.zeros(output_size)


# UI Visualization
class NeuralNetVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Training Visualization")

        # Canvas for Neural Network
        self.canvas_nn = tk.Canvas(root, width=1000, height=600, bg="white")
        self.canvas_nn.grid(row=0, column=0, rowspan=2)

        # Canvas for Training Image
        self.canvas_image = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas_image.grid(row=0, column=1)

        # Labels for Epoch and Loss
        self.epoch_label = tk.Label(root, text="Epoch: 0", font=("Arial", 14))
        self.epoch_label.grid(row=1, column=1)

        self.loss_label = tk.Label(root, text="Loss: N/A", font=("Arial", 14))
        self.loss_label.grid(row=2, column=1)

        self.accuracy_label = tk.Label(root, text="Accuracy: N/A", font=("Arial", 14))
        self.accuracy_label.grid(row=3, column=0, columnspan=2)

    def update_image(self, image):
        """Update the training image visualization."""
        self.canvas_image.delete("all")
        for i in range(128):
            for j in range(128):
                color = int(image[i * 128 + j] * 255)
                hex_color = f"#{color:02x}{color:02x}{color:02x}"
                self.canvas_image.create_rectangle(
                    j * 2, i * 2, (j + 1) * 2, (i + 1) * 2, fill=hex_color, outline="")

    def draw_nn(self, input_image, epoch, loss, accuracy=None):
        """Draw the neural network based on the current input image."""
        self.canvas_nn.delete("all")
        node_radius = 15
        layers_x = [150, 450, 750]
        layer_sizes = [input_size, hidden_size, output_size]
        layer_names = ["Data Layer", "Computational Layer", "Prediction Layer"]

        # Draw layer names
        for layer_idx, x in enumerate(layers_x):
            self.canvas_nn.create_text(
                x, 20, text=layer_names[layer_idx], font=("Arial", 16, "bold"), fill="black")

        # Create a list to store nodes for later drawing connections
        nodes = []

        # Draw nodes for each layer
        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_nodes = []
            for node_idx in range(layer_size):
                x = layers_x[layer_idx]
                y = 80 + node_idx * 40  # Adjusted to fit all nodes vertically
                # Color nodes based on the input image (input layer) or activation (hidden/output layers)
                if layer_idx == 0:  # Input layer
                    color = f"#{int(input_image[node_idx] * 255):02x}0000"
                else:  # Hidden and output layers
                    color = f"#{int(255 - (input_image[node_idx] * 255)):02x}00FF"

                layer_nodes.append(self.canvas_nn.create_oval(
                    x - node_radius, y - node_radius, x + node_radius, y + node_radius, fill=color, outline="black"))
            nodes.append(layer_nodes)

        # Draw edges between layers gradually
        for i, layer_nodes in enumerate(nodes[:-1]):
            next_layer_nodes = nodes[i + 1]
            for start_node in layer_nodes:
                for end_node in next_layer_nodes:
                    weight_color = "#D3D3D3"  # Default color for edges
                    self.canvas_nn.create_line(
                        self.canvas_nn.coords(start_node)[2], self.canvas_nn.coords(start_node)[3],
                        self.canvas_nn.coords(end_node)[0], self.canvas_nn.coords(end_node)[1],
                        fill=weight_color, width=2)

        # Display Epoch, Loss, and Accuracy
        self.epoch_label.config(text=f"Epoch: {epoch}")
        self.loss_label.config(text=f"Loss: {loss:.4f}")
        if accuracy is not None:
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")

    def update_accuracy(self, accuracy):
        """Update accuracy in the UI."""
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")


# Training Functions
def forward_propagation(x):
    z_hidden = np.dot(x, weights_input_hidden) + bias_hidden
    a_hidden = sigmoid(z_hidden)

    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = softmax(z_output)

    return a_hidden, a_output


def backward_propagation(x, y, a_hidden, a_output, learning_rate):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    output_error = a_output - y
    d_weights_hidden_output = np.dot(a_hidden.T, output_error)
    d_bias_output = output_error.sum(axis=0)

    hidden_error = np.dot(output_error, weights_hidden_output.T) * sigmoid_derivative(a_hidden)
    d_weights_input_hidden = np.dot(x.T, hidden_error)
    d_bias_hidden = hidden_error.sum(axis=0)

    weights_input_hidden -= learning_rate * d_weights_input_hidden
    bias_hidden -= learning_rate * d_bias_hidden
    weights_hidden_output -= learning_rate * d_weights_hidden_output
    bias_output -= learning_rate * d_bias_output


def train_model(epochs, learning_rate, visualizer):
    epoch = 0
    epoch_loss = 0
    i = 0

    def train_step():
        nonlocal epoch, epoch_loss, i

        if epoch < epochs:
            x = x_train[i:i + 1]
            y = y_train[i:i + 1]

            a_hidden, a_output = forward_propagation(x)
            epoch_loss += -np.sum(y * np.log(a_output))

            backward_propagation(x, y, a_hidden, a_output, learning_rate)

            if i % 100 == 0:  # Update visualization every 100 steps
                visualizer.update_image(x[0])
                visualizer.draw_nn(x[0], epoch + 1, epoch_loss / (i + 1))

            i += 1
            if i == len(x_train):
                i = 0
                epoch += 1

            visualizer.root.after(10, train_step)  # Call this method again in 10ms

    visualizer.root.after(10, train_step)  # Start training after 10ms delay


# Main Code
root = tk.Tk()
visualizer = NeuralNetVisualizer(root)

train_model(epochs=20, learning_rate=0.01, visualizer=visualizer)

root.mainloop()


# The link to the dataset is as follows:
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia