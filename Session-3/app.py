from flask import Flask, render_template, request, flash
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter  # for image processing
import numpy as np
from scipy.io import wavfile  # for audio processing
import trimesh  # for 3D processing

import random,matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from nltk.corpus import wordnet
import nltk
from torchvision import transforms
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import trimesh
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the 3D model from the .off file
# mesh = trimesh.load('/Users/v.kanukollu/MyStuff/Personal/Courses/ERAv3/Session-3/sample/sample_cube.off')


#######################
# 3D Processing:
#######################
# Function to visualize the 3D model
def plot_mesh(mesh, title="3D Model Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], 
                    mesh.vertices[:, 1], 
                    triangles=mesh.faces, 
                    Z=mesh.vertices[:, 2], 
                    color=(0.5, 0.5, 1, 0.5), edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Save plot to static folder
    plot_path = os.path.join('static', f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


# Augmentation: Random Rotation
def augment_mesh_random_rotation_old(mesh):
    """Applies a random rotation to the mesh."""
    # Generate random rotation angles for x, y, and z axes
    angles = np.radians(np.random.uniform(0, 360, 3))
    rotation_matrix = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2])[:3, :3]
    mesh.apply_transform(rotation_matrix)
    return mesh

# Augmentation: Random Rotation (corrected to use a 4x4 matrix)
def augment_mesh_random_rotation(mesh):
    """Applies a random rotation to the mesh."""
    # Generate random rotation angles for x, y, and z axes
    angles = np.radians(np.random.uniform(0, 360, 3))
    # Create a 4x4 homogeneous rotation matrix
    rotation_matrix = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2])
    # Apply the transformation
    mesh.apply_transform(rotation_matrix)
    return mesh


# Augmentation: Scaling
def augment_mesh_scale(mesh, scale_factor=1.2):
    """Scales the mesh by the given scale factor."""
    mesh.vertices *= scale_factor
    return mesh

# Augmentation: Adding Noise
def augment_mesh_add_noise(mesh, noise_level=0.05):
    """Adds random noise to each vertex of the mesh."""
    noise = np.random.normal(0, noise_level, mesh.vertices.shape)
    mesh.vertices += noise
    return mesh

#######################
# Text Processing:
#######################

# Download the WordNet data if it's not already available
nltk.download('wordnet')

def random_insertion(sentence, n=1):
    """Randomly insert n synonyms into the sentence."""
    words = sentence.split()
    for _ in range(n):
        new_word = get_random_synonym(words)
        if new_word:
            position = random.randint(0, len(words))
            words.insert(position, new_word)
    return ' '.join(words)

def get_random_synonym(words):
    """Get a random synonym for a random word in the sentence."""
    random_word = random.choice(words)
    synonyms = wordnet.synsets(random_word)
    if synonyms:
        synonym = synonyms[0].lemmas()[0].name()  # Select the first synonym found
        if synonym != random_word:
            return synonym
    return None

def random_deletion(sentence, p=0.2):
    """Randomly delete words from the sentence with probability p."""
    words = sentence.split()
    if len(words) == 1:  # Return the original sentence if there's only one word
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    # Ensure there's at least one word
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

def random_swap(sentence, n=1):
    """Randomly swap two words in the sentence n times."""
    words = sentence.split()
    for _ in range(n):
        words = swap_random_words(words)
    return ' '.join(words)

def swap_random_words(words):
    """Swap two random words in the list."""
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words

def synonym_replacement(sentence, n=1):
    """Replace n words with their synonyms."""
    words = sentence.split()
    for _ in range(n):
        new_word = get_random_synonym(words)
        if new_word:
            random_word = random.choice(words)
            words[words.index(random_word)] = new_word
    return ' '.join(words)

#######################
# App:
#######################

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3', 'obj', 'stl', 'off'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(file_path, process_type):
    img = Image.open(file_path)
    processed_path = file_path.rsplit('.', 1)[0] + '_processed.' + file_path.rsplit('.', 1)[1]
    augmented_path = file_path.rsplit('.', 1)[0] + '_augmented.' + file_path.rsplit('.', 1)[1]
    
    # Process
    if process_type == 'grayscale':
        processed = img.convert('L')
    elif process_type == 'blur':
        processed = img.filter(ImageFilter.BLUR)
    elif process_type == 'edge':
        processed = img.filter(ImageFilter.FIND_EDGES)
    elif process_type == 'sharpen':
        processed = img.filter(ImageFilter.SHARPEN)
    
    # Augmentation
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # transforms.RandomCrop(size=(224, 224), pad_if_needed=True)
    ])
    
    # Apply augmentations
    augmented = augmentations(img)
    
    # Save processed image
    processed.save(processed_path)
    
    # Convert augmented tensor to PIL Image and save
    if isinstance(augmented, torch.Tensor):
        # If it's a tensor, convert back to PIL Image
        to_pil = transforms.ToPILImage()
        augmented = to_pil(augmented)
    augmented.save(augmented_path)
    
    return processed_path, augmented_path

def process_text(file_path, process_type):
    processed_path = file_path.rsplit('.', 1)[0] + '_processed.txt'
    augmented_path = file_path.rsplit('.', 1)[0] + '_augmented.txt'
    
    with open(file_path, 'r') as file:
        content = file.read()

    # Processed Content:
    processed_content = content.lower()

    # Augmented Content:
    if process_type == 'random_insertion':
        # Apply random insertion
        augmented_sentence = random_insertion(content, n=2)
        print("Random Insertion:", augmented_sentence)
    
    elif process_type == 'random_deletion':
        # Apply random deletion
        augmented_sentence = random_deletion(content, p=0.3)
        print("Random Deletion:", augmented_sentence)

    elif process_type == 'random_swap':
        # Apply random swap
        augmented_sentence = random_swap(content, n=2)
        print("Random Swap:", augmented_sentence)
    
    elif process_type == 'synonym_replacement':
        augmented_sentence = synonym_replacement(content, n=2)
        print("Synonym Replacement:", augmented_sentence)
    
    # Augment (example: uppercase)
    augmented_content = augmented_sentence
    
    with open(processed_path, 'w') as file:
        file.write(processed_content)
    with open(augmented_path, 'w') as file:
        file.write(augmented_content)
    
    return processed_path, augmented_path

def process_audio(file_path, process_type):
    processed_path = file_path.rsplit('.', 1)[0] + '_processed.' + file_path.rsplit('.', 1)[1]
    augmented_path = file_path.rsplit('.', 1)[0] + '_augmented.' + file_path.rsplit('.', 1)[1]
    
    # Add audio processing logic here
    # This is a placeholder - you'll need to implement actual audio processing
    
    
    return file_path, file_path

def process_3d(file_path, process_type):
    processed_path = file_path.rsplit('.', 1)[0] + '_processed.' + file_path.rsplit('.', 1)[1]
    augmented_path = file_path.rsplit('.', 1)[0] + '_augmented.' + file_path.rsplit('.', 1)[1]
    # Load the 3D model
    mesh = trimesh.load(file_path)
    
    # Create processed mesh (original visualization)
    processed_plot_path = plot_mesh(mesh, title="Original 3D Model")
    processed_mesh = processed_plot_path

    # Apply augmentation based on process type
    if process_type == 'random_rotation':
        # Apply random rotation
        augmented_mesh = augment_mesh_random_rotation(mesh.copy())
        augmented_plot_path = plot_mesh(augmented_mesh, title="Augmented 3D Model (Random Rotation)")
        
    elif process_type == 'scaling':
        # Apply scaling
        augmented_mesh = augment_mesh_scale(mesh.copy(), scale_factor=1.2)
        augmented_plot_path = plot_mesh(augmented_mesh, title="Augmented 3D Model (Scaled)")
        
    elif process_type == 'noise':
        # Apply noise
        augmented_mesh = augment_mesh_add_noise(mesh.copy(), noise_level=0.05)
        augmented_plot_path = plot_mesh(augmented_mesh, title="Augmented 3D Model (With Noise)")
        
    else:
        # Default to original if no valid process type
        augmented_plot_path = processed_plot_path
    
    return  processed_mesh,augmented_plot_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template('index.html')
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get file type and process type from form
            file_type = request.form.get('file-type', 'image')
            process_type = request.form.get(f'{file_type}-process', 'default')
            print(file_type, process_type)
            
            # Process the file based on type
            if file_type == 'image':
                processed_path, augmented_path = process_image(file_path, process_type)
            elif file_type == 'text':
                processed_path, augmented_path = process_text(file_path, process_type)
            elif file_type == 'audio':
                processed_path, augmented_path = process_audio(file_path, process_type)
            # elif file_type == '3d':
                # processed_path, augmented_path = process_3d(file_path, process_type)
            
            return render_template('index.html', 
                                 original_file=file_path,
                                 processed_file=processed_path,
                                 augmented_file=augmented_path,
                                 file_type=file_type)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True) 