import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def extract_features(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Get the new dimensions
    new_width = 500
    new_height = 500

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))

    dist_central_features = color_dist_central(resized_img,200)
    dist_several_features = color_dist_serveral(resized_img,200)
   
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    gray_histogram = cv2.calcHist([gray], [0], None, [8], [0, 256])
    gray_histogram = cv2.normalize(gray_histogram, gray_histogram).flatten()
    # Apply edge detection using the Canny algorithm
    edges = cv2.Canny(gray, 100, 200)
    gray_dist_serveral_features = gray_dist_serveral(gray,200)
    # Calculate the HOG features
    hog_features = hog(edges, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    
    # Extract the color histogram features
    color_histogram = cv2.calcHist([resized_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_histogram = cv2.normalize(color_histogram, color_histogram).flatten()
    
    hog_features = cv2.normalize(hog_features, hog_features).flatten()
    # Combine the HOG and color histogram features into a single feature vector
    feature_vector = np.concatenate((hog_features*0.5, color_histogram*0.3,  gray_histogram*0.3, gray_dist_serveral_features*0.3, dist_central_features*0.3, dist_several_features*0.3))

    return feature_vector

def create_similarity_matrix(image_paths):
    # Extract the feature vectors for each image
    feature_vectors = []
    for image_path in image_paths:
        feature_vectors.append(extract_features(image_path))

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(feature_vectors)

    return similarity_matrix

def color_dist_central(img,tileSize):
   
    h = img.shape[1]
    w = img.shape[0]
    ch = int(h/2)
    cw = int(w/2)
    center_img = img[ch-tileSize:ch+tileSize,cw-tileSize:cw+tileSize]
    # cv2.imshow(center_img)
    # cv2.imwrite("1.jpg", center_img)
    color_histogram = cv2.calcHist([center_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_histogram = cv2.normalize(color_histogram, color_histogram).flatten()
    return color_histogram

def color_dist_serveral(img,tileSize):
   
    h = img.shape[1]
    w = img.shape[0]
    ch = int(h/2)
    cw = int(w/2)
    center_img = img[ch-tileSize:ch+tileSize,cw-tileSize:cw+tileSize]
    top_left_img = img[0:tileSize*2,0:tileSize*2]
    top_right_img = img[0:tileSize*2,h-tileSize*2:h]
    bottom_left_img = img[w-tileSize*2:w,0:tileSize*2]
    bottom_right_img = img[w-tileSize*2:w,h-tileSize*2:h]

 
    
    
    color_histogram = cv2.calcHist([center_img,top_left_img,top_right_img,bottom_left_img,bottom_right_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_histogram = cv2.normalize(color_histogram, color_histogram).flatten()

    

    
    return color_histogram

def gray_dist_serveral(img,tileSize):
   
    h = img.shape[1]
    w = img.shape[0]
    ch = int(h/2)
    cw = int(w/2)
    center_img = img[ch-tileSize:ch+tileSize,cw-tileSize:cw+tileSize]
    top_left_img = img[0:tileSize*2,0:tileSize*2]
    top_right_img = img[0:tileSize*2,h-tileSize*2:h]
    bottom_left_img = img[w-tileSize*2:w,0:tileSize*2]
    bottom_right_img = img[w-tileSize*2:w,h-tileSize*2:h]

 
    
    
    color_histogram = cv2.calcHist([top_left_img,top_right_img,bottom_left_img,bottom_right_img], [0], None, [8], [0, 256])
    color_histogram = cv2.normalize(color_histogram, color_histogram).flatten()

    

    
    return color_histogram
    





# Define a list of image paths
# Define a list of image paths
image_paths = ["Images/01.jpg", "Images/02.jpg", "Images/03.jpg", "Images/04.jpg",
               "Images/05.jpg", "Images/06.jpg", "Images/07.jpg", "Images/08.jpg",
               "Images/09.jpg", "Images/10.jpg", "Images/11.jpg", "Images/12.jpg", "Images/13.jpg"]

similarity_matrix = create_similarity_matrix(image_paths)
print("Feature vector: ", similarity_matrix[1][12])

# Set the number of images and columns
num_images = len(image_paths)
num_cols = 1

# Create a list of tuples with image indices and similarity values
image_similarity = [(i, similarity_matrix[4][i]) for i in range(num_images)]

# Sort the list based on similarity value
sorted_image_similarity = sorted(image_similarity, key=lambda x: x[1], reverse=True)

# Create the figure and axis objects
fig, axs = plt.subplots(num_cols, num_images, figsize=(15, 15))

# Loop over the sorted list and add the images to the plot in the desired order
for i, (index, similarity) in enumerate(sorted_image_similarity):
    # Load the image
    img = cv2.imread(image_paths[index])

    # Resize the image
    img = cv2.resize(img, (500, 500))

    # Add the image to the grid
    ax = axs[i]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')
    ax.axis('off')
    ax.set_title(f"{round(similarity,3)}")

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Show the grid of images
plt.show()