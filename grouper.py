from PIL import Image
import imagehash
import os

threshold = 28 # Define the threshold value for similarity detection

image_dir = "D:\IIT Mandi\DP\Minh\Bikes"  # Replace with the path to the directory containing the images

# Loop over all images in the directory
for root, _, files in os.walk(image_dir):
    for f in files:
    #if filename.endswith(".jpg") or filename.endswith(".png"):  # Check that the file is an image
        filepath = os.path.join(root, f)
        #checkin
        if os.path.exists(filepath):
            # Load the image using Pillow
            image1 = Image.open(filepath)
            # Calculate the perceptual hash of the image using imagehash
            hash1 = imagehash.phash(image1)
            # Compare the image to all other images in the directory
            for root2, _, files2 in os.walk(image_dir):
                for f2 in files2:
                #if other_filename != filename and (other_filename.endswith(".jpg") or other_filename.endswith(".png")):
                    other_filepath = os.path.join(root2, f2)
                    # Load the other image using Pillow
                    image2 = Image.open(other_filepath)
                    # Calculate the perceptual hash of the other image using imagehash
                    hash2 = imagehash.phash(image2)
                    # If the hamming distance between the hashes is less than or equal to the threshold, delete the current image
                    if hash1 - hash2 <= threshold and hash1!=hash2:
                            if os.path.getsize(filepath)<os.path.getsize(other_filepath):
                                print(os.path.getsize(filepath))
                                os.remove(filepath)
                                break
                            else:
                                print(os.path.getsize(other_filepath))
                                os.remove(other_filepath)
                        # Once an image is deleted, stop comparing it to other images in the directory