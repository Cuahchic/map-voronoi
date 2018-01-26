"""
Started off with the code here: https://codegolf.stackexchange.com/a/50345
Adapted it for my own needs
"""

import math
import random
import collections
import os
import functools
import operator as op
import numpy as np
import glob

from scipy.spatial import cKDTree as KDTree
from skimage.filters.rank import entropy
from skimage.morphology import disk, dilation
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from skimage.filters import sobel, gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_tv_bregman
from skimage.transform import downscale_local_mean


# This function denoises an image according to input parameters
def denoise(img, denoise_type = 'bilateral', printout = False):
    if denoise_type in ['bilateral', 'tv_bregman']:
        print('Denoising image using {} method.'.format(denoise_type))
    else:
        print('No denoising required.')
    
    if denoise_type == 'bilateral':
        denoised_img = denoise_bilateral(img['img_array'], sigma_range=0.15, sigma_spatial=15)
    elif denoise_type == 'tv_bregman':
        denoised_img = denoise_tv_bregman(img['img_array'], weight = 1.0)
    else:   # No denoising
        denoised_img = img['img_array']
        printout = False    # No point in printing if we didn't change the image
        
    if printout:
        output_filename = os.path.join(img['path'], '01_' + img['root_filename'] + '_denoised.png')
        print('Printing {}'.format(output_filename))
        imsave(output_filename, denoised_img)
    
    return denoised_img


# This function makes a grayscale and light-alpha-beta representation of the image
def decolour(img, denoised_img, printout = False):
    print('Decolouring the image')
    
    img_gray = rgb2gray(denoised_img)
    img_lab = rgb2lab(denoised_img)
    
    if printout:
        output_filename = os.path.join(img['path'], '02_' + img['root_filename'] + '_grayscale.png')
        print('Printing {}'.format(output_filename))
        imsave(output_filename, img_gray)
    
    return img_gray, img_lab
    

# This function takes a grayscale and calculates entropies
def entropies(img, img_gray, printout = False):
    print('Calculating entropies...')
    
    entropy_weight = 2**(entropy(img_as_ubyte(img_gray), disk(15)))
    entropy_weight /= np.amax(entropy_weight)
    entropy_weight = gaussian_filter(dilation(entropy_weight, disk(15)), 5)
    
    if printout:
        output_filename = os.path.join(img['path'], '03_' + img['root_filename'] + '_entropyweights.png')
        print('Printing {}'.format(output_filename))
        imsave(output_filename, entropy_weight)
    
    return entropy_weight


# This function takes a light-alpha-beta and calculates edge weights
def edges(img, img_lab, printout = False):
    print('Calculating edges...')
    
    color = [sobel(img_lab[:, :, channel])**2 for channel in range(1, 3)]
    edge_weight = functools.reduce(op.add, color) ** (1/2) / 75
    edge_weight = dilation(edge_weight, disk(5))
    
    if printout:
        output_filename = os.path.join(img['path'], '04_' + img['root_filename'] + '_edgeweights.png')
        print('Printing {}'.format(output_filename))
        imsave(output_filename, edge_weight)
    
    return edge_weight


# This function calculates the random Poisson points using a weighted combination of entropy and edge detection
# These are used as centroids for Voronoi diagram, once they have been reduced to n samples
def poisson_disc(img, entropy_weight, edge_weight, n, k=30, printout = False):
    print('Calculating random Poisson sampling...')
    
    h, w = img['img_array'].shape[:2]

    weight = (0.3*entropy_weight + 0.7*edge_weight)
    weight /= np.mean(weight)
    weight = weight

    max_dist = min(h, w) / 4
    avg_dist = math.sqrt(w * h / (n * math.pi * 0.5) ** (1.05))
    min_dist = avg_dist / 4

    dists = np.clip(avg_dist / weight, min_dist, max_dist)

    def gen_rand_point_around(point):
        radius = random.uniform(dists[point], max_dist)
        angle = random.uniform(0, 2 * math.pi)
        
        offset = np.array([int(round(radius * math.sin(angle))), int(round(radius * math.cos(angle)))])
            
        return tuple(point + offset)

    def has_neighbours(point):
        point_dist = dists[point]
        distances, idxs = tree.query(point,
                                    len(sample_points) + 1,
                                    distance_upper_bound=max_dist)

        if len(distances) == 0:
            return True

        for dist, idx in zip(distances, idxs):
            if np.isinf(dist):
                break
            
            tree_data = tuple(tree.data[idx])
            int_tree_data = (int(tree_data[0]), int(tree_data[1]))
            if dist < point_dist and dist < dists[int_tree_data]:
                return True

        return False

    # Generate first point randomly.
    first_point = (round(random.uniform(0, h)), round(random.uniform(0, w)))
    to_process = [first_point]
    sample_points = [first_point]
    tree = KDTree(sample_points)

    while to_process:
        # Pop a random point.
        point = to_process.pop(random.randrange(len(to_process)))

        for _ in range(k):
            new_point = gen_rand_point_around(point)

            if (0 <= new_point[0] < h and 0 <= new_point[1] < w):
                if not has_neighbours(new_point):
                    to_process.append(new_point)
                    sample_points.append(new_point)
                    tree = KDTree(sample_points)
                    if len(sample_points) % 1000 == 0:
                        print("Generated {} points.".format(len(sample_points)))

    print("Total points generated = {} points.".format(len(sample_points)))
    
    if printout:
        poisson_out = np.zeros((h,w))
        for s in sample_points:
            poisson_out[s] = 1.0
        
        output_filename = os.path.join(img['path'], '05_' + img['root_filename'] + '_allpoissonsamples.png')
        print('Printing {}'.format(output_filename))
        imsave(output_filename, poisson_out)

    return sample_points


# For each Voronoi cell sample the colour so we can make the whole cell the same colour
def sample_colors(img, sample_points, n, printout = False):
    h, w = img['img_array'].shape[:2]

    print("Sampling colors...")
    tree = KDTree(np.array(sample_points))
    color_samples = collections.defaultdict(list)
    img_lab = rgb2lab(img['img_array'])
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    pixel_coords = np.c_[xx.ravel(), yy.ravel()]
    nearest = tree.query(pixel_coords)[1]

    i = 0
    for pixel_coord in pixel_coords:
        color_samples[tuple(tree.data[nearest[i]])].append(
            img_lab[tuple(pixel_coord)])
        i += 1

    print("Computing color means...")
    samples = []
    for point, colors in color_samples.items():
        avg_color = np.sum(colors, axis=0) / len(colors)
        samples.append(np.append(point, avg_color))

    if len(samples) > n:
        print("Downsampling {} to {} points...".format(len(samples), n))

    while len(samples) > n:
        tree = KDTree(np.array(samples))
        dists, neighbours = tree.query(np.array(samples), 2)
        dists = dists[:, 1]
        worst_idx = min(range(len(samples)), key=lambda i: dists[i])
        samples[neighbours[worst_idx][1]] += samples[neighbours[worst_idx][0]]
        samples[neighbours[worst_idx][1]] /= 2
        samples.pop(neighbours[worst_idx][0])

    color_samples = []
    for sample in samples:
        color = lab2rgb([[sample[2:]]])[0][0]
        color_samples.append(tuple(sample[:2][::-1]) + tuple(color))
        
    if printout:
        samples_out = np.zeros((h,w))
        for s in color_samples:
            coords = s[:2][::-1]
            int_coords = (int(coords[0]), int(coords[1]))
            samples_out[int_coords] = 1.0
        
        output_filename = os.path.join(img['path'], '06_' + img['root_filename'] + '_filteredpoissonsamples.png')
        print('Printing {}'.format(output_filename))
        imsave(output_filename, samples_out)

    return color_samples


# Turn samples into output image
def render(img, color_samples):
    print("Rendering...")
    h, w = [2*x for x in img['img_array'].shape[:2]]
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    pixel_coords = np.c_[xx.ravel(), yy.ravel()]

    colors = np.empty([h, w, 3])
    coords = []
    for color_sample in color_samples:
        color_tuple = tuple(x*2 for x in color_sample[:2][::-1])
        
        coord = (int(color_tuple[0]), int(color_tuple[1]))
        colors[coord] = color_sample[2:]
        coords.append(coord)

    tree = KDTree(coords)
    idxs = tree.query(pixel_coords)[1]
    data = colors[tuple(tree.data[idxs].astype(int).T)].reshape((w, h, 3))
    data = np.transpose(data, (1, 0, 2))

    return downscale_local_mean(data, (2, 2, 1))


""" **************************************************************
*************************** START HERE ***************************
************************************************************** """
# Main body of code, not designed to be run from command line so no need for main() function
# Initial setup parameters
os.chdir('C:/GitWorkspace/map-voronoi') # Base directory
n = 3000                                # Number of desired centroids for Voronoi

# Read all JPG images prefixed 00 into an array to be Voronoi-ised
imgs = []
for filename in glob.glob('maps/00*.jpg'): #assuming gif
    basename = os.path.basename(filename)
    imgs.append({'path': os.path.dirname(filename),
                 'fullpath': filename,
                 'filename': basename,
                 'root_filename': basename[3:(len(basename)-4)],
                 'img_array': imread(filename)[:, :, :3]})
    
for img in imgs:    
    denoised_img = denoise(img, None, printout = True)
    img_gray, img_lab = decolour(img, denoised_img, printout = True)
    entropy_weight = entropies(img, img_gray, printout = True)
    edge_weight = edges(img, img_lab, printout = True)
    sample_points = poisson_disc(img, entropy_weight, edge_weight, n, k = 30, printout = True)
    samples = sample_colors(img, sample_points, n, printout = True)
    
    # Here we put some code to output centroids so we can display them using D3.js

    output_filename = os.path.join(img['path'], '07_' + img['root_filename'] + '_voronoi.png')
    print('Printing {}'.format(output_filename))
    imsave(output_filename, render(img, samples))

    print("Done!")








