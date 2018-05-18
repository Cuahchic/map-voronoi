# Voronoi-ify Any Image 
Take an image and use ML to turn it into a Voronoi map.

Uses Python 3.6.5.

## How It Works
The initial code for turning the map into a Voronoi diagram comes from [orlp on Stack Exchange](https://codegolf.stackexchange.com/a/50345). This link has a good explanation of the approach taken, I have just cleaned up the code and made small modifications for my own needs.

Inside the maps folder, put any `jpg` images, and use the filename format:
	`00_FILENAME_N.jpg`

where:
	`FILENAME` is the name of the file. Please do not include spaces or any special characters.
	`N` is the number of Voronoi centroids you want to create. If you're not sure, try 2000.
	
Once you have all your desired files, run the Python script and the outputs will be inserted into the same folder. There will be multiple files, namely:
	`01_FILENAME_N_denoised.png` will contain a denoised version of the file if this option has been selected. Only really required for noisy images such as [high ISO](https://photographylife.com/what-is-iso-in-photography) photographs.
	`02_FILENAME_N_grayscale.png` will contain a grayscale version of the image, which is used for edge detection purposes.
	`03_FILENAME_N_entropyweights.png` contains the entropy weights of the image. This makes up 30% of the total weighting.
	`04_FILENAME_N_edgeweights.png` contains the edge weights of the image. This makes up 70% of the total weighting.
	`05_FILENAME_N_allpoissinsamples.png` contains every sample returned from the [Poisson disc sampling](https://www.jasondavies.com/poisson-disc/), prior to this being trimmed down to the number specified by `N`.
	`06_FILENAME_N_filteredpoissinsamples.png` contains only the `N` most relevant samples.
	`07_FILENAME_N_voronoi.png` is a full colour image of the original image but using the `N` Voronoi samples.
	`08_FILENAME_N_voronoicentroids.png` is the output required to use the [D3.js](https://d3js.org/) visualisation described in the section `Interactive Visualisation` below.

## The Effect Of `N` On The Output	
Obviously, the higher number of samples that are used to construct the output the more faithful it will be to the original image. The GIF below shows a run through of the affect of `N` on the output:
![Changing N](https://github.com/Cuahchic/map-voronoi/blob/master/img/effect_of_N.gif)


## Interactive Visualisation
In conjunction with the Python code above, [there is a D3.js visualisation](https://bl.ocks.org/Cuahchic/db05dda3b9abc246ca478eb48ce08e6c) which uses the `08` file outputted above. See below for a preview:

![D3 Preview](https://github.com/Cuahchic/map-voronoi/blob/master/img/d3blocks_preview.PNG)
