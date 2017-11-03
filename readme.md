# Image Segmentation Using Markings and Graph-Cut

Implemented code for semi-automatic binary segmentation based on SLIC superpixels and graph-cuts. 

# Static Segmentation:

For static segmentation, user needs to pass a mask of same size as of image with foreground and background markings. Foreground marking should be in Red color and Background makrings should be in Blue color. Run the following command to execute the segmentation:

> python main.py [original image Path] [marking image Path] [output directory ex. ./]

# Dynamic Segmenttaion:

For run time segmentation, an interactive window will popup. User can draw markings on the image. Red color represents the marking for Foreground and Blue colors represents the markings for Background. Once, atleast one marking is available for FG and BG, corresponding result will pop up in another window showing the binary mask. Mask is updated in real time as a line is drawn by the user.
By default Foreground mode is set. Use following keys to interact with window:

'f' - switch to Foreground mode

'b' - switch to Background mode

'r' - Reset markings

'q' - Quit

Python command:

> python segmentationVideo.py [original image Path]

Refer [Real Time Segmentation](https://github.com/geekyspartan/Image-segmentation-using-SLIC-superpixels-and-graph-cuts/blob/master/Real%20Time%20segmentation.mov) to see the working example.

# Team
[Renu Rani](https://github.com/techiepanda), [Anurag Arora](https://github.com/geekyspartan)
