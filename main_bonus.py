# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png ")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=18.5)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

# Function to handle mouse movements
def draw_segment(event,x,y,flags,param):
    global refPt, haveBG, haveFG, newMask
    # Store the point when mouse dragging is started
    if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
    # Draw line between mouse start and end. Update segmentation if both foreground and background markings are present.
    if event == cv2.EVENT_MBUTTONUP or event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        if fgColor == True:    # Mode = foreground
            haveFG = True
            cv2.line(image,refPt[0],refPt[1],(0,0,255),5)    # Draw Red line
            cv2.line(newMask,refPt[0],refPt[1],(0,0,255),5)
        elif bgColor == True:    # Mode = background
            haveBG = True
            cv2.line(image,refPt[0],refPt[1],(255,0,0),5)    # Draw Blue line
            cv2.line(newMask,refPt[0],refPt[1],(255,0,0),5)
        # If both foreground and backgroudn markings are available then generate the binary image with foreground and background
        if haveBG == True and haveFG == True:
            #Calculating Foreground and Background superpixels using marking "astronaut_marking.png"
            fg_segments, bg_segments = find_superpixels_under_marking(newMask, superpixels)
            
            #Calculating color histograms for FG
            fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
            
            #Calculating color histograms for BG
            bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)
            
            #Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
            graph_cut = do_graph_cut([fg_cumulative_hist,bg_cumulative_hist], [fg_segments,bg_segments], norm_hists, neighbors)
            
            mask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
            # mask is bool, conver 1 to 255 and 0 will remain 0 for displaying purpose
            mask = np.uint8(mask * 255)
            # Show binary image
            cv2.namedWindow('segment')
            cv2.imshow('segment',mask)

if __name__ == '__main__':
   
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    # Reading image
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

    #Calculating SLIC over image
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)
    norm_hists = normalize_histograms(color_hists)

    image = img.copy()

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_segment)

    fgColor = True
    bgColor = False
    haveFG = False
    haveBG = False
    #Create a mask with only markings used in segmentation in original image.
    newMask = np.zeros((img.shape), np.uint8)
    # Fill image with white color(set each pixel to red).
    newMask[:] = (255, 255, 255)
    while(1):
        cv2.imshow('image',image)
        key = cv2.waitKey(1) & 0xFF
        # Stop is user press 'c'
        if key == ord("q"):
            break;
        # Switch to background mode if user press 'c'
        if key == ord("b"):
            bgColor = True
            fgColor = False
        # Switch to foreground mode if user press 'c'
        if key == ord("f"):
            fgColor = True
            bgColor = False
        # Reset markings if user press 'r'
        if key == ord("r"):
            image = img.copy()
            bgColor = False
            fgColor = True
            haveFG = False
            haveBG = False
            newMask[:] = (255, 255, 255)
            cv2.destroyWindow("segment")
            
    cv2.destroyAllWindows()
