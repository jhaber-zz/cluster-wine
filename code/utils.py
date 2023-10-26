#!/usr/bin/env python
# coding: utf-8

"""
@title: Utility functions for data processing and analysis
@author: Jaren Haber, PhD
@date: September 16, 2023

"""


###############################################
# Initialize                 
###############################################

# Import packages
import requests
import shutil
from os.path import join
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


###############################################
# Define function(s)                 
###############################################

def get_unzip(URL, fpath):
    """
    Downloads zip-formatted file from specified URL and extracts it.
    
    Args:
        URL (str) -- points to zipped file
        fpath (str) -- path to folder in which to save output
    Returns:
        extracted zip file in main directory (one level up)
        (returns nothing directly)
        
    """
    
    zipped = requests.get(URL)
    zipped_bytes = zipped.content
    
    fname = URL.split('/')[-1] # Get name for extraction folder using URL
    outfile = join(fpath, fname)
    with open(outfile, "wb") as f:
        f.write(zipped_bytes)

    shutil.unpack_archive(outfile, extract_dir = fpath, format = 'zip')
    
    return


def plot_dendrogram(model, 
                    linkage='average',
                    figsize=(20,10), 
                    color_labels=None, 
                    yrange=None, 
                    xrange=None,
                    title="Dendrogram", 
                    save_plot=None, 
                    leaf_font_size=12,
                    **kwargs):
    
    '''Creates linkage matrix using word embedding model and then plots the dendrogram.
    
    Params:
        model (obj): clustering model from scikit-learn
        figsize (tuple): height x width of dendrogram
        save_plot (boolean or str): whether or not to save dendrogram to disk, True or False; if True, the filename for figure
        
        See documentation for info on other parameters:
            dendrogram: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    Returns: 
        Dendrogram via Scipy saved to disk
    '''
    
    global output_dir # Access filepath set outside function space

    # create the counts of samples under each node
    fig = plt.figure(figsize=figsize)
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    children, distances, counts = model.children_, model.distances_, np.array(counts)

    linkage_matrix = np.column_stack(
        [children, distances, counts]
    ).astype(float)
    
    cluster_threshold = 0.7*max(linkage_matrix[:,2])
    print(f'cluster threshold: {cluster_threshold}')

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, 
               leaf_font_size=leaf_font_size, 
               **kwargs)
    ax = plt.gca()
    
    # Customize plot layout by dendrogram orientation
    if orientation == 'top': # 'top' orientation
        #ax.set_ylim(yrange)
        plt.axhline(y=cluster_threshold, c='k', linestyle='dashed')
        ax.set_ylabel(f'{linkage.title()} linkage distance', fontsize = leaf_font_size)
        if color_labels:
            xlbls = ax.get_xmajorticklabels()
            for lbl in xlbls:
                lbl.set_color(color_labels.get(lbl.get_text(), 'black'))
                      
    elif orientation == 'right': # 'right' orientation
        #ax.set_xlim(xrange)
        plt.axvline(x=cluster_threshold, c='k', linestyle='dashed')
        ax.set_xlabel(f'{linkage.title()} linkage distance', fontsize = leaf_font_size)
        if color_labels:
            ylbls = ax.get_ymajorticklabels()
            for lbl in ylbls:
                lbl.set_color(color_labels.get(lbl.get_text(), 'black'))
    
    plt.title(title)
    if save_plot:
        assert isdir(output_dir), f"ERROR: {output_dir} not a valid target location for saving dendrogram"
        filepath = join(output_dir, str(save_plot) + ".png")
        plt.savefig(filepath, bbox_inches='tight', dpi = 300)
    plt.show()
    
    return model


def cluster_and_visualize(X, 
                          y, 
                          linkage='average', 
                          metric='l2',
                          figsize=(100, 10),
                          yrange=None,
                          xrange=None,
                          title=None,
                          save_plot=None,
                          orientation='top',
                          leaf_rotation=90,
                          leaf_font_size=12,
                          levels=5,
                          n_clusters=4,
                          distance_threshold=None):
    
    '''Creates hierarchical model, saves to disk, and passes to plot_dendrogram(), which creates the dendrogram and saves to disk. 
    
    Params:
        X (arr): standardized array of features (same length as y)
        y (arr): standardized array of outcomes (same length as X)
        figsize (tuple): height x width of dendrogram
        save_plot (boolean or str): whether or not to save dendrogram to disk, True or False; if True, the filename for figure
        
        See documentation for info on other parameters:
            AgglomerativeClustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
            dendrogram: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
        
    Returns: 
        clustering (obj): AgglomerativeClustering() model via scikit-learn saved to disk
    '''
    
    # Set values for a few params
    global models_dir # Access filepath set outside function space
    trunc = 'level' if levels else None
    n_clusters_text = '' if not n_clusters else ('_'+str(n_clusters))
        
    # Create hierarchical cluster model
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        distance_threshold=distance_threshold,
        metric=metric, 
        linkage=linkage, 
        compute_full_tree=True,
        compute_distances=True).fit(X)
    
    # Save clustering model to disk
    joblib.dump(clustering, models_dir+f'clustering{n_clusters_text}.joblib')
    
    # Create dendrogram
    plot_dendrogram(clustering, 
                    linkage=linkage,
                    figsize=figsize,
                    truncate_mode=trunc, 
                    title=title,
                    save_plot=save_plot,
                    p=levels, 
                    #show_contracted = True,
                    leaf_rotation=leaf_rotation, 
                    yrange=yrange,
                    xrange=xrange,
                    show_leaf_counts=True, 
                    distance_sort=True,
                    leaf_font_size=leaf_font_size, 
                    orientation=orientation)
                    #color_threshold=0.7)
        
        
def make_dendrogram(X,
                    y,
                    linkage,
                    metric,
                    fontsize:int,
                    n_clusters=4,
                    distance_threshold=None,
                    levels=5,
                    orientation='top',
                    save_plot=None):
    
    '''Derives parameters to create hierarchical clustering model and dendrograms and passes 
    to cluster_and_visualize() function, which creates the model and dendrogram and saves to disk. 
    
    Params:
        X (arr): standardized array of features (same length as y)
        y (arr): standardized array of outcomes (same length as X)
        dict_type (str): indicates type of dictionary used, either 'core' or 'expanded' (i.e., refined, period-specific)
        save_plot (boolean): whether or not to save dendrogram to disk, either True or False
        
        See documentation for info on other parameters:
            AgglomerativeClustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
            dendrogram: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    '''
    
    n_clusters_text = '' if not n_clusters else ('_'+str(n_clusters))
    
    # make plot/file title and print
    title = f'dendro{n_clusters_text}_{orientation}' #f'Dendrogram with n={n_clusters}'
    print(title)
    if save_plot:
        save_plot = title
        
    # derive dendrogram parameters
    if orientation == 'right':
        xrange=(0, 0.8)
        yrange=None
        figsize=(20, 20) # (10, 24)
        leaf_rotation=0
        leaf_font_size=fontsize # consider 10
        orientation='right'
    elif orientation == 'top':
        xrange=None
        yrange=(0, 0.8)
        figsize=(30, 7)
        leaf_rotation=90
        leaf_font_size=fontsize # consider 28
        orientation='top'
        
    # execute clustering and dendrogram function
    cluster_and_visualize(X, 
                          y, 
                          title=None,
                          linkage=linkage, 
                          metric=metric,
                          xrange=xrange,
                          yrange=yrange,
                          figsize=figsize, 
                          leaf_rotation=leaf_rotation,
                          leaf_font_size=leaf_font_size,
                          orientation=orientation,
                          n_clusters=n_clusters,
                          distance_threshold=distance_threshold,
                          levels=levels,
                          save_plot=save_plot)