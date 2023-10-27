# What Feature Clusters Differentiate Wines?

This codebase clustering methods to find combinations of features that distinguish between and within wine types: red and white varieties of Portuguese _vinho verde_. Using a [cleaned wine dataset shared publicly by Cortez et al. (2009)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/), I extend on [this previous study](https://doi.org/10.1016/j.dss.2009.05.016) by addressing the following **research question:**

> What properties of each wine type differentiate clusters of differing levels of quality?

Rather than using machine learning or regression to identify which *individual* properties matter most, here I use clustering methods to analyze what *groups* of properties are most significant. In particular, I use visualization (t-SNE) and hierarchical clustering to determine most likely cluster counts, compute the clusters using *k*-means clustering, and compare their standardized attributes to find what properties "hang together" in distinct ways between clusters.


## Guide to codebase

*  Detailed findings and most of the code: [code/clustering.ipynb](code/clustering.ipynb)
*  Convenience functions, especially hierarchical clustering: [code/utils.py](code/utils.py)


## Summary of findings

* Clustering appears successful in capturing differences between wine types (red and white) and combinations of features within wine types connected with perceptions of quality
* Both wine types show two distinct clusters through hierarchical clustering (though these are not as distinct visually), with one cluster about a third of a standard deviation higher than the other
* The red wine clusters are differentiated by different groups of features than the white wine clusters: Red wines do better if they are more like white wines (lower acidity and sulphates), while better-tasting white wines are sweeter (higher sugar and density), lower in alcohol, and higher in sulfur dioxide.
* These findings reinforce some of the key features from Cortez et al. (2009)--e.g., sulphates for red wines and alcohol for white--while sidelining others--e.g., sulfur dioxide for red wines and sulphates for white. However, the main contribution here is about _groups_ of features, not individual features


## Example visualization

TBD


## Next steps

### Improving the codebase

* Convert repetitive analysis code into functions for reproducibility
* Look for outliers and consider how these might shape the results
* Compare the cluster results of hierarchical clustering and K-means clustering, e.g. by visualizing clusters and by silhouette scores
* Use cross-validation to get more robust sense for how important these clusters are in their impacts on perceived quality

### To learn about wines

* *Highest priority*: Evaluate the statistical significance of these feature clusters' impact on perceived wine quality, such as by using interaction effects in linear regression
* *Lower priority*: Determine which features are most important for determining quality differential using machine learning methods (e.g., Random Forest)
* *More ambitious*: Use other wine taste datasets to see if similar clusters hold there
