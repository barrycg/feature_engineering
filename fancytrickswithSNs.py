# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:24:28 2019

The file is aiming to enhance my understanding with those fancy tricks with 
simple numbers by going through the examples

@author: barry
"""

import pandas as pd

def binarization():
    #The table contains user-song-count triplets. Only non-zero counts are included.
    #hence, we just need to set the entire count column to 1.
    listen_count = pd.read_csv('data/train_triplets.txt.zip', header=None, delimiter='\t')
    listen_count[2] = 1

## quantization and binning
import json

import matplotlib.pyplot as plt
import seaborn as sns

def visualization():
    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    
#plot the histogram of the review counts
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    biz_df['review_count'].hist(ax=ax, bins=100)
    ax.set_yscale('log')
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Review Count', fontsize=14)
    ax.set_ylabel('Occurrence', fontsize=14)

# 固定宽度的量化计数
import numpy as np
def quantizingwithfixedwidthbins():
    small_counts = np.random.randint(0, 100, 20)
    np.floor_divide(small_counts, 10)

    large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495,
                        91897, 44, 28, 7971, 926, 122, 22222]
    np.floor(np.log10(large_counts))
    
# 分数位装箱
def quantile_binning():
    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    deciles = biz_df['review_count'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])
    
### visualize the deciles on the histogram
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    biz_df['review_count'].hist(ax=ax, bins=100)
    for pos in deciles:
        handle = plt.axvline(pos, color='r')
    ax.legend([handle], ['deciles'], fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Review Count', fontsize=14)
    ax.set_ylabel('Occurrence', fontsize=14)

def binningcountsbyquantiles():
    large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495,
                        91897, 44, 28, 7971, 926, 122, 22222]
    pd.qcut(large_counts, 4, labels=False)
    large_counts_series=pd.Series(large_counts)
    large_counts_series.quantile([0.25, 0.5, 0.75])
    
    
###log Transformation
def Visualizationoflogtransformation():
    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    biz_df['review_count'].hist(ax=ax1, bins=100)
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel('review_count', fontsize=14)
    ax1.set_ylabel('Occurence', fontsize=14)
    
    biz_df['review_count'].hist(ax=ax2, bins=100)
    ax2.tick_params(labelsize=14)
    ax2.set_xlabel('review_count', fontsize=14)
    ax2.set_ylabel('Occurence', fontsize=14)

def Visualizationoflogtransformation2():
    
    df = pd.read_csv('data/OnlineNewsPopularity.csv', delimiter=', ',engine='python')
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    df['n_tokens_content'].hist(ax=ax1, bins=100)
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel('Number of Words in Article', fontsize=14)
    ax1.set_ylabel('Number of Articles', fontsize=14)
    
    df['log_n_tokens_content'].hist(ax=ax2, bins=100)
    ax2.tick_params(labelsize=14)
    ax2.set_xlabel('Log of Number of Words', fontsize=14)
    ax2.set_ylabel('Number of Articles', fontsize=14)
    
# Log tranformation in Action
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

##errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
def predictbusinessratingsbyLT():
    
    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    
    biz_df['log_review_count'] = np.log10(biz_df['review_count'] + 1)
    
    m_orig = linear_model.LinearRegression()
    scores_orig = cross_val_score(m_orig, biz_df[['review_count']],
                                  biz_df['stars'], cv=10)
    m_log = linear_model.LinearRegression()
    scores_log = cross_val_score(m_log, biz_df[['log_review_count']],
                                 biz_df['stars'], cv=10)
    
    print("R-squared score without log transform: %0.5f (+/- %0.5f)" %
              (scores_orig.mean(), scores_orig.std() * 2))
    print("R-squared score with log transform: %0.5f (+/- %0.5f)" % 
          (scores_log.mean(), scores_log.std() * 2))

def predictarticlepopularitybyLT():
    df = pd.read_csv('data/OnlineNewsPopularity.csv', delimiter=', ', engine='python')

    ## Take the log transform of the 'n_tokens_content' feature, which
    ## represents the number of words (tokens) in a news article.
    df['log_n_tokens_content'] = np.log10(df['n_tokens_content'] + 1)

## Train two linear regression models to predict the number of shares
## of an article, one using the original feature and the other the
## log transformed version.
    m_orig = linear_model.LinearRegression()
    scores_orig = cross_val_score(m_orig, df[['n_tokens_content']], df['shares'], cv=10)
    m_log = linear_model.LinearRegression()
    scores_log = cross_val_score(m_log, df[['log_n_tokens_content']], df['shares'], cv=10)
    print("R-squared score without log transform: %0.5f (+/- %0.5f)" % (scores_orig.mean(), scores_orig.std() * 2))
    print("R-squared score with log transform: %0.5f (+/- %0.5f)" % (scores_log.mean(), scores_log.std() * 2))

#  KeyError: 'log_n_tokens_content's
def visualizingchangeofpopularity():
    
    df = pd.read_csv('data/OnlineNewsPopularity.csv', delimiter=', ', engine='python')
    fig2, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(df['n_tokens_content'], df['shares'])
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel('Number of Words in Article', fontsize=14)
    ax1.set_ylabel('Number of Shares', fontsize=14)
    
    ax2.scatter(df['log_n_tokens_content'], df['shares'])
    ax2.tick_params(labelsize=14)
    ax2.set_xlabel('Log of the Number of Words in Article', fontsize=14)
    ax2.set_ylabel('Number of Shares', fontsize=14)
    
def visualizaingchangeofprediction():
    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(biz_df['review_count'], biz_df['stars'])
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel('Review Count', fontsize=14)
    ax1.set_ylabel('Average Star Rating', fontsize=14)
    
    ax2.scatter(biz_df['log_review_count'], biz_df['stars'])
    ax2.tick_params(labelsize=14)
    ax2.set_xlabel('Log of Review Count', fontsize=14)
    ax2.set_ylabel('Average Star Rating', fontsize=14)
    
###Generalization of the Log Transform
from scipy import stats

### errrorrr ValueError: Data must be positive.
def BoxCoxofYelpBusinessReviewCount():
    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    
    biz_df['review_count'].min()
    rc_log = stats.boxcox(biz_df['review_count'], lmbda=0)
    rc_bc, bc_params = stats.boxcox(biz_df['review_count'])    
    print(bc_params)    

def VisualizingChangesOfReviewCounts():
    
    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    # original review count histogram
    biz_df['review_count'].hist(ax=ax1, bins=100)
    ax1.set_yscale('log')
    ax1.tick_params(labelsize=14)
    ax1.set_title('Review Counts Histogram', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Occurrence', fontsize=14)
    
    # review count after log transform
    biz_df['rc_log'].hist(ax=ax2, bins=100)
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=14)
    ax2.set_title('Log Transformed Counts Histogram', fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('Occurrence', fontsize=14)
    
    # review count after optimal Box-Cox transform
    biz_df['rc_bc'].hist(ax=ax3, bins=100)
    ax3.set_yscale('log')
    ax3.tick_params(labelsize=14)
    ax3.set_title('Box-Cox Transformed Counts Histogram', fontsize=14)
    ax3.set_xlabel('')
    ax3.set_ylabel('Occurrence', fontsize=14)
    
def ProbabilityPlotsAginstNormalDistribution():

    biz_file=open('data/yelp_academic_dataset.json')
    biz_df=pd.DataFrame(json.loads(x) for x in biz_file.readlines())
    biz_file.close()
    
    fig2, (ax1, ax2, ax3) = plt.subplots(3,1)
    prob1 = stats.probplot(biz_df['review_count'], dist=stats.norm, plot=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Probplot against normal distribution')
    prob2 = stats.probplot(biz_df['rc_log'], dist=stats.norm, plot=ax2)
    ax2.set_xlabel('')
    ax2.set_title('Probplot after log transform')
    prob3 = stats.probplot(biz_df['rc_bc'], dist=stats.norm, plot=ax3)
    ax3.set_xlabel('Theoretical quantiles')
    ax3.set_title('Probplot after Box-Cox transform')


### waiting to be checked again
import sklearn.preprocessing as preproc

def FeatureScaling():
    df = pd.read_csv('data/OnlineNewsPopularity.csv', delimiter=', ', engine='python')

# Look at the original data - the number of words in an article
    df['n_tokens_content'].as_matrix()

# Min-max scaling
    df['minmax'] = preproc.minmax_scale(df[['n_tokens_content']])
    df['minmax'].as_matrix()

# Standardization - note that by definition, some outputs will be negative
    df['standardized'] = preproc.StandardScaler().fit_transform(df[['n_tokens_content']])
    df['standardized'].as_matrix()

# L2-normalization
    df['l2_normalized'] = preproc.normalize(df[['n_tokens_content']], axis=0)
    df['l2_normalized'].as_matrix()

#visualizing
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
    fig.tight_layout()

    df['n_tokens_content'].hist(ax=ax1, bins=100)
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel('Article word count', fontsize=14)
    ax1.set_ylabel('Number of articles', fontsize=14)

    df['minmax'].hist(ax=ax2, bins=100)
    ax2.tick_params(labelsize=14)
    ax2.set_xlabel('Min-max scaled word count', fontsize=14)
    ax2.set_ylabel('Number of articles', fontsize=14)

    df['standardized'].hist(ax=ax3, bins=100)
    ax3.tick_params(labelsize=14)
    ax3.set_xlabel('Standardized word count', fontsize=14)
    ax3.set_ylabel('Number of articles', fontsize=14)

    df['l2_normalized'].hist(ax=ax4, bins=100)
    ax4.tick_params(labelsize=14)
    ax4.set_xlabel('L2-normalized word count', fontsize=14)
    ax4.set_ylabel('Number of articles', fontsize=14)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preproc


### OnlineNewsPopularity is exceptional errrorrrrrrrrrrrrrr
def interactionfeaturesinPredictions():
    
    df = pd.read_csv('data/OnlineNewsPopularity.csv', delimiter=', ', engine='python')
    print(df.columns)
'''
    X = df[features]
    y = df[['shares']]
    
    X2 = preproc.PolynomialFeatures(include_bias=False).fit_transform(X)
    X2.shape

### Create train/test sets for both feature sets
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = \
                train_test_split(X, X2, y, test_size=0.3, random_state=123)

    def evaluate_feature(X_train, X_test, y_train, y_test):
        ###Fit a linear regression model on the training set and score on the test set
        model = linear_model.LinearRegression().fit(X_train, y_train)
        r_score = model.score(X_test, y_test)
        return (model, r_score)

### Train models and compare score on the two feature sets
    (m1, r1) = evaluate_feature(X1_train, X1_test, y_train, y_test)
    (m2, r2) = evaluate_feature(X2_train, X2_test, y_train, y_test)
    print("R-squared score with singleton features: %0.5f" % r1)
    print("R-squared score with pairwise features: %0.10f" % r2)
 '''   

if __name__ == '__main__':
    
#    binarization()
#    visualization()
#    quantizingwithfixedwidthbins()
#    quantile_binning()
#    binningcountsbyquantiles()
#    Visualizationoflogtransformation()
#    Visualizationoflogtransformation2()
#    predictbusinessratingsbyLT()
#    predictarticlepopularitybyLT()
#    visualizingchangeofpopularity()
#    visualizaingchangeofprediction()
#    BoxCoxofYelpBusinessReviewCount()
#    VisualizingChangesOfReviewCounts()
#    ProbabilityPlotsAginstNormalDistribution()
#    FeatureScaling()
    interactionfeaturesinPredictions()
    
    
    
    