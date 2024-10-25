import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
Set of functions for visualising data and GP.
"""



def plot_K(K:torch.tensor, loc:str, file:str):
    """
    Plot covariance matrix K.
    args:
          K: Covariance matrix.
        loc: Location to store figure.
       file: File name.
    """

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = sns.heatmap(K, cmap='Blues')
    
    ax.set_xlabel(r"$y_{1}(t')$                                 $y_{2}(t')$")
    ax.set_ylabel(r"$y_{2}(t)$                                 $y_{1}(t)$")

    # change x and y labels
    xticks, xlabels = plt.xticks()
    tmax = int( len(xlabels)/2 )
    xlabels[tmax:] = xlabels[:tmax] 
    plt.xticks(xticks, xlabels)
    plt.yticks(xticks, xlabels)

    fig.savefig(loc + file + '.png')
    plt.show()
    
    return 0



def plot_GP(mu:np.array, K:np.array, Y:np.array, T:np.array, loc:str, file:str, train_indices=[]):
    """Plot mean function and 95% CI of GP alongside observations.
    args:
               mu: mean function.
                K: Covariance matrix.
                Y: (m*len(T)) data vector.
                T: sampled time points.
              loc: directory to save figure.
             file: file name for figure.
    train_indices: (optional) indices of T which were used in the training set. 
    """

    tmax = len(T)
    Y1 = Y[0:tmax]
    Y2 = Y[tmax:2*tmax]
    conf = 2*np.sqrt( np.diag(K) ) # 95% Confidence intervals
    

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout()
    
    if len(train_indices) == 0: # plot ground truth in same color
        ax1.plot(T, Y1, 'k*' )
        ax2.plot(T, Y2, 'k*' )
    else: 
        ax1.plot(T, Y1, 'k*', label='unobserved')
        ax1.plot(T[train_indices], Y1[train_indices], 'r*', label='observed')
        ax2.plot(T, Y2, 'k*', label='unobserved')
        ax2.plot(T[train_indices], Y2[train_indices], 'r*', label='observed')
        ax1.legend()
        ax2.legend()

    # plot means
    ax1.plot(T, mu[:tmax], 'b')
    ax2.plot(T, mu[tmax:2*tmax], 'b')

    # Shade between the lower and upper confidence bounds
    low1 = mu[:tmax] - conf[:tmax]
    upp1 = mu[:tmax] + conf[:tmax]
    low2 = mu[tmax:2*tmax] - conf[tmax:2*tmax]
    upp2 = mu[tmax:2*tmax] + conf[tmax:2*tmax]
    
    ax1.fill_between(T, low1, upp1, alpha=0.5)
    ax2.fill_between(T, low2, upp2, alpha=0.5)

    ax1.set_title('Y1')
    ax2.set_title('Y2')

    fig.savefig(loc + file + '.png')
    plt.show()

    return 0



def plot_latents(loc:str, file:str, truevids:np.array, truepath:np.array, nplots:int, reconvids:np.array=None):
    """
    Plots an array of input videos and reconstructions.
    args:
             loc: location to store file.
            file: filename.
        truevids: (batch, len(T), py, px) array of videos.
        truepath: (batch, len(T), 2) array of latent positions.
          nplots: number of cols of plot, col row is one video.
       reconvids: (batch, len(T), py, px) array of reconstructed videos.
    returns:
             ax: figure object with all plots.
    """

    if reconvids is not None:
        fig, ax = plt.subplots(3, nplots, figsize=(6, 8))
    else:
        fig, ax = plt.subplots(2, nplots, figsize=(6, 8))

    fig.tight_layout()
        
    for axi in ax:
        for axj in axi:
            axj.clear()

    _, tmax, _, _ = truevids.shape

    # get axis limits for the latent space
    xmin = np.min( [truepath[:nplots,:,0].min() -0.1, -2.5] )
    xmax = np.max( [truepath[:nplots,:,0].max() +0.1, 2.5] )
    ymin = np.min( [truepath[:nplots,:,1].min() -0.1, -2.5] )
    ymax = np.max( [truepath[:nplots,:,1].max() +0.1, 2.5] )

    def make_heatmap(vid):
        """
        args:
            vid: tmax, px, py
        returns:
            flat_vid: px, py
        """
        vid = np.array([(t+4)*v for t,v in enumerate(vid)])
        flat_vid = np.max(vid, 0)*(1/(4+tmax))
        return flat_vid

    

    def plot_set(i):
        # i is batch element = plot column

        # first row is original video
        tv = make_heatmap(truevids[i,:,:,:])
        ax[0][i].imshow(1-tv, origin='lower', cmap='Greys')
        ax[0][i].axis('off')

        # second row is trajectories
        ax[1][i].plot(truepath[i,:,0], truepath[i,:,1])
        ax[1][i].set_xlim([xmin, xmax])
        ax[1][i].set_ylim([ymin, ymax])
        ax[1][i].scatter(truepath[i,0,0], truepath[i,0,1], marker='o', c='C0')
        ax[1][i].scatter(truepath[i,-1,0], truepath[i,-1,1], marker='*', c='C0')

        # Third row is reconstructed video
        if reconvids is not None:
            rv = make_heatmap(reconvids[i,:,:,:])
            ax[2][i].imshow(1-rv, origin='lower', cmap='Greys')
            ax[2][i].axis('off') 
    
    for i in range(nplots):
        plot_set(i)

    fig.savefig(loc + file + '.png')
    plt.show()
    
    return ax



