import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
import matplotlib as mpl

# Set up Matplotlib styles
plt.style.use(['default'])
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.titleweight'] = 'normal'
mpl.rcParams['savefig.dpi'] = 350
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2

def plot_abundance_maps(predicted, ref, cls=0):
    """
    Plot predicted and reference abundance maps.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    
    im1 = axs[0].imshow(predicted, cmap='viridis')
    axs[0].set_title("Predicted")
    axs[0].axis('on')
    cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.05, pad=0.04)
    cbar1.set_label('Abundance')
    
    im2 = axs[1].imshow(ref, cmap='viridis')
    axs[1].set_title("Reference")
    axs[1].axis('on')
    cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.05, pad=0.04)
    cbar2.set_label('Abundance')
    
    plt.tight_layout()
    plt.show()

def plot_histograms(predictions, C_ref):
    """
    Plot histograms of predictions and reference values.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.hist(predictions, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Predictions Histogram')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(C_ref, bins=10, color='salmon', edgecolor='black', alpha=0.7)
    ax2.set_title('Reference Histogram')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def calculate_ssim(pred, ref):
    """
    Calculate SSIM between predicted and reference images.
    """
    pred_float = img_as_float(pred)
    ref_float = img_as_float(ref)
    data_range = np.max(ref_float)
    mask = ref_float > 0
    masked_ref = np.where(mask, ref_float, 0)
    masked_pred = np.where(mask, pred_float, 0)
    ssim_index, _ = ssim(masked_pred, masked_ref, data_range=data_range, full=True)
    return ssim_index

def plot_spectral_data(wav, X, y, n1, n2, n3, n4, a, b, thresholds=[10, 90]):
    """
    Plot spectral data with selected features
    """
    plt.figure(figsize=(10, 6))
    for t, c in zip(thresholds, ['k', 'r']):
        thresh = np.nanpercentile(y, t)
        mask = y > thresh
        plt.plot(wav, X[mask, :].T, alpha=0.01, color=c)
    
    plt.axvline(wav[n1], color='g', ls='-')
    if a != 0:
        plt.axvline(wav[n2], color='g', ls='--' if a > 0 else ':')
    
    plt.axvline(wav[n3], color='b', ls='-')
    if b != 0:
        plt.axvline(wav[n4], color='b', ls='--' if b > 0 else ':')
    
    plt.xlim(np.min(wav), np.max(wav))
    plt.ylim(0.2, 1.0)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Spectral Data with Selected Bands')
    plt.show()