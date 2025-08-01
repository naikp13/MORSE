import numpy as np
from hylite import io, HyImage
from hylite.analyse import band_ratio
import rasterio
from rasterio.windows import from_bounds
import rasterio.warp
import os
from hklearn import Stack

def load_data(hydat_path=None, abund_path=None, train_stack_path=None, test_stack_path=None):
    """
    Load training data.
    """
    if hydat_path and abund_path:
        hydat = io.load(hydat_path)
        abund = io.load(abund_path)
        return hydat, abund, None, None
    elif train_stack_path and test_stack_path:
        train_stack = Stack.load(train_stack_path)
        test_stack = Stack.load(test_stack_path)
        return None, None, train_stack, test_stack
    else:
        raise ValueError("Must provide either hyperspectral/reference data paths. Use custom functions for other modality")

def preprocess_data(hydat=None, abund=None, train_stack=None, test_stack=None, cls=0):
    """
    Preprocess data (functions defined for hyper/multi-spectral). For other modality create a custom function
    """
    if hydat and abund:
        # Handle invalid values
        hydat.data[hydat.data <= 0.] = np.nan
        hydat.fill_holes()
        hydat.data = hydat.data.astype(np.float32)
        hydat.data = np.round(hydat.data, 4)

        abund.data[abund.data <= 0.] = np.nan
        abund.fill_holes()
        abund.data = abund.data.astype(np.float32)
        abund.data = np.round(abund.data, 4)

        # Reshape data
        X_data = hydat.data
        C_data = abund.data
        X = reshape(X_data)
        C = reshape(C_data)
        C_ref = C[:, cls]

        return X, C_ref, hydat.get_wavelengths(), None, None, None, None
    elif train_stack and test_stack:
        # Apply hull correction
        train_stack = train_stack.hc(hull={'sensor1': 'upper', 'sensor2': 'lower', 'sensor3': 'lower'})
        test_stack = test_stack.hc(hull={'sensor1': 'upper', 'sensor2': 'lower', 'sensor3': 'lower'})

        # Prepare data for GPU (optional)
        try:
            import cupy as cp
            train_data = {s: cp.array(train_stack.X(s)) for s in ['sensor1', 'sensor2', 'sensor3']}
            test_data = {s: cp.array(test_stack.X(s)) for s in ['sensor1', 'sensor2', 'sensor3']}
            y_train = cp.array(train_stack.y())
            y_test = cp.array(test_stack.y())
        except ImportError:
            train_data = {s: train_stack.X(s) for s in ['sensor1', 'sensor2', 'sensor3']}
            test_data = {s: test_stack.X(s) for s in ['sensor1', 'sensor2', 'sensor3']}
            y_train = train_stack.y()
            y_test = test_stack.y()

        return None, None, train_stack.get_wavelengths(), train_data, test_data, y_train, y_test
    else:
        raise ValueError("Must provide either hyperspectral/abundance or stack data. Use custom function for other modality")

def reshape(arr):
    """
    Reshape 3D array to 2D for model input.
    """
    num_rows, num_cols, num_bands = arr.shape
    reshp_arr = np.reshape(arr, (num_rows * num_cols, num_bands))
    return reshp_arr

def resample_images(sensor1_path, sensor2_path, outdir):
    """
    Resample abundance images to a common resolution.
    """
    paths = {'sensor1': sensor1_path, 'sensor2': sensor2_path}
    with rasterio.open(paths['sensor1']) as src1, rasterio.open(paths['sensor2']) as src2:
        bounds1 = src1.bounds
        bounds2 = src2.bounds
        x0 = max(bounds1.left, bounds2.left)
        y0 = max(bounds1.bottom, bounds2.bottom)
        x1 = min(bounds1.right, bounds2.right)
        y1 = min(bounds1.top, bounds2.top)
        common_transform, width, height = rasterio.warp.calculate_default_transform(
            src1.crs, src1.crs, width=src1.width, height=src1.height, left=x0, bottom=y0, right=x1, top=y1
        )
        for name, src in zip(['sensor1', 'sensor2'], [src1, src2]):
            window = from_bounds(x0, y0, x1, y1, src.transform)
            arr = src.read(window=window, out_shape=(src.count, height, width), resampling=rasterio.enums.Resampling.average)
            profile = src.profile.copy()
            profile.update({'height': height, 'width': width, 'transform': common_transform})
            outpath = os.path.join(outdir, f"{name}_resampled.tif")
            with rasterio.open(outpath, 'w', **profile) as dst:
                dst.write(arr)
            arr = np.moveaxis(arr, [0, 1, 2], [2, 1, 0])
            hyimg = io.HyImage(arr)
            io.save(os.path.join(outdir, f"{name}_am_resampled"), hyimg)