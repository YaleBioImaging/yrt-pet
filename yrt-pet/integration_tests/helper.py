import os
import sys

import numpy as np

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(fold_py))
import pyyrtpet as yrt


# %% Test folders
def get_test_folders():
    env_data = "YRTPET_TEST_DATA"
    env_out = "YRTPET_TEST_OUT"

    fold_data_from_env = os.getenv(env_data)
    fold_out_from_env = os.getenv(env_out)
    if fold_data_from_env is None:
        raise RuntimeError("Environment variable " + env_data
                           + " need to be set")
    if fold_out_from_env is None:
        raise RuntimeError("Environment variable " + env_out
                           + " need to be set")
    if not os.path.exists(fold_data_from_env):
        raise RuntimeError("Path specified by " + env_data +
                           " does not exist.")
    if not os.path.exists(fold_out_from_env):
        os.mkdir(fold_out_from_env)

    fold_bin = os.path.join(os.path.dirname(__file__), '../executables')
    return fold_data_from_env, fold_out_from_env, fold_bin


# %% Helper test functions
def _test_subsets(scanner: yrt.Scanner, img_params: yrt.ImageParams,
                  projData: yrt.ProjectionData, **args):
    k = yrt.ProjectionOper(scanner, img_params, projData, **args)

    x = np.random.random([img_params.nz, img_params.ny, img_params.nx])
    y = np.random.random(projData.count())

    Ax = k.A(x)

    num_subsets = 4
    Ax_sub = np.zeros_like(Ax)
    for subset in range(num_subsets):
        k_sub = yrt.ProjectionOper(
            scanner,
            img_params, projData,
            idx_subset=subset, num_subsets=num_subsets,
            **args)
        Ax_s = k_sub.A(x)
        Ax_sub += Ax_s
        # Adjoint test
        if subset == 2:
            Aty_s = k_sub.At(y)
            np.testing.assert_allclose(
                np.sum(Ax_s * y), np.sum(x * Aty_s), atol=0, rtol=1e-3)
    # Check that combination of subsets yields full projection
    np.testing.assert_allclose(Ax, Ax_sub)


def _test_adjoint(scanner: yrt.Scanner, img_params: yrt.ImageParams,
                  projData: yrt.ProjectionData, **args):
    k = yrt.ProjectionOper(scanner, img_params, projData, **args)

    x = np.random.random([img_params.nz, img_params.ny, img_params.nx])
    y = np.random.random(projData.count())

    Ax = k.A(x)
    ATy = k.At(y)

    lhs = np.sum(Ax * y)
    rhs = np.sum(x * ATy)
    np.testing.assert_allclose(lhs, rhs, atol=0, rtol=1e-3)


def join_file_path_recursive(prefix, path):
    if isinstance(path, (list, tuple)):
        for i in range(len(path)):
            path[i] = join_file_path_recursive(prefix, path[i])
        return path
    return os.path.join(prefix, path)


def join_file_paths(dataset_paths, out_paths, ref_paths, util_paths,
                    fold_data: str, fold_out: str):
    for key, value in dataset_paths.items():
        dataset_paths[key] = join_file_path_recursive(
            fold_data, dataset_paths[key])
    for key, value in util_paths.items():
        util_paths[key] = join_file_path_recursive(fold_data, util_paths[key])
    for key, value in out_paths.items():
        out_paths[key] = join_file_path_recursive(fold_out, out_paths[key])
    for key, value in ref_paths.items():
        ref_paths[key] = join_file_path_recursive(
            os.path.join(fold_data, 'ref'), ref_paths[key])


# Test helpers
def get_test_summary(x0, x1):
    x0_max = np.max(np.abs(x0))
    x0_mean = np.mean(x0)
    x0_median = np.median(x0)

    rmse = np.sqrt(np.mean((x0 - x1)**2))
    nrmse = rmse / np.sqrt(np.mean(x0**2))
    linf = np.max(np.abs(x0 - x1))
    npix_diff = np.size(np.nonzero(x0 != x1)[0])
    return {'x0_max': x0_max,
            'x0_mean': x0_mean,
            'x0_median': x0_median,
            'rmse': rmse,
            'nrmse': nrmse,
            'linf': linf,
            'npix_diff': npix_diff}


def get_npix_diff(x0, x1):
    return np.size(np.nonzero(x0 != x1)[0])


def get_linf(x0, x1):
    return np.max(np.abs(x0 - x1))


def get_rmse(x0, x1):
    return np.sqrt(np.mean((x0 - x1)**2))


def get_nrmse(x0, x1):
    rmse = get_rmse(x0, x1)
    return rmse / np.sqrt(np.mean(x0**2))


# %% Datasets

# Prepend data path
fold_data, fold_out, fold_bin = get_test_folders()
