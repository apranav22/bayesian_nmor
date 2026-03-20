import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates
import time
from datetime import datetime
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import MaxNLocator

# Data loading
@dataclass
class DataContext: 
    """Bundle of file strings that allow loading of the data we are interested in. 
    
    Stores paths to simulation and experiment data, interpolator cache, and handles
    timestamped result directories.
         
    Attrs:
    - sim_time: Path to simulation time axis csv file
    - sim_freq: Path to simulation frequency axis csv file
    - sim_by: Path to simulation By axis csv file
    - sim_intensities: List of simulation intensity file paths
    - exp_time: Path to experiment time axis csv file
    - exp_freq_axis: Path to experiment frequency bias axis csv file
    - exp_data: Path to experiment probe transmission data csv file
    - interpolator: Path to cached interpolator pickle file (or None)
    - save_path: Base directory for results (timestamped subdirectory auto-created)
    """
    sim_time: str
    sim_freq: str
    sim_by: str
    sim_intensities: list[str]
    exp_time: str
    exp_freq_axis: str
    exp_data: str
    save_path: str
    interpolator: Optional[str]
    Aligned: bool = False

    # Pulse parameters
    sim_pulse_thresh: float = 0.3
    exp_pulse_thresh: float = 0.3


    def __post_init__(self):
        """Generate timestamped save directory and create it."""
        timestamp = datetime.now().date().strftime("%Y_%m_%d")
        object.__setattr__(self, 'save_path', os.path.join(self.save_path, f'Test_{timestamp}'))
        os.makedirs(self.save_path, exist_ok=True)


@dataclass
class ParameterContext:
    """Bundle of runtime parameters for experiments and alt-opt runs.

    This centralizes commonly used knobs with sensible defaults so you can
    pass one object around instead of many separate arguments.

    Longitudinal/Y Estimation (common):
    - curr_time: Start time (s)
    - print_plot: Whether to show intermediate plots
    - zoom_factor: Adaptive zoom factor (> 1)
    - max_time: End time (s)
    - t_step: Measurement step (s)
    - B_unk_bound: Prior bound for unknown field (|B| <= bound)
    - init_resolution: Initial grid resolution
    - sigma_noise: Observation noise stddev
    - f_bias_offset_nuisance: Constant offset added to exp bias axis (Z only)

    AltOpt-only:
    - num_iter: Alternating optimization iterations
    - tol_bz / tol_by: early-stop tolerances
    - patience: consecutive stable iterations required to early-stop
    - Est_First_Z: whether to estimate Z first
    - fixed_bz_estimate / fixed_by_estimate: initial seeds
    """

    # Common experiment parameters
    curr_time: float = 5.0
    print_plot: bool = False
    zoom_factor: float = 1.1
    zoom_trigger_multiple: int = 400
    zoom_trigger_ratio: float = 0.2
    max_time: float = 30.0
    t_step: float = 0.2
    B_unk_bound_longitudinal: float = 1.0
    B_unk_bound_transverse_lower: float = 0.0
    B_unk_bound_transverse_upper: float = 0.5
    init_resolution_longitudinal: float = 0.001
    init_resolution_transverse: float = 0.0001
    sigma_noise_longitudinal: float = 0.1
    sigma_noise_transverse: float = 0.1
    f_bias_offset_nuisance: float = 0.0
    Test: str = f"Starting with {curr_time} us_T Step {t_step} us"

    #FIXME: T_STEP and sigma_noise must be set such that the variable list of inputs are also accepted

    # Pulse parameters
    sim_pulse_thresh: float = 0.3
    exp_pulse_thresh: float = 0.3
    
    # Adaptive experiment/grid
    kl_y_grid_size: int = 200

    # AltOpt parameters
    num_iter: int = 5
    tol_bz: float = 1e-8
    tol_by: float = 1e-8
    patience: int = 2

    Est_First_Z: bool = False
    fixed_bz_estimate: float = 0.0
    fixed_by_estimate: float = 0.0

    def validate(self) -> tuple[bool, list[str]]:
        """Basic parameter sanity checks.

        Returns (ok, problems). Problems contains human-readable strings.
        """
        problems: list[str] = []
        if self.zoom_factor <= 1.0:
            problems.append("zoom_factor must be > 1.0")
        if self.t_step <= 0:
            problems.append("t_step must be > 0")
        if self.max_time <= self.curr_time:
            problems.append("max_time must be > curr_time")
        if self.B_unk_bound <= 0:
            problems.append("B_unk_bound must be > 0")
        if self.init_resolution <= 0:
            problems.append("init_resolution must be > 0")
        if self.sigma_noise < 0:
            problems.append("sigma_noise must be >= 0")
        if self.num_iter <= 0:
            problems.append("num_iter must be > 0")
        if self.patience < 1:
            problems.append("patience must be >= 1")
        return (len(problems) == 0, problems)

class GPUInterpolator:
    def __init__(self, t_axis = None, f_axis = None, by_axis = None, data_cube = None):
        if data_cube is not None:
            self.t_axis = cp.asarray(t_axis)
            self.f_axis = cp.asarray(f_axis)
            self.by_axis = cp.asarray(by_axis)
            self.data_cube = cp.asarray(data_cube, dtype=cp.float64)
            self.dims = self.data_cube.shape
            self.nt, self.nz, self.ny = self.dims

            self.t_start = t_axis[0]
            self.f_start = f_axis[0]
            self.by_start = by_axis[0]

            self.t_step = t_axis[1] - t_axis[0]
            self.f_step = f_axis[1] - f_axis[0]
            self.by_step = by_axis[1] - by_axis[0]

            # Precompute the maximum valid indices for clamping
            self.max_t_idx = len(t_axis) - 1
            self.max_f_idx = len(f_axis) - 1
            self.max_by_idx = len(by_axis) - 1

            self.kernel = cp.ElementwiseKernel(
                in_params = 'float64 t_idx, float64 z_idx, float64 y_idx, raw float64 cube, int32 nt, int32 nz, int32 ny',
                out_params = 'float64 out',
                operation = """
                    int ti = max(0, min((int)floor(t_idx), nt - 2));
                    int zi = max(0, min((int)floor(z_idx), nz - 2));
                    int yi = max(0, min((int)floor(y_idx), ny - 2));

                    float dt = t_idx - ti;
                    float dz = z_idx - zi;
                    float dy = y_idx - yi;

                    int stride_t = nz*ny;
                    int stride_z = ny;
                    int stride_y = 1;

                    int idx000 = ti*stride_t + zi*stride_z + yi*stride_y;

                    float v000 = cube[idx000];
                    float v001 = cube[idx000 + 1];
                    float v010 = cube[idx000 + stride_z];
                    float v011 = cube[idx000 + stride_z + 1];
                    float v100 = cube[idx000 + stride_t];
                    float v101 = cube[idx000 + stride_t + 1];
                    float v110 = cube[idx000 + stride_t + stride_z];
                    float v111 = cube[idx000 + stride_t + stride_z + 1];

                    float c00 = v000*(1-dy) + v001*dy;
                    float c01 = v010*(1-dy) + v011*dy;
                    float c10 = v100*(1-dy) + v101*dy;
                    float c11 = v110*(1-dy) + v111*dy;

                    float c0 = c00*(1-dz) + c01*dz;
                    float c1 = c10*(1-dz) + c11*dz;

                    out = c0*(1-dt) + c1*dt;
                """,
                name = "trilinear_extrapolation_and_interpolation_kernel"
            )

    def to_indices(self, t_query, f_query, by_query):
        t_idx = (t_query - self.t_start) / self.t_step
        f_idx = (f_query - self.f_start) / self.f_step
        f_idx = cp.clip(f_idx, 0, self.nz - 1)
        idx = cp.searchsorted(self.by_axis, by_query, side='right') - 1
        idx = cp.clip(idx, 0, self.ny - 2)
        y0 = self.by_axis[idx]
        y1 = self.by_axis[idx + 1]
        by_idx = idx + (by_query - y0) / (y1 - y0)
        return t_idx, f_idx, by_idx
    
    def interpolate(self, t_query, f_query, by_query):  
        # Note: Query the flattened arrays directly to keep calculations simple and efficient
        t_idx, f_idx, by_idx = self.to_indices(t_query, f_query, by_query)
        # coords = cp.stack([t_idx, f_idx, by_idx], axis = 0)
        # predictions = map_coordinates(self.data_cube, coords, order=1, mode='nearest')
        return self.kernel(t_idx.astype(cp.float64), f_idx.astype(cp.float64), by_idx.astype(cp.float64), self.data_cube, self.nt, self.nz, self.ny)
    
    def save(self, filepath):
        np.savez(filepath, 
                 t_axis=cp.asnumpy(self.t_axis), 
                 f_axis=cp.asnumpy(self.f_axis), 
                 by_axis=cp.asnumpy(self.by_axis), 
                 metadata=np.array([
                self.t_start, self.t_step,
                self.f_start, self.f_step,
                self.by_start, self.by_step
                ]),
                 data_cube=cp.asnumpy(self.data_cube))
        
    @classmethod
    def load(cls, filepath):
        loaded = np.load(filepath)
        t_axis = cp.asarray(loaded['t_axis'])
        f_axis = cp.asarray(loaded['f_axis'])
        by_axis = cp.asarray(loaded['by_axis'])
        data_cube = cp.asarray(loaded['data_cube'], dtype = cp.float64)
        meta = loaded['metadata']

        #i dont need to pass time, by, bz because all that matters are the min and steps, which have been passed
        instance = cls()
        instance.data_cube = data_cube
        instance.by_axis = by_axis
        instance.nt, instance.nz, instance.ny = data_cube.shape
        instance.t_start, instance.t_step = float(meta[0]), float(meta[1])
        instance.f_start, instance.f_step = float(meta[2]), float(meta[3])
        instance.by_start, instance.by_step = float(meta[4]), float(meta[5])
        instance.max_t_idx, instance.max_f_idx, instance.max_by_idx = len(t_axis) - 1, len(f_axis) - 1, len(by_axis) - 1    
        instance.kernel = cp.ElementwiseKernel(
            in_params = 'float64 t_idx, float64 z_idx, float64 y_idx, raw float64 cube, int32 nt, int32 nz, int32 ny',
            out_params = 'float64 out',
            operation = """
                int ti = max(0, min((int)floor(t_idx), nt - 2));
                int zi = max(0, min((int)floor(z_idx), nz - 2));
                int yi = max(0, min((int)floor(y_idx), ny - 2));
                float dt = t_idx - ti;
                float dz = z_idx - zi;
                float dy = y_idx - yi;
                int stride_t = nz*ny;
                int stride_z = ny;
                int stride_y = 1;
                int idx000 = ti*stride_t + zi*stride_z + yi*stride_y;
                float v000 = cube[idx000];
                float v001 = cube[idx000 + 1];
                float v010 = cube[idx000 + stride_z];
                float v011 = cube[idx000 + stride_z + 1];
                float v100 = cube[idx000 + stride_t];
                float v101 = cube[idx000 + stride_t + 1];
                float v110 = cube[idx000 + stride_z + stride_t];
                float v111 = cube[idx000 + stride_z + stride_t + 1];
                float c00 = v000*(1-dy) + v001*dy;
                float c01 = v010*(1-dy) + v011*dy;
                float c10 = v100*(1-dy) + v101*dy;
                float c11 = v110*(1-dy) + v111*dy;
                float c0 = c00*(1-dz) + c01*dz;
                float c1 = c10*(1-dz) + c11*dz;
                out = c0*(1-dt) + c1*dt;
                """,
                name = "trilinear_extrapolation_and_interpolation_kernel"
            )
        return t_axis, f_axis, by_axis, instance

# default object: 
DEFAULT_PARAMS = ParameterContext()

DATA_DIR = r"DataFiles_to_Dinesh_Pranav\Data_files\Simulation\Dataset_7"
DATASET3 = DataContext(
    sim_time = os.path.join(DATA_DIR, "t_array.csv"),
    sim_freq = os.path.join(DATA_DIR, "delz_MHz.csv"),
    sim_by = os.path.join(DATA_DIR, "dely_MHz.csv"),
    sim_intensities = [],
    exp_time = r"DataFiles_to_Dinesh_Pranav\Data_files\Experiment\t_exp.csv",
    exp_freq_axis = r"DataFiles_to_Dinesh_Pranav\Data_files\Experiment\Y_MHz_Exp.csv",
    exp_data = r"DataFiles_to_Dinesh_Pranav\Data_files\Experiment\Probe_trans_Exp.csv",
    save_path = r"Results\Dataset_7",
    interpolator=r"DataFiles_to_Dinesh_Pranav\Data_files\Simulation\Dataset_7\gpu_sim_interpolator.npz",
    Aligned = False
)



## Preprocessing
def find_pulse_start(trace, t_axis, threshold):
    idx = np.where(trace > threshold)[0]
    if len(idx) > 0:
        return t_axis[idx[0]], idx[0]
    return t_axis[0], 0


def load_simulation_cube(config: DataContext):
    """Load simulation axes and cube using DataContext paths."""
    print("...Loading Simulation Cube Data...")
    to = time.time()
    t_axis = pd.read_csv(config.sim_time, header=None).values.flatten()
    f_axis = pd.read_csv(config.sim_freq, header=None).values.flatten()
    by_axis = pd.read_csv(config.sim_by, header=None).values.flatten()

    print(
        f"  > Axes Loaded: Time[{len(t_axis)}], Freq[{len(f_axis)}], By[{len(by_axis)}]"
    )

    t_sim_0, _ = 0.0, 0
    matrix_list = []
    for fname in config.sim_intensities:
        data = pd.read_csv(fname, header=None).values

        if data.shape != (len(t_axis), len(f_axis)):
            data = data.T
        if len(matrix_list) == 0:
            scale = 1 / (np.max(data) - np.min(data))
            calibrated_data = (data - np.min(data)) * scale
            calibrated_data = np.clip(calibrated_data, 0.0, 1.0)
            t_sim_0, _ = find_pulse_start(calibrated_data[:, 0], t_axis, config.sim_pulse_thresh)

            t_axis_0 = t_axis[_:] - t_sim_0

        data = data[_:, :]
        scale = 1 / (np.max(data) - np.min(data))
        calibrated_data = (data - np.min(data)) * scale
        calibrated_data = np.clip(calibrated_data, 0.0, 1.0)
        matrix_list.append(calibrated_data)

    cube = np.stack(matrix_list, axis=2)
    print(f"  > Sim Start: {t_sim_0:.4f}s")
    print(f"  > Cube Built. Final Shape: {cube.shape}")
    print("Time taken to load sim data:", time.time() - to)
    return t_axis_0, f_axis, by_axis, cube


def load_experiment(config: DataContext, Aligned=DATASET3.Aligned):
    to = time.time()
    print(f"Loading Experiment from {config.exp_data}...")
    t = pd.read_csv(config.exp_time, header=None).values.flatten()
    f_bias = pd.read_csv(config.exp_freq_axis, header=None).values.flatten()
    raw_data = pd.read_csv(config.exp_data, header=None).values

    if raw_data.shape != (len(t), len(f_bias)):
        raw_data = raw_data.T

    if not Aligned:
        scale = 1 / (np.max(raw_data) - np.min(raw_data))
        calibrated_data = (raw_data - np.min(raw_data)) * scale
        calibrated_data = np.clip(calibrated_data, 0.0, 1.0)
        t_exp_0, __ = find_pulse_start(calibrated_data[:, 0], t, config.exp_pulse_thresh)

        t = t[__:] - t_exp_0
        raw_data = raw_data[__:,]
        print(f"  > Exp Start: {t_exp_0:.4f}s")
    
    '''#this is for the new dataset frmo 13/02 only FIXME
    t = t * 1e6
    t_index = np.where(np.asarray(t)>70)[0][0]
    f_index_1 = np.where(np.asarray(f_bias)<-1.5)[0][0]
    f_index_2 = np.where(np.asarray(f_bias)>1.5)[0][0]
    t = t[:t_index]
    raw_data = raw_data[:t_index,:]'''
    calibrated_data= calibration(raw_data, np.min(raw_data), np.max(raw_data))
   
    print(time.time() - to, "was the time taken to load exp data.")
    return t, f_bias, calibrated_data


def calibration(data, v_dark, v_max):
    scale = 1 / (v_max - v_dark)
    calibrated_data = (data - v_dark) * scale
    return calibrated_data


def plot_heat_and_surface(
    X_vec,
    Y_vec,
    Z_arr,
    title_prefix="Data",
    estimation_mode="longitudinal",
    trajectory_mode=False,
    traj_bias=[],
    curr_time=5.0,
):
    Yg = Yg * 0.7 #FIXME Mhz to Gauss conversion for plotting
    Xg, Yg = np.meshgrid(X_vec, Y_vec, indexing="ij")  # note: meshgrid order (cols = y)
    # heatmap
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(Xg, Yg, Z_arr, shading="auto")
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Applied Bias Field (Gauss)")
    if trajectory_mode:
        curr_time_index = np.searchsorted(X_vec, curr_time)
        traj_bias = np.repeat(
            traj_bias, (X_vec.shape[0] - curr_time_index - 1) // len(traj_bias)
        )
        traj_bias = np.asarray(traj_bias)*0.7 #FIXME Mhz to Gauss conversion for plotting
        plt.plot(
            X_vec[curr_time_index] + X_vec[: len(traj_bias)],
            traj_bias,
            color="red",
            marker="o",
            markersize=5,
            label="Trajectory",
        )
    plt.colorbar(label="Probe Transmitted Intensity")
    plt.title(f"{title_prefix} heatmap (rows=time, cols=freq)")
    plt.tight_layout()
    plt.show()
    # 3D surface (might be heavy)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Yg, Xg, Z_arr, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("Applied Bias Field (Gauss)")
    ax.set_ylabel("Time (microseconds)")
    ax.set_zlabel("Probe Transmitted Intensity")
    plt.title(f"{title_prefix} 3D surface")
    plt.tight_layout()
    plt.show()


def plot_slice(by_idx, file_path):
    t_sim, f_sim, by_sim, sim_cube = load_simulation_cube(file_path)
    plt.figure(figsize=(10, 6))
    data_slice = sim_cube[:, :, by_idx].T

    plt.imshow(
        data_slice,
        aspect="auto",
        origin="lower",
        extent=(t_sim[0], t_sim[-1], f_sim[0], f_sim[-1]),
        cmap="viridis",
    )
    plt.colorbar(label="Intensity")
    plt.title(f"Simulation Slice at By = {by_sim[by_idx]:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

    # interpolator and interpolation validator for Y field estimation


def get_final_interpolator(config: Optional[DataContext], params: ParameterContext):
    # if the config DataContext stringblob has an interpolator path, use that to load the interpolator otherwise generate.
    print("...Building Final Interpolator...")
    if not os.path.exists(config.interpolator):
        to = time.time()
        t_sim, f_sim, by_sim, cube = load_simulation_cube(config)
        full_interp = GPUInterpolator(t_sim, f_sim, by_sim, cube)
        full_interp.save(config.interpolator)
        t_sim, f_sim, by_sim = cp.asarray(t_sim), cp.asarray(f_sim), cp.asarray(by_sim)
    else:
        to = time.time()
        t_sim, f_sim, by_sim, full_interp = GPUInterpolator.load(config.interpolator)  
    print("Time taken to load Interpolator:", time.time() - to)
    return t_sim, f_sim, by_sim, full_interp

#interpolator predictions
def get_predictions_batch(interpolator, t_pts, z_vals, y_grid):   
    #the two elif conditions are for the case of querying in the longitudinal estimation run
    #with the first elif being used by the KL divergence step and second being used by the likelihood
    #distinction necessary because of the different array shapes based on context
    if (cp.isscalar(y_grid) or y_grid.ndim == 0) and t_pts.ndim == 0:
        # T_mesh, Z_mesh = np.meshgrid(t_pts, z_vals, indexing='ij')
        T_mesh = cp.full_like(z_vals, t_pts, dtype=float)
        Z_mesh = z_vals
        Y_mesh = cp.full_like(T_mesh, y_grid)
        predictions = interpolator.interpolate(T_mesh.ravel(), Z_mesh.ravel(), Y_mesh.ravel())
        return predictions.reshape(T_mesh.shape)
    
    #for longiduinal case, i need to make the edit here for the marginalisation of Bz FIXME: Havent done marginalisation part for KL divergence, implement that as well
    elif (cp.isscalar(y_grid) or y_grid.ndim == 0) and z_vals.ndim == 2:
        # T_mesh, Z_mesh = np.meshgrid(t_pts, z_vals, indexing='ij')
        # Y_mesh = np.full_like(T_mesh, y_grid)
        T_array = t_pts[:,None,None]
        Z_mesh = z_vals[None,:,:]
        final_shape = (T_array.shape[0], Z_mesh.shape[1], Z_mesh.shape[2])
        T_mesh = cp.broadcast_to(T_array, final_shape)
        Z_mesh = cp.broadcast_to(Z_mesh, final_shape)
        Y_mesh = cp.full_like(T_mesh, y_grid)
        predictions = interpolator.interpolate(T_mesh.ravel(), Z_mesh.ravel(), Y_mesh.ravel())
        return predictions.reshape(final_shape)    
    
    elif (cp.isscalar(y_grid) or y_grid.ndim == 0):
        T_array = t_pts[:,None]
        Z_mesh = z_vals[None,:]
        final_shape = (T_array.shape[0], Z_mesh.shape[1])
        T_mesh = cp.broadcast_to(T_array, final_shape)
        Z_mesh = cp.broadcast_to(Z_mesh, final_shape)
        Y_mesh = cp.full_like(T_mesh, y_grid)
        predictions = interpolator.interpolate(T_mesh.ravel(), Z_mesh.ravel(), Y_mesh.ravel())
        return predictions.reshape(final_shape)       
    
    #FIXME; write get prediction for Likelihood for transverse likelihood here, Bz has 1 dimension instead of the zero earlier 
    # y grid is 1 dim, b grid is 1 dim, and time is also 1 dim
    elif t_pts.ndim == 1 and y_grid.ndim == 1 and z_vals.ndim == 1:
        T_array = t_pts[:,None,None]
        Z_mesh = z_vals[None,:,None]     
        Y_mesh = y_grid[None,None,:]
        final_shape = (T_array.shape[0], z_vals.shape[0], y_grid.shape[0])
        T_mesh = cp.broadcast_to(T_array, final_shape)
        Z_mesh = cp.broadcast_to(Z_mesh, final_shape)
        Y_mesh = cp.broadcast_to(Y_mesh, final_shape)
        predictions = interpolator.interpolate(T_mesh.flatten(), Z_mesh.flatten(), Y_mesh.flatten())
        return predictions.reshape(T_mesh.shape)
    
    else:
        T_array = cp.asarray([t_pts])[:,None,None]
        Z_mesh = z_vals[None,:,None]     
        Y_mesh = y_grid[None,None,:]
        final_shape = (T_array.shape[0], z_vals.shape[0], y_grid.shape[0])
        T_mesh = cp.broadcast_to(T_array, final_shape)
        Z_mesh = cp.broadcast_to(Z_mesh, final_shape)
        Y_mesh = cp.broadcast_to(Y_mesh, final_shape)
        predictions = interpolator.interpolate(T_mesh.flatten(), Z_mesh.flatten(), Y_mesh.flatten())
        return predictions.reshape(final_shape[1:])

# Likelihood and KL for Longtudinal Estimation
def calculate_likelihood_gpu(
    y_meas, t_sim_abs, curr_bias, candidate_biases, y_grid, b_grid_gpu, sim_spline, sigma_noise = DEFAULT_PARAMS.sigma_noise_longitudinal
):
    to = time.time()
    candidate_biases_gpu = cp.array(candidate_biases)
    curr_bias_index = int(cp.where(candidate_biases_gpu == curr_bias)[0][0])
    start_idx = max(0, curr_bias_index - 1)
    end_idx = min(len(candidate_biases_gpu)-1, curr_bias_index + 2)
    f_total_query = b_grid_gpu[:, None] + candidate_biases[None, start_idx:end_idx]
    
    marginalisation_prob_density = cp.zeros_like(candidate_biases_gpu[start_idx:end_idx])
    marginalisation_prob_density = marginalisation_prob_density + 1/(candidate_biases_gpu[end_idx-1] - candidate_biases_gpu[start_idx]) # set the current bias to zero so that it is not included in marginalisation
    marginalisation_prob_density = marginalisation_prob_density*(candidate_biases_gpu[1]-candidate_biases_gpu[0])

    y_theory_gpu = cp.asarray(get_predictions_batch(sim_spline, t_sim_abs, f_total_query, y_grid))
    y_meas_gpu = cp.asarray(y_meas)

    resid_sq = (y_meas_gpu[:, None, None] - y_theory_gpu)**2
    sse = cp.sum(resid_sq, axis=0)

    log_L = -sse / (2 * sigma_noise**2)
    L = cp.exp(log_L - cp.max(log_L))
    L = L*marginalisation_prob_density[None, :]
    L = cp.sum(L, axis=1)
    log_L = cp.log(L + 1e-15) 
    del y_theory_gpu, y_meas_gpu, resid_sq, L
    cp.get_default_memory_pool().free_all_blocks()
    if cp.any(cp.isnan(log_L)):
        print("WARNING: NaNs detected in Likelihood!")
        print(f"Min SSE: {cp.min(sse)}, Max LogL: {cp.max(log_L)}")

    print(time.time() - to,"was the time taken to compute the likelihood.|", end=" ")
    return log_L - cp.max(log_L)

def calculate_kl_divergence_gpu(
    posterior_gpu,
    b_grid_gpu,
    y_grid,
    sim_spline,
    t_next_start,
    t_next_end,
    candidate_biases,
    batch_size=10,
    y_grid_size=DEFAULT_PARAMS.kl_y_grid_size,
    sigma_noise=DEFAULT_PARAMS.sigma_noise_longitudinal
):
    to = time.time()
    # hypothetical signal
    y_grid_gpu = cp.linspace(0, 1, y_grid_size)

    # Time point for prediction (Midpoint of next step), rather than doing this across all time, do it for one time stamp, wont make a big difference IG FIXME
    t_mid = (t_next_start + t_next_end) / 2.0

    n_candidates = len(candidate_biases)
    n_grid = len(b_grid_gpu)
    # dB = b_grid_gpu[1] - b_grid_gpu[0]
    candidate_biases_gpu = cp.array(candidate_biases)
    expected_kl_values = cp.zeros_like(candidate_biases_gpu)

    posterior_gpu_reshaped = posterior_gpu.reshape(1, 1, n_grid)

    for i in range(0, n_candidates, batch_size):
        end = min(i+batch_size, n_candidates)
        batch_candidates = candidate_biases[i:end]
        B_total_batch = batch_candidates[:,None] + b_grid_gpu[None, :]
        mu_gpu = get_predictions_batch(sim_spline, t_mid, B_total_batch, y_grid)
        diff_sq = (y_grid_gpu[None,:,None] - mu_gpu[:,None,:])**2
        L_tensor = cp.exp(-diff_sq / (2 * sigma_noise**2))
        #P_y = cp.sum(L_tensor * posterior_gpu_reshaped, axis=2) * dB
        P_y = cp.trapz(L_tensor * posterior_gpu_reshaped, x = b_grid_gpu, axis=2) 
        Posterior_tensor = (L_tensor * posterior_gpu_reshaped) / (P_y[:, :, None] + 1e-15)
        log_term = cp.log(cp.clip(Posterior_tensor, 1e-15, None) / cp.clip(posterior_gpu_reshaped, 1e-15, None)) 
        integrand = Posterior_tensor * log_term
        integrand = cp.where(Posterior_tensor < 1e-15, 0, integrand)  # Avoid log(0) issues by zeroing out contributions where Posterior is negligible
        KL_per_meas = cp.trapz(integrand, x = b_grid_gpu,axis=2)
        batch_expected_kl = cp.trapz(P_y * KL_per_meas, x = y_grid_gpu, axis=1)
        expected_kl_values[i:end] = batch_expected_kl
        del L_tensor,Posterior_tensor,diff_sq, log_term, KL_per_meas, batch_expected_kl
        cp.get_default_memory_pool().free_all_blocks()

    print(
        time.time() - to, "was the time taken to compute the KL optimisation.", end=" "
    )
    return candidate_biases[int(cp.argmax(expected_kl_values))], expected_kl_values


# KL and Likelihood for Y field Estimation
def calculate_likelihood_by(
    y_meas,
    t_chunk_sim,
    current_bias_z,
    candidate_biases,
    by_grid_gpu,
    sim_interp,
    sigma_noise=DEFAULT_PARAMS.sigma_noise_transverse,
    fixed_bz_estimate=DEFAULT_PARAMS.fixed_bz_estimate
):
    to = time.time()
    curr_bias_idx = int(cp.where(candidate_biases == current_bias_z)[0][0])
    start_idx = max(0, curr_bias_idx - 1)
    end_idx = min(len(candidate_biases)-1, curr_bias_idx + 2)
    z_total = fixed_bz_estimate + candidate_biases[start_idx:end_idx]
    y_pred_gpu = get_predictions_batch(sim_interp, t_chunk_sim, z_total, by_grid_gpu)
    dy_pred_gpu = cp.diff(y_pred_gpu, axis=0)

    marginalisation_prob_density = cp.zeros_like(candidate_biases[start_idx:end_idx])
    marginalisation_prob_density = marginalisation_prob_density + 1/(candidate_biases[end_idx-1] - candidate_biases[start_idx]) # set the current bias to zero so that it is not included in marginalisation
    marginalisation_prob_density = marginalisation_prob_density * (candidate_biases[1]-candidate_biases[0])

    y_meas_gpu = cp.asarray(y_meas)
    dy_meas_gpu = cp.diff(y_meas_gpu, axis=0)
    resid_sq = (y_meas_gpu[:, None, None] - y_pred_gpu)**2

    sse = cp.sum(resid_sq, axis=0)
    log_L = -sse/(2* sigma_noise**2)
    L = cp.exp(log_L - cp.max(log_L))
    L = L*marginalisation_prob_density[:, None]
    L = cp.sum(L, axis=0)
    log_L = cp.log(L + 1e-15) 
    del y_pred_gpu, y_meas_gpu, resid_sq, L
    cp.get_default_memory_pool().free_all_blocks()
    print(time.time() - to,"was the time taken to compute the Transverse Likelihood calculation. |", end=" ")
    return log_L - cp.max(log_L)


def calculate_kl_by(
    posterior_gpu,
    by_grid_gpu,
    sim_interp,
    t_next_start,
    t_next_end,
    candidate_biases_z,
    batch_size=20,
    y_grid_size=DEFAULT_PARAMS.kl_y_grid_size,
    sigma_noise=DEFAULT_PARAMS.sigma_noise_transverse,
    fixed_bz_estimate=DEFAULT_PARAMS.fixed_bz_estimate,
):
    t_start = time.time()
    t_mid = (t_next_start + t_next_end) / 2.0
    z_candidates = fixed_bz_estimate + candidate_biases_z
    y_grid_gpu = cp.linspace(0, 1, y_grid_size)
    n_candidates = len(z_candidates)
    n_grid = len(by_grid_gpu)
    candidate_biases_gpu = cp.array(z_candidates)
    expected_kl = cp.zeros_like(candidate_biases_gpu)

    posterior_gpu_reshaped = posterior_gpu.reshape(1, 1, n_grid)

    for i in range(0, n_candidates, batch_size):
        end = min(i+batch_size, n_candidates)
        batch_candidates = z_candidates[i:end]
        mu_matrix_gpu = get_predictions_batch(sim_interp, t_mid, batch_candidates, by_grid_gpu)

        diff_sq = (mu_matrix_gpu[:,None,:] - y_grid_gpu[None, :, None])**2
        L_tensor = cp.exp(-diff_sq / (2 * sigma_noise**2))
        P_y = cp.trapz(L_tensor * posterior_gpu_reshaped, x = by_grid_gpu, axis=2) 
        Posterior_tensor = (L_tensor * posterior_gpu_reshaped) / (P_y[:, :, None] + 1e-15)
        log_term = cp.log(cp.clip(Posterior_tensor, 1e-15, None) / cp.clip(posterior_gpu_reshaped, 1e-15, None))
        integrand = Posterior_tensor * log_term
        integrand = cp.where(Posterior_tensor < 1e-15, 0, integrand)  # Avoid log(0) issues by zeroing out contributions where Posterior is negligible
        KL_per_meas = cp.trapz(integrand, x = by_grid_gpu,axis=2)
        batch_expected_kl = cp.trapz(P_y * KL_per_meas, x = y_grid_gpu, axis=1)
        expected_kl[i:end] = batch_expected_kl
        del L_tensor,Posterior_tensor,diff_sq, log_term, KL_per_meas, batch_expected_kl
        cp.get_default_memory_pool().free_all_blocks()

    best_idx = int(cp.argmax(expected_kl))
    best_bias = candidate_biases_z[best_idx]
    print("KL Calculation Time:", time.time() - t_start, end = " ")
    return best_bias, expected_kl


# zoom and summary stats - need to replace these values from parameter context FIXME
def check_and_apply_zoom(
    posterior_gpu: cp.ndarray,
    b_grid_gpu: cp.ndarray,
    current_res: float,
    zoom_factor: float = 1.5,
    zoom_trigger_multiple: int = 400,
    zoom_trigger_ratio: float = 0.2,
    initial_resolution=0.00025,
    B_unk_bound_longitudinal=1,
) -> tuple:
    # compute B_unk range
    B_unk_init_range = (-B_unk_bound_longitudinal, B_unk_bound_longitudinal)

    # PDF -> CDF
    to = time.time()
    dB: cp.ndarray = b_grid_gpu[1:] - b_grid_gpu[:-1]
    pdf_mass = cp.zeros_like(posterior_gpu)
    pdf_mass[:-1] = posterior_gpu[:-1] * dB
    pdf_mass[-1] = posterior_gpu[-1] * dB[-1]
    cdf = cp.cumsum(pdf_mass)
    cdf = cdf/cdf[-1]  # Ensure it ends exactly at 1.0

    #need a reset posterior condition incase cdf is zero everywhere
    if cdf[-1] == 0 or cdf[-1] < 1e-10:
        print("Warning: CDF is zero everywhere. Resetting posterior to uniform distribution.")
        posterior_gpu = cp.full_like(posterior_gpu, (1 / 2 / B_unk_bound_longitudinal), dtype=posterior_gpu.dtype)
        dB = b_grid_gpu[1:] - b_grid_gpu[:-1]
        pdf_mass = cp.zeros_like(posterior_gpu)
        pdf_mass[:-1] = posterior_gpu[:-1] * dB
        pdf_mass[-1] = posterior_gpu[-1] * dB[-1]
        cdf = cp.cumsum(pdf_mass)
        cdf = cdf/ cdf[-1] # Ensure it ends exactly at 1.0

    # Finding region where majority of the probability is concentrated
    for i in range(5, 10):
        a = 10 ** (-i)
        idx_01 = int(cp.searchsorted(cdf, cp.array(a)))
        idx_99 = int(cp.searchsorted(cdf, cp.array(1 - a)))
        idx_01 = int(max(0, idx_01))
        idx_99 = int(min(len(b_grid_gpu) - 1, idx_99))
        if idx_99 - idx_01 > 0:
            break
    else:
        idx_01 = int(max(0, idx_01 - 50))
        idx_99 = int(min(len(b_grid_gpu) - 1, idx_99 + 50))

    del cdf
    cp.get_default_memory_pool().free_all_blocks()

    # total_points = len(b_grid_gpu)
    # Check Trigger
    width_idx = idx_99 - idx_01
    new_res = current_res
    if new_res <= 0:
        print("new_res:", cp.min(new_res))
        raise ValueError("Resolution is less than zero")
    new_b_grid_gpu = cp.copy(b_grid_gpu)
    new_posterior_gpu = cp.copy(posterior_gpu)
    del b_grid_gpu, posterior_gpu
    cp.get_default_memory_pool().free_all_blocks()

    if (
        (
            (new_res * zoom_trigger_multiple)
            > (new_b_grid_gpu[idx_99] - new_b_grid_gpu[idx_01])
        )
        or width_idx < 100
        or len(new_b_grid_gpu) >= 15000
    ):
        zoom_index = 0
        while (
            (
                (new_res * zoom_trigger_multiple)
                > (new_b_grid_gpu[idx_99] - new_b_grid_gpu[idx_01])
            )
            or width_idx < 100
            or len(new_b_grid_gpu) >= 15000
        ) and zoom_index < 5:
            print(
                f"  [ZOOM] Triggered! Mass concentrated in {width_idx} points.|",
                end=" ",
            )

            # Define New Grid resolution
            new_res = np.abs(new_res) / zoom_factor
            fine_b_grid_gpu = cp.arange(
                new_b_grid_gpu[idx_01], new_b_grid_gpu[idx_99], new_res
            )

            # Create New Grid
            if len(new_b_grid_gpu) + len(fine_b_grid_gpu) < 15000:
                print(
                    "New Res:",
                    new_res,
                    "| Idx_01:",
                    idx_01,
                    "|Idx_99:",
                    idx_99,
                    "| new_b_grid_gpu[idx_01]:",
                    new_b_grid_gpu[idx_01],
                    "| new_b_grid_gpu[idx_99]:",
                    new_b_grid_gpu[idx_99],
                    end=" ",
                )
                new_b_grid_gpu_1 = cp.unique(
                    cp.concatenate((new_b_grid_gpu, fine_b_grid_gpu))
                )

                new_posterior_gpu_1 = cp.interp(new_b_grid_gpu_1, new_b_grid_gpu, new_posterior_gpu)

                # new_posterior_gpu_1 = cp.full(
                #     new_b_grid_gpu_1.shape,
                #     (1 / 2 / B_unk_bound_longitudinal),
                #     dtype=new_posterior_gpu.dtype,
                # )  # type: ignore
                # old_indices = cp.searchsorted(new_b_grid_gpu_1, new_b_grid_gpu)
                # old_indices = cp.clip(old_indices, 0, len(new_b_grid_gpu) - 1)
                # new_posterior_gpu_1[old_indices] = new_posterior_gpu
                normalization = cp.trapz(new_posterior_gpu_1, new_b_grid_gpu_1)
                new_posterior_gpu = new_posterior_gpu_1
                new_b_grid_gpu = new_b_grid_gpu_1
                new_posterior_gpu = new_posterior_gpu / normalization
                del (
                    new_posterior_gpu_1,
                    fine_b_grid_gpu,
                    new_b_grid_gpu_1,
                    normalization,
                )
                cp.get_default_memory_pool().free_all_blocks()

                dB = new_b_grid_gpu[1:] - new_b_grid_gpu[:-1]  # type: ignore
                pdf_mass = cp.zeros_like(new_posterior_gpu)
                pdf_mass[:-1] = new_posterior_gpu[:-1] * dB
                pdf_mass[-1] = new_posterior_gpu[-1] * dB[-1]
                cdf = cp.cumsum(pdf_mass)
                cdf = cdf/cdf[-1]  # Ensure it ends exactly at 1.0

                # Finding region where majority of the probability is concentrated
                for i in range(5, 10):
                    a = 10 ** (-i)
                    idx_01 = int(cp.searchsorted(cdf, cp.array(a)))
                    idx_99 = int(cp.searchsorted(cdf, cp.array(1 - a)))
                    idx_01 = int(max(0, idx_01))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99))
                    if idx_99 - idx_01 > 0:
                        break
                else:
                    idx_01 = int(max(0, idx_01 - 50))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99 + 50))
                #    print("Width index is zero, cannot proceed with zooming.")
                width_idx = idx_99 - idx_01
                total_points = len(new_b_grid_gpu)
                del cdf
                cp.get_default_memory_pool().free_all_blocks()
                if new_res <= 0:
                    print("new_res:", cp.min(new_res))
                    raise ValueError("Resolution is less than zero")
                # debugging =- explicitly check the break conditions FIXME
                zoom_index += 1

            else:
                new_res = initial_resolution
                print("New Res:",new_res,"| Idx_01:", idx_01, "|Idx_99:", idx_99, "| new_b_grid_gpu[idx_01]:",new_b_grid_gpu[idx_01],"| new_b_grid_gpu[idx_99]:", new_b_grid_gpu[idx_99], end = " ")
                new_b_grid_gpu = cp.arange(
                    B_unk_init_range[0], B_unk_init_range[1], new_res
                )  # type: ignore
                # 5. Interpolate Posterior (Nearest Left Neighbor)
                # We find which index in OLD grid is just left of each NEW point
                # searchsorted(old, new, side='right') - 1 gives the left neighbor index
                # indices = cp.searchsorted(new_b_grid_gpu, new_b_grid_gpu_1, side='right') - 1
                # indices = cp.clip(indices, 0, len(new_b_grid_gpu) - 1)
                # new_posterior_gpu_1 = new_posterior_gpu[indices]
                # new_posterior_gpu_1 = cp.interp(new_b_grid_gpu_1, new_b_grid_gpu, new_posterior_gpu)
                new_posterior_gpu = cp.full(
                    new_b_grid_gpu.shape,
                    (1 / 2 / B_unk_bound_longitudinal),
                    dtype=new_posterior_gpu.dtype,
                )

                normalization = cp.trapz(new_posterior_gpu, new_b_grid_gpu)
                new_posterior_gpu = new_posterior_gpu / normalization
                del normalization
                cp.get_default_memory_pool().free_all_blocks()

                dB = new_b_grid_gpu[1:] - new_b_grid_gpu[:-1]
                pdf_mass = cp.zeros_like(new_posterior_gpu)
                pdf_mass[:-1] = new_posterior_gpu[:-1] * dB
                pdf_mass[-1] = new_posterior_gpu[-1] * dB[-1]
                cdf = cp.cumsum(pdf_mass)
                cdf = cdf/cdf[-1]  # Ensure it ends exactly at 1.0

                if cp.min(dB) <= 0:
                    print(cp.min(dB))
                    raise ValueError("differential less than/equal to zero")

                # Finding region where majority of the probability is concentrated
                for i in range(5, 10):
                    a = 10 ** (-i)
                    idx_01 = int(cp.searchsorted(cdf, cp.array(a)))
                    idx_99 = int(cp.searchsorted(cdf, cp.array(1 - a)))
                    idx_01 = int(max(0, idx_01))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99))
                    if idx_99 - idx_01 > 0:
                        break
                else:
                    idx_01 = int(max(0, idx_01 - 50))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99 + 50))
                    print("Width index is zero, cannot proceed with zooming.")
                #     raise ValueError("Width index is zero, cannot proceed with zooming.")
                # debugging =- explicitly check the break conditions FIXME
                width_idx = idx_99 - idx_01
                del cdf
                cp.get_default_memory_pool().free_all_blocks()
                if new_res <= 0:
                    print("new_res:", cp.min(new_res))
                    raise ValueError("Resolution is less than zero")
                zoom_index += 1

        print(
            f"  [ZOOM] Res: {current_res:.6f} -> {new_res:.6f} | Grid Size: {len(new_b_grid_gpu)}|",
            end=" ",
        )
        print(time.time() - to, "was the time taken to increase resolution.", end=" ")
        return new_posterior_gpu, new_b_grid_gpu, new_res

    # No Zoom
    return new_posterior_gpu, new_b_grid_gpu, current_res


## Model definition
def calculate_summary_stats(posterior, B_grid):
    # Mean
    mean = cp.trapz(B_grid * posterior, x=B_grid)

    # Mode
    mode_index = cp.argmax(posterior)
    mode = B_grid[mode_index]

    # Variance
    variance = cp.trapz(((B_grid - mean) ** 2) * posterior, x=B_grid)

    return float(mode), float(variance**0.5)


def run_experiment_longitudinal_estimation(
    config: DataContext,
    params: ParameterContext,
    fixed_by_estimate: float = 0.0,
):
    print("Starting Adaptive Experiment...")
    time_cursor = params.curr_time

    t_sim, f_sim, by_sim, sim_spline = get_final_interpolator(config, params)
    t_exp, f_bias_axis, exp_matrix = load_experiment(config, config.Aligned)
    t_exp, f_bias_axis, exp_matrix = cp.asarray(t_exp), cp.asarray(f_bias_axis), cp.asarray(exp_matrix)
    f_bias_axis = f_bias_axis + params.f_bias_offset_nuisance

    b_grid_gpu = cp.arange(-params.B_unk_bound_longitudinal, params.B_unk_bound_longitudinal, params.init_resolution_longitudinal)  # type: ignore
    curr_res = params.init_resolution_longitudinal
    pdf_val = 1.0 / (2 * params.B_unk_bound_longitudinal)
    posterior_gpu = cp.full_like(b_grid_gpu, pdf_val)

    curr_bias = 0.0

    history = {
        "time": [],
        "est": [],
        "stddev": [],
        "bias": [],
        "res": [],
        "posteriors": [],
        "bgrids": [],
        "expectedkl": [],
    }
    history["posteriors"].append(posterior_gpu.get())
    history["bgrids"].append(b_grid_gpu.get())

    start_wall = time.time()

    while time_cursor < params.max_time:
        t_next = time_cursor + params.t_step
        t_start_abs = time_cursor
        t_end_abs = t_next

        idx_start = cp.searchsorted(t_exp, cp.asarray(t_start_abs))
        idx_end   = cp.searchsorted(t_exp, cp.asarray(t_end_abs))

        bias_idx = (cp.abs(f_bias_axis - curr_bias)).argmin()
        curr_bias = f_bias_axis[bias_idx]

        if idx_start >= idx_end:
            break

        y_obs = exp_matrix[idx_start:idx_end, bias_idx]
        t_chunk_exp = t_exp[idx_start:idx_end]
        t_sim_eval = t_chunk_exp

        if params.print_plot:
            plt.plot(b_grid_gpu.get(),posterior_gpu.get(), marker = '.')
            plt.title(f"Prior before likelihood update for curr_time {time_cursor}")
            plt.show()

        likelihood = calculate_likelihood_gpu(
            y_obs,
            t_sim_eval,
            curr_bias,
            f_bias_axis,
            fixed_by_estimate,
            b_grid_gpu,
            sim_spline,
            sigma_noise=params.sigma_noise_longitudinal,
        )

        posterior_gpu_1 = cp.log(posterior_gpu) + likelihood
        posterior_gpu_1 = posterior_gpu_1 - cp.max(posterior_gpu_1)
        posterior_gpu = cp.exp(posterior_gpu_1)
        norm = cp.trapz(posterior_gpu, x = b_grid_gpu)
        posterior_gpu = posterior_gpu / norm


        if params.print_plot:
            plt.plot(b_grid_gpu.get(),posterior_gpu.get(), marker = '.')
            plt.title(f"Posterior after likelihood update for curr_time {time_cursor}")
            plt.show()

        posterior_gpu, b_grid_gpu, curr_res = check_and_apply_zoom(
            posterior_gpu,
            b_grid_gpu,
            curr_res,
            zoom_factor=params.zoom_factor,
            zoom_trigger_multiple=params.zoom_trigger_multiple,
            zoom_trigger_ratio=params.zoom_trigger_ratio,
            initial_resolution=params.init_resolution_longitudinal,
            B_unk_bound_longitudinal=params.B_unk_bound_longitudinal,
        )

        mode, stddev = calculate_summary_stats(posterior_gpu, b_grid_gpu)

        history['time'].append(np.float64(t_next))
        history['est'].append(np.float64(mode))
        history['stddev'].append(np.float64(stddev))
        history['bias'].append(np.float64(curr_bias))
        history['res'].append(np.float64(curr_res))
        history['posteriors'].append(posterior_gpu.get())
        history['bgrids'].append(b_grid_gpu.get())

        print(
            "\n",
            f"T={t_next:.1f}s | Bias={curr_bias:.3f} | Est={mode:.5f} | StdDev={stddev:.1e} | Res={curr_res:.1e}",
            end=" ",
        )

        t_fut_1 = cp.float64(t_next)
        t_fut_2 = cp.float64(t_next + params.t_step)

        f_index_1 = cp.where(f_bias_axis > f_sim[10])[0][0]
        f_index_2 = cp.where(f_bias_axis > f_sim[-10])[0][0]
        next_bias, expectedkl = calculate_kl_divergence_gpu(
            posterior_gpu,
            b_grid_gpu,
            fixed_by_estimate,
            sim_spline,
            t_fut_1,
            t_fut_2,
            f_bias_axis[f_index_1:f_index_2],
            y_grid_size=params.kl_y_grid_size,
            sigma_noise=params.sigma_noise_longitudinal,
            batch_size=131,
        )

        history["expectedkl"].append(expectedkl.get())
        curr_bias = min(max(next_bias, -1.5), 1.5)
        time_cursor = t_next

    history_longitudinal_estimation = {
        "time": np.array(history["time"]),
        "est": np.array(history["est"]),
        "stddev": np.array(history["stddev"]),
        "bias": np.array(history["bias"]),
        "res": np.array(history["res"]),
        "posteriors": np.array(history["posteriors"], dtype=object),
        "bgrids": np.array(history["bgrids"], dtype=object),
        "expectedkl": np.array(history["expectedkl"], dtype=object),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(
        f"{config.save_path}/longitudinal_estimation_results_{params.Test}_{ts}.npz",
        **history_longitudinal_estimation,
    )
    print(f"Results for Test longitudinal_estimation_results_{params.Test}_{ts} saved successfully.", end=" ")

    print("\n", f"Done in {time.time() - start_wall:.2f}s")
    return history


def check_and_apply_zoom_by(
    posterior_gpu,
    b_grid_gpu,
    current_res,
    zoom_factor=1.5,
    zoom_trigger_multiple=400,
    zoom_trigger_ratio=0.2,
    b_y_bounds=(DEFAULT_PARAMS.B_unk_bound_transverse_lower, DEFAULT_PARAMS.B_unk_bound_transverse_upper),
    initial_resolution=0.00025,
):
    to = time.time()
    # PDF -> CDF
    dB = b_grid_gpu[1:] - b_grid_gpu[:-1]
    pdf_mass = cp.zeros_like(posterior_gpu)
    pdf_mass[:-1] = posterior_gpu[:-1] * dB
    pdf_mass[-1] = posterior_gpu[-1] * dB[-1]
    cdf = cp.cumsum(pdf_mass)
    cdf = cdf/cdf[-1]  # Ensure it ends exactly at 1.0

    #need a reset posterior condition incase cdf is zero everywhere
    if cdf[-1] == 0 or cdf[-1] < 1e-10:
        print("Warning: CDF is zero everywhere. Resetting posterior to uniform distribution.")
        posterior_gpu = cp.full_like(posterior_gpu, (1 / (b_y_bounds[1] - b_y_bounds[0])), dtype=posterior_gpu.dtype)
        dB = b_grid_gpu[1:] - b_grid_gpu[:-1]
        pdf_mass = cp.zeros_like(posterior_gpu)
        pdf_mass[:-1] = posterior_gpu[:-1] * dB
        pdf_mass[-1] = posterior_gpu[-1] * dB[-1]
        cdf = cp.cumsum(pdf_mass)
        cdf = cdf/ cdf[-1] # Ensure it ends exactly at 1.0

    for i in range(5, 10):
        a = 10 ** (-i)
        idx_01 = int(cp.searchsorted(cdf, cp.array(a)))
        idx_99 = int(cp.searchsorted(cdf, cp.array(1 - a)))
        idx_01 = int(max(0, idx_01))
        idx_99 = int(min(len(b_grid_gpu) - 1, idx_99))
        if idx_99 - idx_01 != 0:
            break
    else:
        idx_01 = int(max(0, idx_01 - 50))
        idx_99 = int(min(len(b_grid_gpu) - 1, idx_99 + 50))
    width_idx = idx_99 - idx_01
    total_points = len(b_grid_gpu)

    del cdf
    cp.get_default_memory_pool().free_all_blocks()

    # Check Trigger
    # if idx_99 - idx_01 < (ZOOM_TRIGGER_RATIO * total_points):
    new_res = current_res
    new_b_grid_gpu = cp.copy(b_grid_gpu)
    new_posterior_gpu = cp.copy(posterior_gpu)
    del b_grid_gpu, posterior_gpu
    cp.get_default_memory_pool().free_all_blocks()

    if (
        (
            (new_res * zoom_trigger_multiple)
            > (new_b_grid_gpu[idx_99] - new_b_grid_gpu[idx_01])
        )
        or width_idx < 100
        or len(new_b_grid_gpu) >= 15000
    ):
        zoom_index = 0
        while (
            (
                (new_res * zoom_trigger_multiple)
                > (new_b_grid_gpu[idx_99] - new_b_grid_gpu[idx_01])
            )
            or width_idx < 100
            or len(new_b_grid_gpu) >= 15000
        ) and zoom_index < 5:
            print(
                f"  [ZOOM] Triggered! Mass concentrated in {width_idx} points.|",
                end=" ",
            )
            new_res = np.abs(new_res) / zoom_factor
            fine_b_grid_gpu = cp.arange(
                new_b_grid_gpu[idx_01], new_b_grid_gpu[idx_99], new_res
            )
            if len(new_b_grid_gpu) + len(fine_b_grid_gpu) < 15000:
                print("New Res:",new_res,"| Idx_01:", idx_01, "|Idx_99:", idx_99, "| new_b_grid_gpu[idx_01]:",new_b_grid_gpu[idx_01],"| new_b_grid_gpu[idx_99]:", new_b_grid_gpu[idx_99], end = " ")
                new_b_grid_gpu_1 = cp.unique(
                    cp.concatenate((new_b_grid_gpu, fine_b_grid_gpu))
                )
                new_posterior_gpu_1 = cp.interp(new_b_grid_gpu_1, new_b_grid_gpu, new_posterior_gpu)

                # new_posterior_gpu_1 = cp.full(
                #     new_b_grid_gpu_1.shape,
                #     (1 / (b_y_bounds[1] - b_y_bounds[0])),
                #     dtype=new_b_grid_gpu_1.dtype,
                # )  # type: ignore

                # old_indices = cp.searchsorted(new_b_grid_gpu_1, new_b_grid_gpu)
                # old_indices = cp.clip(old_indices, 0, len(new_b_grid_gpu) - 1)
                # new_posterior_gpu_1[old_indices] = new_posterior_gpu
                normalization = cp.trapz(new_posterior_gpu_1, new_b_grid_gpu_1)
                new_posterior_gpu = new_posterior_gpu_1
                new_b_grid_gpu = new_b_grid_gpu_1
                new_posterior_gpu = new_posterior_gpu / normalization
                del (
                    new_posterior_gpu_1,
                    fine_b_grid_gpu,
                    new_b_grid_gpu_1,
                    normalization,
                )
                cp.get_default_memory_pool().free_all_blocks()

                dB = new_b_grid_gpu[1:] - new_b_grid_gpu[:-1]  # type: ignore
                pdf_mass = cp.zeros_like(new_posterior_gpu)
                pdf_mass[:-1] = new_posterior_gpu[:-1] * dB
                pdf_mass[-1] = new_posterior_gpu[-1] * dB[-1]
                cdf = cp.cumsum(pdf_mass)
                cdf = cdf/cdf[-1]  # Ensure it ends exactly at 1.0

                for i in range(5, 10):
                    a = 10 ** (-i)
                    idx_01 = int(cp.searchsorted(cdf, cp.array(a)))
                    idx_99 = int(cp.searchsorted(cdf, cp.array(1 - a)))
                    idx_01 = int(max(0, idx_01))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99))
                    if idx_99 - idx_01 > 0:
                        break
                else:
                    idx_01 = int(max(0, idx_01 - 50))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99 + 50))
                width_idx = idx_99 - idx_01
                total_points = len(new_b_grid_gpu)
                del cdf
                cp.get_default_memory_pool().free_all_blocks()
                if new_res <= 0:
                    print("new_res:", cp.min(new_res))
                    raise ValueError("Resolution is less than zero")
                zoom_index+=1
            else:
                new_res = initial_resolution
                print("New Res:",new_res,"| Idx_01:", idx_01, "|Idx_99:", idx_99, "| new_b_grid_gpu[idx_01]:",new_b_grid_gpu[idx_01],"| new_b_grid_gpu[idx_99]:", new_b_grid_gpu[idx_99], end = " ")
                new_b_grid_gpu = cp.arange(b_y_bounds[0], b_y_bounds[1], new_res)  # type: ignore
                # We find which index in OLD grid is just left of each NEW point
                # searchsorted(old, new, side='right') - 1 gives the left neighbor index
                # indices = cp.searchsorted(new_b_grid_gpu, new_b_grid_gpu_1, side='right') - 1
                # indices = cp.clip(indices, 0, len(new_b_grid_gpu) - 1)
                # new_posterior_gpu_1 = new_posterior_gpu[indices]
                # new_posterior_gpu_1 = cp.interp(new_b_grid_gpu_1, new_b_grid_gpu, new_posterior_gpu)
                new_posterior_gpu = cp.full(
                    new_b_grid_gpu.shape,
                    (1 / (b_y_bounds[1] - b_y_bounds[0])),
                    dtype=new_posterior_gpu.dtype,
                )

                normalization = cp.trapz(new_posterior_gpu, new_b_grid_gpu)
                new_posterior_gpu = new_posterior_gpu / normalization
                del normalization
                cp.get_default_memory_pool().free_all_blocks()

                dB = new_b_grid_gpu[1:] - new_b_grid_gpu[:-1]
                pdf_mass = cp.zeros_like(new_posterior_gpu)
                pdf_mass[:-1] = new_posterior_gpu[:-1] * dB
                pdf_mass[-1] = new_posterior_gpu[-1] * dB[-1]
                cdf = cp.cumsum(pdf_mass)
                cdf = cdf/cdf[-1]  # Ensure it ends exactly at 1.0

                for i in range(5, 10):
                    a = 10 ** (-i)
                    idx_01 = int(cp.searchsorted(cdf, cp.array(a)))
                    idx_99 = int(cp.searchsorted(cdf, cp.array(1 - a)))
                    idx_01 = int(max(0, idx_01))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99))
                    if idx_99 - idx_01 > 0:
                        break
                else:
                    idx_01 = int(max(0, idx_01 - 50))
                    idx_99 = int(min(len(new_b_grid_gpu) - 1, idx_99 + 50))
                    print("Width index is zero, cannot proceed with zooming.")
                width_idx = idx_99 - idx_01
                del cdf
                cp.get_default_memory_pool().free_all_blocks()
                if new_res <= 0:
                    print("new_res:", cp.min(new_res))
                    raise ValueError("Resolution is less than zero")
                zoom_index+=1

        print(
            f"  [ZOOM] Res: {current_res} -> {new_res} | Grid Size: {len(new_b_grid_gpu)}|",
            end=" ",
        )
        print(time.time() - to, "was the time taken to increase resolution.", end=" ")
        return new_posterior_gpu, new_b_grid_gpu, new_res
    # No Zoom
    return new_posterior_gpu, new_b_grid_gpu, current_res


def run_experiment_y_estimation(
    config: DataContext,
    params: ParameterContext,
    fixed_bz_estimate: float = 0.0,
):

    print("...Starting Adaptive Experiment for Y Estimation...")
    time_cursor = params.curr_time

    t_exp, f_bias_axis, exp_matrix = load_experiment(config, config.Aligned)
    t_exp, f_bias_axis, exp_matrix = cp.asarray(t_exp), cp.asarray(f_bias_axis), cp.asarray(exp_matrix)
    t_sim, f_sim, by_sim, sim_interp = get_final_interpolator(config, params)

    b_grid_gpu = cp.arange(params.B_unk_bound_transverse_lower, params.B_unk_bound_transverse_upper, params.init_resolution_transverse)  # type: ignore
    curr_res = params.init_resolution_transverse

    posterior_gpu = cp.ones_like(b_grid_gpu)
    posterior_gpu /= cp.sum(posterior_gpu) * curr_res

    curr_bias_z = 0.0

    history = {
        "time": [],
        "est": [],
        "stddev": [],
        "bias": [],
        "res": [],
        "posteriors": [],
        "bgrids": [],
        "expectedkl": [],
    }
    history["posteriors"].append(posterior_gpu.get())
    history["bgrids"].append(b_grid_gpu.get())

    start_wall = time.time()

    while time_cursor < params.max_time:
        t_next = time_cursor + params.t_step
        t_abs_start = time_cursor
        t_abs_end = t_next
        idx_start = cp.searchsorted(t_exp, cp.asarray(t_abs_start))
        idx_end = cp.searchsorted(t_exp, cp.asarray(t_abs_end))
        bias_idx = (cp.abs(f_bias_axis - curr_bias_z)).argmin()
        curr_bias_z = f_bias_axis[bias_idx]

        if idx_start >= idx_end:
            break
        y_obs = exp_matrix[idx_start:idx_end, bias_idx]
        t_chunk_exp = t_exp[idx_start:idx_end]
        t_chunk_sim = t_chunk_exp

        if params.print_plot:
            plt.plot(b_grid_gpu.get(),posterior_gpu.get(), marker = '.')
            plt.title(f"Prior before likelihood update for curr_time {time_cursor}")
            plt.show()

        likelihood = calculate_likelihood_by(
            y_obs,
            t_chunk_sim,
            curr_bias_z,
            f_bias_axis,
            b_grid_gpu,
            sim_interp,
            sigma_noise=params.sigma_noise_transverse,
            fixed_bz_estimate=fixed_bz_estimate,
        )

        if params.print_plot:
            plt.plot(b_grid_gpu.get(), posterior_gpu.get(), marker = '.')  # type: ignore
            plt.title(f"Posterior after likelihood update for curr_time {time_cursor}")
            plt.show()

        posterior_gpu = cp.log(posterior_gpu) + likelihood
        posterior_gpu = posterior_gpu - cp.max(posterior_gpu)
        posterior_gpu = cp.exp(posterior_gpu)
        norm = cp.trapz(posterior_gpu, b_grid_gpu)
        posterior_gpu /= norm

        posterior_gpu, b_grid_gpu, curr_res = check_and_apply_zoom_by(
            posterior_gpu,
            b_grid_gpu,
            curr_res,
            zoom_factor=params.zoom_factor,
            zoom_trigger_multiple=params.zoom_trigger_multiple,
            zoom_trigger_ratio=params.zoom_trigger_ratio,
            b_y_bounds=(params.B_unk_bound_transverse_lower, params.B_unk_bound_transverse_upper),
            initial_resolution=params.init_resolution_transverse,
        )

        mode, stddev = calculate_summary_stats(posterior_gpu, b_grid_gpu)

        history['time'].append(np.float64(t_next))
        history['est'].append(np.float64(mode))
        history['stddev'].append(np.float64(stddev))
        history['bias'].append(np.float64(curr_bias_z))
        history['res'].append(np.float64(curr_res))
        history['posteriors'].append(posterior_gpu.get())
        history['bgrids'].append(b_grid_gpu.get())

        print("\n",
            f"T={t_next:.1f} | BiasZ={curr_bias_z:.3f} | Est By={mode:.5f} | Std={stddev:.4f}| Res={curr_res:.1e}", end = ""
        )

        t_fut_1 = np.float64(t_next)
        t_fut_2 = np.float64(t_next + params.t_step)

        f_index_1 = cp.where(f_bias_axis > f_sim[10])[0][0]
        f_index_2 = cp.where(f_bias_axis > f_sim[-10])[0][0]
        next_bias, expectedkl = calculate_kl_by(
            posterior_gpu, b_grid_gpu, sim_interp, t_fut_1, t_fut_2, f_bias_axis[f_index_1:f_index_2], fixed_bz_estimate=fixed_bz_estimate, sigma_noise=params.sigma_noise_transverse, batch_size=131, y_grid_size=params.kl_y_grid_size
        )

        history["expectedkl"].append(expectedkl.get())
        curr_bias_z = min(max(next_bias, -1.5), 1.5)
        time_cursor = t_next

    history_y_estimation = {
        "time": np.array(history["time"]),
        "est": np.array(history["est"]),
        "stddev": np.array(history["stddev"]),
        "bias": np.array(history["bias"]),
        "res": np.array(history["res"]),
        "posteriors": np.array(history["posteriors"], dtype=object),
        "bgrids": np.array(history["bgrids"], dtype=object),
        "expectedkl": np.array(history["expectedkl"], dtype=object),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(
        f"{config.save_path}/transverse_field_estimation_results_{params.Test}_{ts}.npz",
        **history_y_estimation,
    )
    print(f"Results for Transverse Field, Test_{params.Test}_{ts} saved successfully.", end=" ")

    print("\n", f"Done in {time.time() - start_wall:.2f}s")
    return history


def altopt(
    config: DataContext, 
    params: ParameterContext,
):
    # ! load these arguments using some parameter context. 
    num_iter = params.num_iter
    max_time = params.max_time
    curr_time = params.curr_time
    t_step = params.t_step
    tol_bz = params.tol_bz
    tol_by = params.tol_by
    patience = params.patience
    Est_First_Z = params.Est_First_Z
    fixed_bz_estimate = params.fixed_bz_estimate
    fixed_by_estimate = params.fixed_by_estimate
    Test = params.Test
    if Est_First_Z:
        bz_history = []
        by_history = [fixed_by_estimate]
        stable_bz_runs = 0
        stable_by_runs = 0

        # Initial Bz estimation using current By seed
        current_by = fixed_by_estimate
        by_history.append(current_by)
        results = run_experiment_longitudinal_estimation(
            config,
            params,
            fixed_by_estimate=current_by,
        )
        current_bz = results["est"][-1]
        bz_history.append(current_bz)
        print(f"Estimated Bz: {current_bz}")

        for i in range(num_iter):
            # Estimate By given current Bz
            results_by = run_experiment_y_estimation(
                config,
                params,
                fixed_bz_estimate=current_bz,
            )
            new_by = results_by["est"][-1]

            # UPDATE THE GLOBAL BY ESTIMATE
            current_by = new_by

            # Early-stop tracking for By
            if by_history and abs(new_by - by_history[-1]) <= tol_by:
                stable_by_runs += 1
            else:
                stable_by_runs = 0
            by_history.append(new_by)
            print(f"Estimated By: {new_by} (stable {stable_by_runs}/{patience})")

            # Re-estimate Bz given new By
            results = run_experiment_longitudinal_estimation(
                config,
                params,
                fixed_by_estimate=current_by,
            )
            new_bz = results["est"][-1]

            # UPDATE THE GLOBAL BZ ESTIMATE
            current_bz = new_bz

            # Early-stop tracking for Bz
            if abs(new_bz - bz_history[-1]) <= tol_bz:
                stable_bz_runs += 1
            else:
                stable_bz_runs = 0
            bz_history.append(new_bz)
            print(f"Estimated Bz: {new_bz} (stable {stable_bz_runs}/{patience})")

            # Check combined early-stop condition
            if (stable_bz_runs >= patience) and (stable_by_runs >= patience):
                print(
                    f"Early stopping at iteration {i + 1}: Bz and By stable for >= {patience} steps (tol_bz={tol_bz}, tol_by={tol_by})."
                )
                break

        return bz_history, by_history

    else:
        bz_history = [fixed_bz_estimate]
        by_history = []
        stable_bz_runs = 0
        stable_by_runs = 0

        # Initial By estimation using current Bz seed
        current_bz = fixed_bz_estimate
        results = run_experiment_y_estimation(
            config,
            params,
            fixed_bz_estimate=current_bz,
        )
        current_by = results["est"][-1]
        by_history.append(current_by)
        print(f"Estimated By: {current_by}")

        for i in range(num_iter):
            # Estimate By given current Bz
            results_bz = run_experiment_longitudinal_estimation(
                config,
                params,
                fixed_by_estimate=current_by,
            )
            new_bz = results_bz["est"][-1]

            # UPDATE THE GLOBAL BY ESTIMATE
            current_bz = new_bz

            # Early-stop tracking for By
            if bz_history and abs(new_bz - bz_history[-1]) <= tol_bz:
                stable_bz_runs += 1
            else:
                stable_bz_runs = 0
            bz_history.append(new_bz)
            print(f"Estimated By: {new_bz} (stable {stable_bz_runs}/{patience})")

            # Re-estimate Bz given new By
            results = run_experiment_y_estimation(
                config,
                params,
                fixed_bz_estimate=current_bz,
            )
            new_by = results["est"][-1]

            # UPDATE THE GLOBAL BZ ESTIMATE
            current_by = new_by

            # Early-stop tracking for Bz
            if abs(new_by - by_history[-1]) <= tol_by:
                stable_by_runs += 1
            else:
                stable_by_runs = 0
            by_history.append(new_by)
            print(f"Estimated By: {new_by} (stable {stable_by_runs}/{patience})")

            # Check combined early-stop condition
            if (stable_bz_runs >= patience) and (stable_by_runs >= patience):
                print(
                    f"Early stopping at iteration {i + 1}: Bz and By stable for >= {patience} steps (tol_bz={tol_bz}, tol_by={tol_by})."
                )
                break

        return bz_history, by_history


def plot_altopt(config, bz_history, by_history, trajectory_mode=True):
    iterations_bz = list(range(len(bz_history)))
    iterations_by = list(range(len(by_history)))

    plt.figure(figsize=(15, 5))

    # Plot for Bz
    plt.subplot(1, 3, 1)
    plt.plot(iterations_bz, bz_history, marker="o")
    plt.title("Convergence of Bz Estimates")
    plt.xlabel("Iteration")
    plt.ylabel("Bz")
    plt.grid(True, alpha=0.3)

    # Plot for By
    plt.subplot(1, 3, 2)
    plt.plot(iterations_by, by_history, marker="o")
    plt.title("Convergence of By Estimates")
    plt.xlabel("Iteration")
    plt.ylabel("By")
    plt.grid(True, alpha=0.3)

    # Trajectory Plot
    if trajectory_mode:
        plt.subplot(1, 3, 3)

        plt.plot(by_history, bz_history, marker="o")
        plt.title("Trajectory of Estimates in By-Bz Space")
        plt.xlabel("By Estimates")
        plt.ylabel("Bz Estimates")
        plt.grid(True, alpha=0.3)
        for idx, (by, bz) in enumerate(zip(by_history, bz_history), start=0):
            plt.annotate(
                str(idx),
                (by, bz),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    plt.suptitle("AltOpt Convergence and Trajectory", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{config.save_path}/AltOpt_Convergence_Trajectory.png")
    plt.show()

def test_altopt_sweep(
    config, 
    t_step_values=[0.1, 0.2, 0.5, 1.0],
    fixed_by_init_values=[0.0, 0.05, 0.10, 0.12],
    max_time=30,
    num_iter=3,
    tol_bz=1e-4,
    tol_by=1e-4,
    patience=3,
    *,
    params: ParameterContext,
):
    """
    Run altopt for different combinations of T_STEP and initial By estimate.
    Returns results dict for plotting.
    """
    results = []

    for t_step in t_step_values:
        for by_init in fixed_by_init_values:
            print(f"\n{'=' * 60}")
            print(f"Testing: T_STEP={t_step}, Initial By={by_init}")
            print(f"{'=' * 60}")
            try:
                params_run = replace(
                    params,
                    t_step=t_step,
                    fixed_by_estimate=by_init,
                    max_time=max_time,
                    num_iter=num_iter,
                    tol_bz=tol_bz,
                    tol_by=tol_by,
                    patience=patience,
                )
                bz_hist, by_hist = altopt(
                    config, 
                    params_run,
                )  # type: ignore
                results.append(
                    {
                        "t_step": t_step,
                        "by_init": by_init,
                        "bz_history": bz_hist,
                        "by_history": by_hist,
                        "label": f"T={t_step}, By₀={by_init}",
                    }
                )

            except Exception as e:
                print(f"Error with T_STEP={t_step}, By_init={by_init}: {e}")
    return results


def plot_altopt_first3_combined(results, config):
    """
    Produce a single combined 3x3 figure:

        Row 1: Bz convergence curves (one per t_step column)
        Row 2: By convergence curves (one per t_step column)
        Row 3: Bz–By trajectory plots (one per t_step column)

    Columns correspond to different t_step values.

    Saves output to Tests/Actual/CW/AltOpt by default.
    """

    os.makedirs(config.save_path, exist_ok=True)

    # Extract unique t_step values
    t_steps = sorted(set(r["t_step"] for r in results))
    n = len(t_steps)

    # Create 3xN grid
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 14))
    fig.suptitle("AltOpt Comparison Across T_STEP Values", fontsize=20)

    # If only 1 t_step, axes will not be 2D; normalize
    if n == 1:
        axes = np.array([axes]).reshape(3, 1)

    for col, t_step in enumerate(t_steps):
        subset = [r for r in results if r["t_step"] == t_step]
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(subset)))

        # Unpack row axes
        ax_Bz = axes[0, col]
        ax_By = axes[1, col]
        ax_traj = axes[2, col]

        # Column title
        ax_Bz.set_title(f"T_STEP = {t_step}", fontsize=16)

        # 1. Bz convergence
        for i, res in enumerate(subset):
            it = range(len(res["bz_history"]))
            ax_Bz.plot(
                it,
                res["bz_history"],
                marker="o",
                color=colors[i],
                alpha=0.8,
                label=res["label"],
            )
        ax_Bz.set_xlabel("Iteration")
        ax_Bz.set_ylabel("Bz")
        ax_Bz.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_Bz.grid(True, alpha=0.3)
        ax_Bz.legend(fontsize=7)

        # 2. By convergence
        for i, res in enumerate(subset):
            it = range(len(res["by_history"]))
            ax_By.plot(
                it,
                res["by_history"],
                marker="s",
                color=colors[i],
                alpha=0.8,
                label=res["label"],
            )
        ax_By.set_xlabel("Iteration")
        ax_By.set_ylabel("By")
        ax_By.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_By.grid(True, alpha=0.3)
        ax_By.legend(fontsize=7)

        # 3. Bz–By trajectory
        for i, res in enumerate(subset):
            by_traj = [res["by_init"]] + res["by_history"]
            bz_traj = res["bz_history"]
            ax_traj.plot(
                bz_traj,
                by_traj,
                marker="o",
                markersize=6,
                linewidth=2,
                color=colors[i],
                alpha=0.8,
                label=res["label"],
            )
            ax_traj.plot(
                bz_traj[0],
                by_traj[0],
                "x",
                color=colors[i],
                markersize=10,
                markeredgewidth=2,
            )
        ax_traj.set_xlabel("Bz")
        ax_traj.set_ylabel("By")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend(fontsize=7)

    plt.tight_layout()

    # Save
    outpath = os.path.join(config.save_path, "altopt_first3_combined.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print("Saved:", outpath)

    plt.show()


if __name__ == "__main__":
    pass
