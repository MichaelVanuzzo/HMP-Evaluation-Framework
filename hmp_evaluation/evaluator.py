"""
High-level evaluation wrapper implementing the HMP Evaluation Framework protocol
on top of `hmp_utils`.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Pre-computed cache of PSDS (Wasserstein distance) values between dataset splits,
# obtained by running `SpectrumMetricsProcessor.wasserstein_distance(psd[split_a], psd[split_b])`
# on the reference PSD files (`*.npy`) bundled in `hmp_utils/pds_files/`.
# See `get_reference_psds()` below for the dynamic computation from `hmp_utils`.
PSDS_TRAIN_TEST_REF: Dict[str, float] = {
    "h36m_25fps": 0.21093169828233155,
    "amass_25fps": 0.02207852818256258,
    "amass_30fps": 0.02967092962258535,
    "amass_60fps": 0.02643893393737065,
    "amass2h36m_25fps": 0.02238238663029078,
}

PSDS_TRAIN_VALID_REF: Dict[str, float] = {
    "h36m_25fps": 0.18438128533297415,
    "amass_25fps": 0.01614558220990185,
    "amass_30fps": 0.02569505731241571,
    "amass_60fps": 0.02949968719698807,
    "amass2h36m_25fps": 0.02006714514640240,
}

PSDS_TEST_VALID_REF: Dict[str, float] = {
    "h36m_25fps": 0.16259248173014204,
    "amass_25fps": 0.020590669094298856,
    "amass_30fps": 0.02718313135124091,
    "amass_60fps": 0.02819242576850198,
    "amass2h36m_25fps": 0.02139417700475526,
}

COMPACT_TIMESTAMPS_25FPS: List[int] = [40, 120, 200, 400, 600, 800, 1000]
EXTENDED_TIMESTAMPS_25FPS: List[int] = [40, 80, 120, 160, 200, 320, 400, 600, 800, 1000]
COMPACT_TIMESTAMPS_60FPS: List[int] = [17, 100, 200, 400, 600, 800, 1000]

H36M_ACTIONS: List[str] = [
    "walking",
    "eating",
    "smoking",
    "discussion",
    "directions",
    "greeting",
    "phoning",
    "posing",
    "purchases",
    "sitting",
    "sittingdown",
    "takingphoto",
    "waiting",
    "walkingdog",
    "walkingtogether",
]


def get_reference_psds(
    dataset: str = "amass_25fps",
    split_a: str = "train",
    split_b: str = "test",
) -> float:
    """
    Compute the reference PSDS (Wasserstein distance) between two splits (`split_a` and `split_b`,
    by default `"train"` and `"test"`) directly from the dataset PSD `.npy` files bundled in
    `hmp_utils/pds_files/`. Falls back to the cached reference dictionary if `hmp_utils` is not
    installed in the current Python environment.

    Args:
        dataset: `"h36m_25fps"`, `"amass_25fps"`, `"amass_30fps"`, `"amass_60fps"`, or `"amass2h36m_25fps"`.
        split_a: First dataset split (`"train"`, `"test"`, or `"validation"`).
        split_b: Second dataset split (`"train"`, `"test"`, or `"validation"`).

    Returns:
        Reference PSDS (float).
    """
    ds_map = {
        "h36m_25fps": ("H36M", 25, False),
        "amass_25fps": ("SMPL", 25, False),
        "amass_30fps": ("SMPL", 30, False),
        "amass_60fps": ("SMPL", 60, False),
        "amass2h36m_25fps": ("H36M", 25, True),
    }
    if dataset not in ds_map:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(ds_map.keys())}.")

    try:
        from hmp_utils import BodyModelType, EvaluationProtocol, SpectrumMetricsProcessor

        model_name, fps, is_a2h = ds_map[dataset]
        body_type = BodyModelType.H36M if model_name == "H36M" else BodyModelType.SMPL
        psd_dict, _ = EvaluationProtocol.load_power_spectral_density_dict(
            body_model_type=body_type,
            framerate=fps,
            is_amass2h36m=is_a2h,
        )
        return float(SpectrumMetricsProcessor.wasserstein_distance(psd_dict[split_a], psd_dict[split_b]))
    except ImportError:
        pair = {split_a, split_b}
        if pair == {"train", "test"}:
            return PSDS_TRAIN_TEST_REF[dataset]
        if pair == {"train", "validation"}:
            return PSDS_TRAIN_VALID_REF[dataset]
        if pair == {"test", "validation"}:
            return PSDS_TEST_VALID_REF[dataset]
        raise


def compute_zero_velocity_baseline(last_input_pose: np.ndarray, target_seq_len: int) -> np.ndarray:
    """
    Generate Zero-Velocity (ZeroVel) baseline predictions by repeating the last
    observed input frame across `target_seq_len` future frames.

    Args:
        last_input_pose: Array of shape `(batch_size, num_joints, 3)` or `(batch_size, 1, num_joints, 3)`.
        target_seq_len: Number of future frames to predict.

    Returns:
        Zero-velocity prediction array of shape `(batch_size, target_seq_len, num_joints, 3)`.
    """
    pose = np.asarray(last_input_pose, dtype=float)
    if pose.ndim == 3:
        pose = np.expand_dims(pose, axis=1)
    if pose.ndim != 4 or pose.shape[1] != 1:
        raise ValueError("last_input_pose must have shape (batch, num_joints, 3) or (batch, 1, num_joints, 3)")
    return np.repeat(pose, repeats=target_seq_len, axis=1)


def compute_hmpe_score(
    mpjpe_t: float,
    jad_t: float,
    psds_1000: float,
    zero_vel_mpjpe_t: float,
    zero_vel_jad_t: float,
    psds_train_test: float,
) -> Tuple[float, Tuple[float, float, float]]:
    """
    Compute the Human Motion Prediction Error (HMPE_t) on the [0, 100] percentage
    scale and the three normalized radar axes (JAD_hat, MPJPE_hat, PSDS_hat) in [0, 1],
    according to Equations (8)-(9) of the paper.

    Args:
        mpjpe_t: Mean Per Joint Position Error at time `t` (in mm or m, matching `zero_vel_mpjpe_t`).
        jad_t: Joint Angle Difference at time `t` (in rad).
        psds_1000: Power Spectral Density Shift over the first 1000 ms window.
        zero_vel_mpjpe_t: Zero-Velocity baseline MPJPE at time `t`.
        zero_vel_jad_t: Zero-Velocity baseline JAD at time `t`.
        psds_train_test: Baseline PSDS between the training and test sets of the dataset.

    Returns:
        Tuple `(hmpe_t, (norm_jad, norm_mpjpe, norm_psds))` where `hmpe_t` is in `[0, 100]`
        (with `ZeroVel = 100.0`) and the normalized radar axes are in `[0, 1]`.
    """
    norm_jad = float(np.clip(jad_t / zero_vel_jad_t, 0.0, 1.0))
    norm_mpjpe = float(np.clip(mpjpe_t / zero_vel_mpjpe_t, 0.0, 1.0))
    norm_psds = float(np.clip(psds_1000 / (10.0 * psds_train_test), 0.0, 1.0))

    hmpe_t = 100.0 * (norm_jad * norm_mpjpe + norm_mpjpe * norm_psds + norm_psds * norm_jad) / 3.0
    return float(hmpe_t), (norm_jad, norm_mpjpe, norm_psds)


class HMPEvaluator:
    """
    High-level evaluator wrapping `hmp_utils` for the HMP Evaluation Framework.

    Handles skeleton normalization (scaling the left femur to 1 meter), instant
    per-frame MPJPE (in millimeters) and JAD (in radians on major joints),
    sliding-window PSDS, and CSV export compatible with all plotting and table
    utilities.
    """

    def __init__(
        self,
        dataset: str = "amass",
        framerate: int = 25,
        is_amass2h36m: bool = False,
        normalize_skeleton: bool = True,
    ) -> None:
        """
        Initialize the evaluator for a target dataset and framerate.

        Args:
            dataset: `"amass"` (SMPL skeleton) or `"h36m"` (Human3.6M skeleton).
            framerate: Sequence framerate in Hz (`25`, `30`, or `60`).
            is_amass2h36m: Set to `True` when evaluating on AMASS converted to the H3.6M skeleton.
            normalize_skeleton: If `True` (default, as in the paper), normalizes the reference
                left-femur joint (`JOINT_TO_NORMALIZE`) to 1 meter before computing forward kinematics.
        """
        from hmp_utils import (
            EvaluationProtocol,
            ForwardKinematics,
            H36MConstants,
            SmplConstants,
        )

        dataset_key = dataset.strip().lower()
        if dataset_key in ("h36m", "human3.6m", "amass2h36m"):
            self.body_constants = H36MConstants
            if dataset_key == "amass2h36m":
                is_amass2h36m = True
        elif dataset_key in ("amass", "smpl", "smplh", "smplx"):
            self.body_constants = SmplConstants
        else:
            raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'amass' or 'h36m'.")

        self.dataset = "amass2h36m" if is_amass2h36m else dataset_key
        self.framerate = int(framerate)
        self.is_amass2h36m = bool(is_amass2h36m)
        self.normalize_skeleton = bool(normalize_skeleton)

        self.fk_processor = ForwardKinematics(
            self.body_constants.OFFSETS,
            self.body_constants.PARENTS,
        )

        self.fk_processor.set_body_model_type(self.body_constants.BODY_MODEL_TYPE)
        if self.normalize_skeleton:
            self.fk_processor.normalize_joints(self.body_constants.JOINT_TO_NORMALIZE)

        self.metrics_dict, self.metric_processor_dict = EvaluationProtocol.initialize_metrics_dicts(self.fk_processor)
        self.metrics_res_by_action: Dict[str, dict] = {}

    def reset(self) -> None:
        """Clear accumulated per-action metric results."""
        self.metrics_res_by_action = {}

    def positions_to_angle_axis(
        self,
        joint_positions: Union[list, np.ndarray],
        joint_indices: Optional[Union[list, np.ndarray]] = None,
        ignore_root_joint: bool = True,
    ) -> np.ndarray:
        """
        Convert 3D joint positions (in meters) into local angle-axis rotations using
        `hmp_utils.motion.inverse_kinematics.InverseKinematics`.

        Args:
            joint_positions: 3D joint coordinates in meters with shape `(..., num_used_joints, 3)`.
            joint_indices: Indices of the joints present in `joint_positions` (e.g. `np.arange(4, 22)`
                for an 18-joint SMPL subset). If `None`, assumes all skeleton joints
                `0..num_keypoints-1` are provided.
            ignore_root_joint: If `True` (default), sets the root joint and its immediate
                children to the canonical rest offsets.

        Returns:
            Angle-axis rotations of shape `(..., num_keypoints, 3)` ready for `evaluate_action()`.
        """
        from hmp_utils.motion.inverse_kinematics import InverseKinematics

        pos_arr = np.asarray(joint_positions, dtype=float)
        n_total = len(self.body_constants.OFFSETS)
        if joint_indices is None:
            if pos_arr.shape[-2] != n_total:
                raise ValueError(f"joint_positions has {pos_arr.shape[-2]} joints, expected {n_total} or an explicit `joint_indices` list.")
            indices: Union[list, np.ndarray] = list(range(n_total))
        else:
            indices = joint_indices

        ik = InverseKinematics(
            offsets=self.body_constants.OFFSETS,
            parents=self.body_constants.PARENTS,
        )
        ik.set_body_model_type(self.body_constants.BODY_MODEL_TYPE)
        return np.asarray(
            ik.compute_ik(
                joint_pos=pos_arr,
                joint_indices=indices,
                ignore_root_joint=ignore_root_joint,
            ),
            dtype=float,
        )

    def evaluate_action(
        self,
        predictions: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
        action: str = "amass",
        target_seq_len: Optional[int] = None,
        zero_velocity: bool = False,
        default_error: Optional[Union[float, np.ndarray]] = None,
        set_of_psd_to_use: str = "train",
    ) -> dict:
        """
        Compute all accuracy (MPJPE, JAD, MAE XYZ/ZXY) and realism (PSDS) metrics
        for a given action or dataset split.

        Args:
            predictions: Predicted joint rotations in angle-axis representation,
                shape `(batch_size, seq_len, num_joints, 3)`.
            targets: Ground-truth joint rotations in angle-axis representation,
                shape `(batch_size, seq_len, num_joints, 3)`.
            action: Action label (e.g. `"amass"` for AMASS, or one of `H36M_ACTIONS` for H3.6M).
            target_seq_len: Number of frames for per-frame accuracy metrics (defaults to
                1 second, i.e., `framerate` frames, while the full `seq_len` is used for
                sliding-window PSDS if longer than 1 second).
            zero_velocity: Set to `True` when evaluating the Zero-Velocity baseline.
            default_error: Optional model-specific default error tensor of shape
                `(batch_size, seq_len)` (defaults to zeros if omitted).
            set_of_psd_to_use: Reference PSD split (`"train"`, `"test"`, or `"validation"`).

        Returns:
            Dictionary of computed metrics for `action`.
        """
        from hmp_utils import EvaluationProtocol

        preds_arr = np.asarray(predictions, dtype=float)
        targs_arr = np.asarray(targets, dtype=float)
        if target_seq_len is None:
            target_seq_len = min(preds_arr.shape[-3], self.framerate)

        if default_error is None or np.ndim(default_error) == 0:
            fill_val = 0.0 if default_error is None else float(default_error)
            default_err_arr = np.full(preds_arr.shape[:-2], fill_val, dtype=float)
        else:
            default_err_arr = np.asarray(default_error, dtype=float)

        action_norm = "amass" if action.strip().lower() == "amass" else action.strip()
        self.metrics_res_by_action[action_norm] = {}

        EvaluationProtocol.compute_all_metrics(
            predictions=preds_arr,
            targets=targs_arr,
            framerate=self.framerate,
            target_seq_len=target_seq_len,
            default_error=default_err_arr,
            bodyConstants=self.body_constants,
            fk_processor=self.fk_processor,
            metrics_res_by_action_dict=self.metrics_res_by_action,
            action=action_norm,
            metrics_dict=self.metrics_dict,
            metric_processor_dict=self.metric_processor_dict,
            set_of_psd_to_use=set_of_psd_to_use,
            zero_velocity=zero_velocity,
            use_global_rotations=False,
            is_amass2h36m=self.is_amass2h36m,
        )
        return self.metrics_res_by_action[action_norm]

    def compute_mpjpe(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute instant per-frame MPJPE (in millimeters) on the normalized 1-m left-femur
        skeleton over the major body joints (Equation 2 of the paper).

        Returns:
            1D numpy array of shape `(num_frames,)` with MPJPE in millimeters.
        """
        from hmp_utils import MetricsPostProcessor, MetricsProcessor

        joints = [i for i in self.body_constants.MAJOR_JOINTS if i != 0]
        m_proc = MetricsProcessor(predictions=predictions, targets=targets, joints_to_compute=joints)
        pos_err_m = m_proc.positional_error(self.fk_processor)
        post = MetricsPostProcessor(pos_err_m)
        num_frames = np.asarray(predictions).shape[-3]
        return np.array([post.instant(at_frame=t) * 1000.0 for t in range(num_frames)], dtype=float)

    def compute_jad(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute instant per-frame Joint Angle Difference (JAD, in radians) over the
        major body joints excluding the root joint (Equation 4 of the paper).

        Returns:
            1D numpy array of shape `(num_frames,)` with JAD in radians.
        """
        from hmp_utils import MetricsPostProcessor, MetricsProcessor

        joints = [i for i in self.body_constants.MAJOR_JOINTS if i != 0]
        m_proc = MetricsProcessor(predictions=predictions, targets=targets, joints_to_compute=joints)
        jad_err = m_proc.joint_angle_difference(use_global_rotations=False)
        post = MetricsPostProcessor(jad_err)
        num_frames = np.asarray(predictions).shape[-3]
        return np.array([post.instant(at_frame=t) for t in range(num_frames)], dtype=float)

    def compute_psds(
        self,
        predictions: np.ndarray,
        zero_velocity: bool = False,
        set_of_psd_to_use: str = "train",
    ) -> np.ndarray:
        """
        Compute the Power Spectral Density Shift (PSDS) across sliding 1-second windows
        (Equations 5-7 of the paper).

        Returns:
            1D numpy array of shape `(seq_len - framerate + 1,)` with PSDS values starting
            at 1000 ms.
        """
        from hmp_utils import EvaluationProtocol

        w_list = EvaluationProtocol.compute_wasserstein_distance_list(
            predictions=predictions,
            framerate=self.framerate,
            fk_processor=self.fk_processor,
            bodyConstants=self.body_constants,
            set_of_psd_to_use=set_of_psd_to_use,
            zero_velocity=zero_velocity,
            is_amass2h36m=self.is_amass2h36m,
        )
        return np.asarray(w_list, dtype=float)

    def save_results(self, csv_dir: Union[str, Path], print_table: bool = False) -> Path:
        """
        Save all accumulated metrics to `csv_dir` using `EvaluationProtocol.save_all_metrics`.
        """
        from hmp_utils import EvaluationProtocol

        out_dir = Path(csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        EvaluationProtocol.save_all_metrics(
            metrics_res_by_action_dict=self.metrics_res_by_action,
            metrics_dict=self.metrics_dict,
            framerate=self.framerate,
            csv_path=str(out_dir),
            print_table=print_table,
        )
        return out_dir
