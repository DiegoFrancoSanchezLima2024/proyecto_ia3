"""Carga reproducible del regresor SHAPY en Windows."""

from __future__ import annotations

from pathlib import Path

from .shapy_compat import importar_shapy


def resolved_shapy_config(project_root: Path):
    """Combina el YAML oficial y sustituye rutas relativas por rutas reales."""

    from omegaconf import OmegaConf

    root = Path(project_root).resolve()
    body3d = root / "pretrained" / "body3d"
    base_config, _ = importar_shapy(root)
    config = OmegaConf.create(OmegaConf.to_container(base_config))
    official_yaml = (
        body3d
        / "shapy-master"
        / "regressor"
        / "configs"
        / "b2a_expose_hrnet_demo.yaml"
    )
    config.merge_with(OmegaConf.load(str(official_yaml)))

    model_root = body3d / "models" / "smplx" / "models"
    expose_data = body3d / "expose_release" / "data"
    flame_data = body3d / "expose_release" / "utility_files" / "flame"

    config.output_folder = str(root / "outputs" / "body3d")
    config.use_cuda = True
    config.body_model.model_folder = str(model_root)
    config.body_model.smplx.mean_pose_path = str(expose_data / "all_means.pkl")
    config.body_model.smplx.j14_regressor_path = str(
        expose_data / "SMPLX_to_J14.pkl"
    )
    config.body_model.smplx.head_verts_ids_path = str(
        flame_data / "SMPL-X__FLAME_vertex_ids.npy"
    )
    shape_priors = body3d / "utility_files" / "shape_priors"
    gender_prior = config.losses.body.shape.prior.gender_shape
    gender_prior.female_stats_path = str(shape_priors / "female_normal.npz")
    gender_prior.male_stats_path = str(shape_priors / "male_normal.npz")

    # El checkpoint completo se carga después de construir la arquitectura.
    # Evita que HRNet intente interpretar ese archivo como pesos de backbone.
    config.network.smplx.backbone.hrnet.pretrained_path = ""

    # Sastre-IA mide la malla con su módulo propio. Esto elimina la extensión
    # CUDA antigua mesh-mesh-intersection y los modelos Lightning de atributos.
    config.network.smplx.compute_measurements = False
    config.network.smplx.use_b2a = False
    config.network.smplx.use_a2b = False
    return config


def cargar_modelo_shapy(project_root: Path, device: str = "cuda"):
    """Construye SHAPY, carga el checkpoint oficial y devuelve un reporte."""

    import torch

    root = Path(project_root).resolve()
    config = resolved_shapy_config(root)
    _, build_model = importar_shapy(root)
    model_dict = build_model(config)
    network = model_dict["network"]

    checkpoint_path = (
        root
        / "pretrained"
        / "body3d"
        / "trained_models"
        / "shapy"
        / "SHAPY_A"
        / "checkpoints"
        / "best_checkpoint"
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = network.load_state_dict(state_dict, strict=False)

    target = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    network = network.to(target).eval()
    report = {
        "checkpoint": str(checkpoint_path),
        "device": str(target),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
    }
    return network, config, report
