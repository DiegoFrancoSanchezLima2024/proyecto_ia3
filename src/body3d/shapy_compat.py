"""Compatibilidad acotada para ejecutar la configuración antigua de SHAPY."""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path
from typing import Callable


_ORIGINAL_DATACLASS = dataclasses.dataclass


def install_legacy_dataclass_compat() -> None:
    """Permite defaults anidados de SHAPY sin alterar dataclasses externas.

    Python 3.11 rechaza instancias no-hashables como defaults. El repositorio
    original de SHAPY fue escrito antes de esa validación. Hacer hashables solo
    sus clases de configuración reproduce el comportamiento anterior y evita
    modificar el código licenciado descargado por el usuario.
    """

    if getattr(dataclasses.dataclass, "_sastre_shapy_compat", False):
        return

    def compatible_dataclass(cls=None, **kwargs):
        def wrap(target):
            scoped_kwargs = dict(kwargs)
            if (
                (
                    target.__module__.startswith("human_shape.config")
                    or target.__name__ == "Struct"
                )
                and "unsafe_hash" not in scoped_kwargs
            ):
                scoped_kwargs["unsafe_hash"] = True
            return _ORIGINAL_DATACLASS(target, **scoped_kwargs)

        return wrap(cls) if cls is not None else wrap

    compatible_dataclass._sastre_shapy_compat = True
    dataclasses.dataclass = compatible_dataclass


def add_shapy_to_path(project_root: Path) -> Path:
    source_root = Path(project_root) / "pretrained" / "body3d" / "shapy-master"
    regressor_root = source_root / "regressor"
    if not (regressor_root / "human_shape" / "__init__.py").is_file():
        raise FileNotFoundError(f"No se encontró SHAPY en {regressor_root}")
    path = str(regressor_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return regressor_root


def _install_inference_stubs() -> None:
    """Aísla subsistemas de entrenamiento que el inferidor no necesita.

    SHAPY importa de forma ansiosa su extensión CUDA de intersección y los
    modelos Lightning de atributos. La malla final se mide en Sastre-IA, así
    que esos componentes no participan en la inferencia de forma y se pueden
    sustituir por guardas explícitas.
    """

    import torch.nn as nn

    class _UnavailableTrainingComponent(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError(
                "Este componente de entrenamiento no está habilitado en el "
                "runner de inferencia de Sastre-IA."
            )

        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            return cls(*args, **kwargs)

    if "body_measurements" not in sys.modules:
        module = types.ModuleType("body_measurements")
        module.BodyMeasurements = _UnavailableTrainingComponent
        sys.modules["body_measurements"] = module

    if "attributes" not in sys.modules:
        module = types.ModuleType("attributes")
        module.A2B = _UnavailableTrainingComponent
        module.B2A = _UnavailableTrainingComponent
        sys.modules["attributes"] = module


def importar_shapy(project_root: Path) -> tuple[object, Callable]:
    """Carga configuración y constructor oficial con compatibilidad Windows."""

    # Torch debe inicializarse antes del parche para no afectar sus dataclasses.
    import torch  # noqa: F401
    import matplotlib
    import matplotlib.cm as mpl_cm
    import torchvision.models.resnet as tv_resnet

    # Matplotlib reciente movió get_cmap a su registro global.
    if not hasattr(mpl_cm, "get_cmap"):
        mpl_cm.get_cmap = matplotlib.colormaps.get_cmap
    # Torchvision retiró este diccionario; SHAPY solo necesita que el símbolo
    # exista durante el import (la configuración seleccionada usa HRNet).
    if not hasattr(tv_resnet, "model_urls"):
        tv_resnet.model_urls = {}

    install_legacy_dataclass_compat()
    add_shapy_to_path(project_root)
    _install_inference_stubs()
    from human_shape.config.defaults import conf
    from human_shape.models.build import build_model

    # El Struct dinámico original usa dataclasses con arrays NumPy como
    # defaults, algo prohibido desde Python 3.11. Para cargar NPZ basta un
    # contenedor de atributos equivalente.
    from human_shape.models.body_models import body_models as body_models_module

    class ArrayStruct:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def keys(self):
            return list(self.__dict__)

    body_models_module.Struct = ArrayStruct

    return conf, build_model
