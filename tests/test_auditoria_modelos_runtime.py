from scripts.auditar_modelos_runtime import construir_auditoria


def test_ruta_oficial_declara_solo_los_componentes_reales():
    auditoria = construir_auditoria()
    estados = {item["nombre"]: item["estado"] for item in auditoria["modelos"]}

    assert auditoria["ruta_oficial_consistente"] is True
    assert estados["MediaPipe Pose Landmarker Lite"] == "activo_oficial"
    assert estados["SAM 2.1 Hiera Small"] == "activo_oficial"
    assert estados["exp_008_perfiles_arboles (ExtraTrees)"] == "activo_oficial"
    assert estados["exp_007_multitarea_sin_peso (CNN)"] == "ablacion_rechazada"
    assert estados["SHAPY + SMPL-X"] == "instalado_no_conectado"


def test_todos_los_artefactos_auditados_existen():
    auditoria = construir_auditoria()
    faltantes = [
        item["nombre"]
        for item in auditoria["modelos"]
        if not item["artefactos_presentes"]
    ]
    assert faltantes == []
