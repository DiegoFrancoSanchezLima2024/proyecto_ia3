# Piloto preentrenado - no es resultado final

Configuracion:

- EfficientNet-B0 ImageNet;
- tres epocas;
- 100 lotes de train y 20 lotes de validacion por epoca;
- encoder congelado;
- batch 16, FP16 y dos vistas;
- evaluacion final sobre las 892 fotos completas de validacion.

Resultados:

- baseline media de train: 4.005 cm MAE global;
- baseline Ridge con altura y genero: 3.188 cm;
- red multivista con BatchNorm adaptativo despues del piloto: 2.752 cm;
- porcentaje global dentro de +/-2 cm: 58.3%;
- mejores resultados: wrist 0.76 cm, ankle 0.88 cm, shoulder-breadth 1.20 cm;
- principales brechas: waist 7.26 cm, chest 5.69 cm, hip 5.47 cm.

Una ablacion congelando tambien las estadisticas BatchNorm obtuvo 3.21 cm y fue
descartada como configuracion final. El perfil definitivo adapta BatchNorm durante
las tres primeras epocas y luego lo congela.

Interpretacion: el pipeline aprende y supera los baselines simples, pero no cumple
todavia el objetivo de sastreria en los contornos principales. No se deben usar
estos pesos para generar patrones. El siguiente experimento debe entrenar el
calendario completo, comparar la perdida heterocedastica contra Huber simple y
evaluar Test-A/Test-B sin ajustar sobre ellos.
