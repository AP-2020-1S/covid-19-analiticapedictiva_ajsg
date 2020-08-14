# Pronostico de la evolución de casos activos de SARS-CoV-2 en Colombia


## Introducción

Colombia no ha sido ajena a la crisis generada por la aparición del síndrome respiratorio agudo severo (COVID-19), enfermedad que ha llevado al limite a la sociedad debido a su impacto y a las medidas  disruptivas, rigurosas y extremas que han tomado los gobiernos para controlar su transmisión, desde su reconocimiento como una pandemia mundial el 11 de marzo de 2020. Uno los principales elementos para la toma de decisiones ha sido el pronóstico y la modelación del total de casos diagnosticados, el total de casos activos, la cantidad de casos recuperados y las muertes esperadas, tanto en el corto como en el largo plazo.

En la literatura más relevante, la evolución de casos en general ha sido abordada de dos formas diferentes: la primera es mediante el uso de modelos SIR y sus variaciones (Susceptible-Infectado-Recuperado), que son básicamente modelos de simulación basados en supuestos muy fuertes sobre la evolución de la enfermedad. la segunda aproximación corresponde al uso de metodologías de pronóstico de series de tiempo (tanto estadísticas como de inteligencia artificial) para pronosticar los nuevos casos activos o los casos confirmados; sin embargo, esta aproximación también supone fuertes hipótesis sobre como se comporta la enfermedad.

## Objetivo

El objetivo de este proyecto es la predicción de corto y mediano plazo de el total de casos confirmados, los nuevos casos, los casos activos, recuperados y muertes para las 5 principales ciudades de Colombia, utilizando técnicas estadísticas, de inteligencia artificial o modelos híbridos.

## Información utilizada

Use la información disponible en www.datos.gov.co.

## Metodología

Use la metodología basada en las siguientes fases o dimensiones del problema (ver notas de clase).

1. Definición del problema real.
2. Definición del problema de analítica.
3. Datos.
4. Metodologías propuestas.
5. Desarrollo de los modelos.
6. Despliegue de los resultados (producto de datos)
7. Gestión del ciclo de vida del producto de datos. 

Puede consultar las metodologías CRISP-DM, ASUM-DM u otras similares como complemento a las fases especificadas.


## Requerimientos

Los requerimientos del sistema son los siguientes:

* El producto de datos propuesto debe descargar la última información dispobile de forma automática cada vez que se ejecute. 

* La limpieza de datos, recalibración de los modelos, etc, deben ejecutarse automaticamente sin intervención humana. Es decir, el usuario ejecuta unicamente la rutina principal y todo el sistema se actualiza automaticamente.

* Debe presentar los resultados en un dashboard que use GitHub pages (recuerde que puede utilizar jinja2, HTML, etc)

* Debe presentar intervalos de confianza para los pronósticos.

* Debe presentar métricas de calibración para la información histórica.


## Documentación

El producto de datos debe contener información sobre los pasos de la metodología empleada, explicando las distintas fases. Recuerde que debe ganar la confianza de las personas que usarán los resultados del modelo.

## Entrega

El repo debe contener el producto desarrollado.





