import ee
from typing import List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import EE_PROJECT, EE_AOI_ASSET, DRIVE_OUTPUT_FOLDER_S2

class Sentinel2ImageProcessor:
    """
    Clase para procesar y exportar imágenes Sentinel-2 desde Google Earth Engine.

    Attributes:
        project_id (str): ID del proyecto en Google Earth Engine
        aoi_path (str): Ruta del Asset con el área de interés
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'
        bands (List[str]): Lista de bandas a procesar
        output_folder (str): Carpeta de Google Drive para exportar
        crs (str): Sistema de coordenadas de referencia
        scale (int): Escala en metros
        cloud_max (int): Porcentaje máximo de nubosidad permitido
    """

    def __init__(
        self,
        project_id: str = EE_PROJECT,
        aoi_path: str = EE_AOI_ASSET,
        start_date: str = '2018-01-01',
        end_date: str = '2025-12-31',
        bands: Optional[List[str]] = None,
        output_folder: str = DRIVE_OUTPUT_FOLDER_S2,
        crs: str = 'EPSG:32630',
        scale: int = 20,
        cloud_max: int = 80,
        cloud_prob_max: int = 50,
    ):
        self.project_id = project_id
        self.aoi_path = aoi_path
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands or ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        # self.bands = bands or ['B5', 'B6', 'B7','B8A']
        self.index_bands = ['TCB', 'TCG', 'TCW']
        self.output_bands = list(dict.fromkeys(self.bands + self.index_bands))
        self.output_folder = output_folder
        self.crs = crs
        self.scale = scale
        self.cloud_max = cloud_max
        self.cloud_prob_max = cloud_prob_max

        self.aoi = None
        self.s2 = None
        self.s2_5daily = None

        # Contadores cacheados para evitar múltiples .getInfo()
        self._s2_count: Optional[int] = None
        self._mosaic_count: Optional[int] = None

        self._initialize_ee()
        self._load_aoi()
        self._load_sentinel2_images()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _initialize_ee(self) -> None:
        """Inicializa la autenticación y conexión con Earth Engine."""
        try:
            ee.Initialize(project=self.project_id)
            print(f"✓ Earth Engine inicializado con proyecto: {self.project_id}")
        except Exception:
            print("Autenticación requerida...")
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            print("✓ Earth Engine autenticado e inicializado")

    def _load_aoi(self) -> None:
        """Carga el área de interés desde un Asset de Earth Engine."""
        try:
            self.aoi = ee.FeatureCollection(self.aoi_path)
            print(f"✓ Área de interés cargada desde: {self.aoi_path}")
        except Exception as e:
            raise ValueError(f"Error al cargar AOI: {e}")

    # ------------------------------------------------------------------
    # Máscara de nubes
    # ------------------------------------------------------------------

    def _mask_s2_clouds(self, image: ee.Image) -> ee.Image:
        """
        Enmascara píxeles con nubes combinando SCL + S2_CLOUD_PROBABILITY.

        SCL – clases enmascaradas (excluidas):
            1  = Saturado / Defectuoso
            3  = Sombra de nubes
            8  = Nube probabilidad media
            9  = Nube probabilidad alta
            10 = Cirros finos

        Cloud Probability – umbral: self.cloud_prob_max (default 50%).
        """
        scl = image.select('SCL')
        scl_mask = (
            scl.neq(1)
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
        )

        cloud_prob = ee.Image(image.get('s2cloudless')).select('probability')
        prob_mask = cloud_prob.lt(self.cloud_prob_max)

        return image.updateMask(scl_mask.And(prob_mask))

    # ------------------------------------------------------------------
    # Carga de imágenes
    # ------------------------------------------------------------------

    def _load_sentinel2_images(self) -> None:
        """
        Carga imágenes Sentinel-2 del período especificado.

        Siempre incluye SCL para poder enmascarar. Si el usuario no la
        pidió en `bands`, se elimina tras enmascarar.
        """


        s2_sr = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(self.aoi)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_max))
        )
        s2_clouds = (
            ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(self.aoi)
            .filterDate(self.start_date, self.end_date)
        )
        joined = ee.ImageCollection(
            ee.Join.saveFirst('s2cloudless').apply(
                primary=s2_sr,
                secondary=s2_clouds,
                condition=ee.Filter.equals(
                    leftField='system:index',
                    rightField='system:index'
                )
            )
        )
        raw = joined.map(self._mask_s2_clouds)
        def calculate_index(image):
            # Tasseled Cap Brightness
            TCB = (
                image.select('B2').multiply(0.3510)
                .add(image.select('B3').multiply(0.3813))
                .add(image.select('B4').multiply(0.3437))
                .add(image.select('B8').multiply(0.7196))
                .add(image.select('B11').multiply(0.2396))
                .add(image.select('B12').multiply(0.1949))
            ).round().toInt16()
            # Tasseled Cap Greenness
            TCG = (
                image.select('B2').multiply(-0.3599)
                .add(image.select('B3').multiply(-0.3533))
                .add(image.select('B4').multiply(-0.4734))
                .add(image.select('B8').multiply(0.663300))
                .add(image.select('B11').multiply(0.0087))
                .add(image.select('B12').multiply(-0.2856))
            ).round().toInt16()
            # Tasseled Cap Wetness
            TCW = (
                image.select('B2').multiply(0.2578)
                .add(image.select('B3').multiply(0.2305))
                .add(image.select('B4').multiply(0.0883))
                .add(image.select('B8').multiply(0.1071))
                .add(image.select('B11').multiply(-0.7611))
                .add(image.select('B12').multiply(-0.5308))
            ).round().toInt16()
            return image.addBands(TCB.rename('TCB')).addBands(TCG.rename('TCG')).addBands(TCW.rename('TCW'))
            # Agregar índices Tasseled Cap
        raw = raw.map(calculate_index)
            
        if 'SCL' in self.bands:
            self.s2 = raw.select(list(dict.fromkeys(self.output_bands + ['SCL'])))
        else:
            self.s2 = raw.select(self.output_bands)

        self._s2_count = self.s2.size().getInfo()
        print(f"✓ {self._s2_count} imágenes Sentinel-2 cargadas "
              f"({self.start_date} → {self.end_date}, nubosidad < {self.cloud_max}%)")

    # ------------------------------------------------------------------
    # Mosaicos cada N días (100% server-side)
    # ------------------------------------------------------------------

    def create_ndaily_mosaic(self, n_days: int = 5) -> 'Sentinel2ImageProcessor':
        """
        Crea mosaicos cada n días, completamente en server-side.

        Args:
            n_days: Intervalo en días para cada mosaico (default: 5).

        Returns:
            self: Para encadenamiento de métodos.
        """
        start = ee.Date(self.start_date)
        end = ee.Date(self.end_date)

        n_periods = end.difference(start, 'day').divide(n_days).ceil()
        intervals = ee.List.sequence(0, n_periods.subtract(1))

        def mosaic_period(i):
            date = start.advance(ee.Number(i).multiply(n_days), 'day')
            # FIX #5: el período no puede superar end_date
            raw_end = date.advance(n_days, 'day')
            period_end = ee.Date(ee.Algorithms.If(
                raw_end.millis().gt(end.millis()), end, raw_end
            ))
            filtered = self.s2.filterDate(date, period_end)
            mosaic = filtered.mosaic()
            return (
                mosaic
                .set('system:time_start', date.millis())
                .set('system:index', date.format('yyyy-MM-dd'))
                .set('n_bands', mosaic.bandNames().length())
            )

        all_mosaics = ee.ImageCollection(intervals.map(mosaic_period))
        # Filtrar mosaicos vacíos (períodos sin imágenes)
        self.s2_5daily = all_mosaics.filter(ee.Filter.gt('n_bands', 0))

        # Calcular conteo localmente para no exceder memoria de GEE
        from datetime import datetime as _dt
        _start = _dt.strptime(self.start_date, '%Y-%m-%d')
        _end = _dt.strptime(self.end_date, '%Y-%m-%d')
        self._mosaic_count = int((_end - _start).days / n_days) + (1 if (_end - _start).days % n_days else 0)
        print(f"✓ ~{self._mosaic_count} mosaicos cada {n_days} días creados")
        return self

    # ------------------------------------------------------------------
    # Apilado temporal de bandas
    # ------------------------------------------------------------------

    def _stack_band_with_unique_names(
        self, ic: ee.ImageCollection, band: str
    ) -> ee.Image:
        """
        Apila una banda en el tiempo. Los nombres de cada capa siguen el
        patrón '{system:index}_{band}' que toBands() genera automáticamente
        cuando system:index está correctamente asignado.

        Args:
            ic: ImageCollection con system:index asignado.
            band: Nombre de la banda a apilar.

        Returns:
            Imagen multibanda con una capa por fecha.
        """
        # FIX #1: eliminado .getInfo() innecesario dentro del bucle por banda
        # FIX #2: toBands() usa system:index como prefijo automáticamente,
        #          no hace falta rename_band por separado (que era ignorado).
        #          El resultado será: '{yyyy-MM-dd}_{band}'
        return ic.select([band]).toBands()

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def export_bands_to_drive(
        self, collection: Optional[ee.ImageCollection] = None
    ) -> None:
        """
        Exporta cada banda como GeoTIFF independiente a Google Drive.

        Args:
            collection: ImageCollection a exportar. Si es None usa self.s2_5daily.
        """
        if collection is None:
            collection = self.s2_5daily

        if collection is None:
            raise ValueError(
                "No hay colección de mosaicos. "
                "Ejecuta create_ndaily_mosaic() o create_daily_mosaic() primero."
            )

        # FIX #6: verificar que la colección tiene imágenes antes de exportar
        count = self._mosaic_count if collection is self.s2_5daily else collection.size().getInfo()
        if count == 0:
            raise ValueError(
                "La colección de mosaicos está vacía. "
                "Revisa el rango de fechas o el umbral de nubosidad."
            )

        aoi_geom = self.aoi.geometry()

        for band in self.output_bands:
            band_stack = self._stack_band_with_unique_names(collection, band)
            band_stack = band_stack.clip(aoi_geom)
            band_stack = band_stack.round().toInt16()

            description = f'S2_{band}_{self.start_date}_{self.end_date}'
            file_prefix = f'S2_{band}_{self.start_date}_{self.end_date}'

            task = ee.batch.Export.image.toDrive(
                image=band_stack,
                description=description,
                folder=self.output_folder,
                fileNamePrefix=file_prefix,
                scale=self.scale,
                region=aoi_geom,
                crs=self.crs,
                maxPixels=1e13,
            )
            task.start()
            print(f"✓ Export iniciado → banda {band} | task: {task.id}")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Retorna información sobre la configuración actual (usa valores cacheados)."""
        # FIX #7: indicar claramente cuando el mosaico aún no se ha generado
        mosaic_info = (
            self._mosaic_count
            if self._mosaic_count is not None
            else "No generado aún (llama a create_ndaily_mosaic() o create_daily_mosaic())"
        )
        return {
            'project_id': self.project_id,
            'aoi_path': self.aoi_path,
            'date_range': f"{self.start_date} → {self.end_date}",
            'bands': self.bands,
            'indices': self.index_bands,
            'output_bands': self.output_bands,
            'cloud_max': self.cloud_max,
            'output_folder': self.output_folder,
            'crs': self.crs,
            'scale': self.scale,
            's2_images_count': self._s2_count,
            's2_mosaic_count': mosaic_info,
        }


# ------------------------------------------------------------------
# Ejemplo de uso
# ------------------------------------------------------------------
if __name__ == '__main__':
    processor = Sentinel2ImageProcessor(
        project_id=EE_PROJECT,
        aoi_path='projects/ee-tomaspugni/assets/rioja',
        start_date='2018-01-01',
        end_date='2025-12-31',
        output_folder=DRIVE_OUTPUT_FOLDER_S2,
        scale=20,
        crs='EPSG:32630',
        cloud_max=100,
        cloud_prob_max=30,
        bands=[ 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A',  'B11', 'B12']
    )

    processor.create_ndaily_mosaic(n_days=5).export_bands_to_drive()
    print(processor.get_info())