"""
framework_ami/cargador.py
=========================
Carga datos raw AMI (ACTIVE_[Mes]_N.parquet), clasifica casas
residencial/comercial y construye las 4 muestras solicitadas.

Modos de muestreo (SamplingMode):
  SINGLE_HOUSE  : 1 casa × 12 meses, ventanas semana completa (lun-dom)
  WEEKLY_MONTH  : todas las casas, 1 semana representativa por mes
  MONTHLY       : todas las casas, mes completo
  ANNUAL        : todas las casas, año completo

Ventana semanal: lun 00:00 → dom 23:00 (168 lecturas horarias brutas).
Si la semana no se completa en el mes, se completa con el mes siguiente.
"""

from __future__ import annotations
import os, gc, warnings
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "Raw_Processed"

MESES_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
MES_NUM = {m: i + 1 for i, m in enumerate(MESES_ES)}

HORAS_SEMANA = 7 * 24   # 168

# Hora pico residencial: 18-23 h; comercial: 8-17 h
PEAK_RESIDENCIAL = set(range(18, 24))
PEAK_COMERCIAL   = set(range(8, 18))


class SamplingMode(Enum):
    SINGLE_HOUSE  = "single_house"
    WEEKLY_MONTH  = "weekly_month"
    MONTHLY       = "monthly"
    ANNUAL        = "annual"


# ══════════════════════════════════════════════════════════════════════════════
class CargadorAMI:
    """
    Punto de entrada principal para cargar y muestrear datos AMI.

    Parámetros
    ----------
    raw_dir : str | Path
        Directorio con archivos ACTIVE_[Mes]_N.parquet
    seed : int
        Semilla de reproducibilidad
    max_files_per_month : int
        Máximo de archivos a leer por mes (None = todos)
    """

    def __init__(
        self,
        raw_dir: str | Path = RAW_DIR,
        seed: int = 42,
        max_files_per_month: Optional[int] = None,
        min_dias: int = 30,
    ):
        self.raw_dir = Path(raw_dir)
        self.seed    = seed
        self.rng     = np.random.default_rng(seed)
        self.max_files = max_files_per_month
        # Cobertura mínima (días con dato) para aceptar un medidor. Con el
        # dataset completo se usa 30; con la MUESTRA anonimizada (un mes,
        # cobertura parcial) se baja, p. ej. a 7, para que la demo funcione.
        self.min_dias = min_dias

        self._cache: Dict[str, pd.DataFrame] = {}   # mes → DataFrame mensual
        self._casas_info: Optional[pd.DataFrame] = None

    # ── Carga de archivos raw ─────────────────────────────────────────────────
    def _archivos_mes(self, mes: str) -> List[Path]:
        files = sorted(self.raw_dir.glob(f"ACTIVE_{mes}_*.parquet"))
        if self.max_files:
            files = files[: self.max_files]
        return files

    def _leer_mes(self, mes: str, use_cache: bool = True) -> pd.DataFrame:
        if use_cache and mes in self._cache:
            return self._cache[mes]

        archivos = self._archivos_mes(mes)
        if not archivos:
            return pd.DataFrame()

        partes = []
        for f in archivos:
            df = pd.read_parquet(f, columns=["ID", "Year", "Month", "Day", "Hour", "Value"])
            # Homogenizar tipos (Oct/Nov guardan Value como string)
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
            df["ID"]    = df["ID"].astype(str)
            partes.append(df)

        df_mes = pd.concat(partes, ignore_index=True)
        df_mes["timestamp"] = pd.to_datetime(
            df_mes[["Year", "Month", "Day", "Hour"]].rename(
                columns={"Year": "year", "Month": "month",
                         "Day": "day",   "Hour": "hour"}
            )
        )
        df_mes = (
            df_mes
            .drop_duplicates(subset=["ID", "timestamp"])
            .sort_values(["ID", "timestamp"])
            .reset_index(drop=True)
        )
        df_mes.dropna(subset=["Value"], inplace=True)

        if use_cache:
            self._cache[mes] = df_mes
        return df_mes

    def _leer_meses(self, meses: List[str]) -> pd.DataFrame:
        partes = [self._leer_mes(m) for m in meses if self._archivos_mes(m)]
        return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

    # ── Clasificación de casas ────────────────────────────────────────────────
    def clasificar_casas(
        self,
        meses_ref: Optional[List[str]] = None,
        min_dias: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Clasifica casas en 'residencial' o 'comercial' usando el perfil
        horario medio de cada medidor.

        Criterio:
            Hora pico en [18-23] → residencial (consumo nocturno doméstico)
            Hora pico en [8-17]  → comercial   (consumo diurno laboral)
            Otros                → indefinido

        También calcula cobertura (días con dato) para selección de casas.
        """
        if self._casas_info is not None:
            return self._casas_info

        if min_dias is None:
            min_dias = self.min_dias   # usa el umbral fijado en el constructor
        if meses_ref is None:
            meses_ref = MESES_ES[:6]   # usar primeros 6 meses como referencia

        print(f"Clasificando casas con meses: {meses_ref}")
        df = self._leer_meses(meses_ref)
        if df.empty:
            return pd.DataFrame()

        df["hour_of_day"] = df["timestamp"].dt.hour

        # Perfil horario medio por casa
        perfil = (
            df.groupby(["ID", "hour_of_day"])["Value"]
            .mean()
            .unstack(fill_value=0)    # shape: (n_casas, 24)
        )
        perfil.columns = [int(c) for c in perfil.columns]

        # Hora pico y tipo
        hora_pico = perfil.idxmax(axis=1)
        tipo = hora_pico.map(
            lambda h: "residencial" if h in PEAK_RESIDENCIAL
                      else ("comercial" if h in PEAK_COMERCIAL else "indefinido")
        )

        # Cobertura: días únicos con datos
        cobertura = df.groupby("ID")["timestamp"].apply(
            lambda s: s.dt.date.nunique()
        )

        # Consumo medio (para seleccionar casas con actividad real)
        consumo_medio = df.groupby("ID")["Value"].mean()

        info = pd.DataFrame({
            "tipo": tipo,
            "hora_pico": hora_pico,
            "dias_cobertura": cobertura,
            "consumo_medio": consumo_medio,
        })
        info = info[info["dias_cobertura"] >= min_dias].copy()

        self._casas_info = info
        print(f"Casas clasificadas: {len(info)} total")
        print(info["tipo"].value_counts().to_string())
        return info

    def seleccionar_casas(
        self,
        n_residencial: int = 6,
        n_comercial: int = 3,
        meses_ref: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Devuelve (ids_residenciales, ids_comerciales) con mayor cobertura.
        """
        info = self.clasificar_casas(meses_ref=meses_ref)

        res = (
            info[info["tipo"] == "residencial"]
            .sort_values("dias_cobertura", ascending=False)
            .head(n_residencial)
            .index.tolist()
        )
        com = (
            info[info["tipo"] == "comercial"]
            .sort_values("dias_cobertura", ascending=False)
            .head(n_comercial)
            .index.tolist()
        )

        if len(res) < n_residencial:
            print(f"[!] Solo {len(res)}/{n_residencial} casas residenciales disponibles")
        if len(com) < n_comercial:
            print(f"[!] Solo {len(com)}/{n_comercial} casas comerciales disponibles")

        return res, com

    # ── Ventanas semanales ────────────────────────────────────────────────────
    @staticmethod
    def _primer_lunes(timestamps: pd.Series) -> Optional[pd.Timestamp]:
        """Primer lunes a las 00:00 dentro de la serie."""
        for ts in timestamps.sort_values():
            if ts.dayofweek == 0 and ts.hour == 0:
                return ts
        return None

    def _ventana_semana(
        self,
        df_casa: pd.DataFrame,
        semana_inicio: pd.Timestamp,
        df_next_mes: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.Series]:
        """
        Extrae una ventana de 168 horas a partir de semana_inicio.
        Si df_next_mes se provee, complementa con datos del mes siguiente.
        Devuelve Serie de 168 valores (índice = 0..167) o None si insuficiente.
        """
        semana_fin = semana_inicio + pd.Timedelta(hours=167)
        ts_range   = pd.date_range(semana_inicio, semana_fin, freq="1h")

        # Reindexar datos de la casa a la ventana solicitada
        s = df_casa.set_index("timestamp")["Value"].reindex(ts_range)

        # Completar con siguiente mes si hay huecos al final
        if s.isna().any() and df_next_mes is not None:
            s_next = df_next_mes.set_index("timestamp")["Value"]
            missing = s[s.isna()].index
            s.update(s_next.reindex(missing))

        # Interpolar huecos cortos (<= 3 horas) dentro de la semana
        n_missing = s.isna().sum()
        if n_missing > 0 and n_missing <= 3:
            s = s.interpolate(method="linear", limit=3)

        # Descartar semana con más del 10% de datos faltantes
        if s.isna().sum() > HORAS_SEMANA * 0.10:
            return None

        s = s.fillna(0)
        s.index = range(HORAS_SEMANA)
        return s

    def _semanas_de_mes(
        self,
        df_mes: pd.DataFrame,
        casa_id: str,
        df_next_mes: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[pd.Timestamp, np.ndarray]]:
        """
        Extrae todas las ventanas semanales válidas de un mes para una casa.
        Devuelve lista de (timestamp_inicio, array_168).
        """
        df_casa = df_mes[df_mes["ID"] == casa_id].copy()
        if df_casa.empty:
            return []

        # Solo timestamps con hora=0 y lunes para buscar inicio de semanas
        ts_min = df_casa["timestamp"].min()
        ts_max = df_casa["timestamp"].max()

        resultados = []
        cur = ts_min.normalize()
        # Avanzar hasta el primer lunes
        days_to_monday = (7 - cur.dayofweek) % 7
        cur += pd.Timedelta(days=days_to_monday)

        while cur + pd.Timedelta(hours=167) <= ts_max + pd.Timedelta(days=7):
            semana = self._ventana_semana(df_casa, cur, df_next_mes)
            if semana is not None:
                resultados.append((cur, semana.values))
            cur += pd.Timedelta(weeks=1)
            if cur > ts_max + pd.Timedelta(days=7):
                break

        return resultados

    # ══════════════════════════════════════════════════════════════════════════
    # MODO 1: UNA SOLA CASA A LO LARGO DEL AÑO
    # ══════════════════════════════════════════════════════════════════════════
    def muestreo_single_house(
        self,
        casa_id: str,
        meses: Optional[List[str]] = None,
    ) -> Dict[str, List[Tuple[pd.Timestamp, np.ndarray]]]:
        """
        Para una casa dada, devuelve ventanas semanales por mes.

        Retorno: {mes: [(ts_inicio, array_168), ...]}
        """
        if meses is None:
            meses = [m for m in MESES_ES if self._archivos_mes(m)]

        print(f"[SINGLE_HOUSE] Casa {casa_id} — {len(meses)} meses")
        resultado: Dict = {}

        for i, mes in enumerate(meses):
            df_mes  = self._leer_mes(mes)
            if df_mes.empty or casa_id not in df_mes["ID"].values:
                continue

            mes_next = meses[i + 1] if i + 1 < len(meses) else None
            df_next  = self._leer_mes(mes_next) if mes_next else None
            df_next_casa = (
                df_next[df_next["ID"] == casa_id].copy()
                if df_next is not None and not df_next.empty else None
            )

            semanas = self._semanas_de_mes(df_mes, casa_id, df_next_casa)
            if semanas:
                resultado[mes] = semanas
                print(f"  {mes}: {len(semanas)} semanas")

        return resultado

    # ══════════════════════════════════════════════════════════════════════════
    # MODO 2: SEMANA REPRESENTATIVA POR MES, TODAS LAS CASAS
    # ══════════════════════════════════════════════════════════════════════════
    def muestreo_weekly_month(
        self,
        meses: Optional[List[str]] = None,
        semana_num: int = 1,           # qué semana del mes (1=primera completa)
    ) -> Dict[str, pd.DataFrame]:
        """
        Para cada mes: una semana representativa, un registro por casa.

        Retorno: {mes: DataFrame(columns=['ID', 'tipo', 'ts_inicio', 'h0'..'h167'])}
        """
        if meses is None:
            meses = [m for m in MESES_ES if self._archivos_mes(m)]

        info_casas = self.clasificar_casas()
        resultado: Dict = {}

        for i, mes in enumerate(meses):
            print(f"[WEEKLY_MONTH] {mes}...")
            df_mes   = self._leer_mes(mes)
            if df_mes.empty:
                continue

            mes_next = meses[i + 1] if i + 1 < len(meses) else None
            df_next  = self._leer_mes(mes_next) if mes_next else None

            ids_mes  = df_mes["ID"].unique()
            filas = []

            for casa_id in ids_mes:
                df_next_casa = (
                    df_next[df_next["ID"] == casa_id].copy()
                    if df_next is not None and not df_next.empty else None
                )
                semanas = self._semanas_de_mes(df_mes, casa_id, df_next_casa)
                if not semanas:
                    continue

                # Tomar semana_num-ésima (índice base 0)
                idx = min(semana_num - 1, len(semanas) - 1)
                ts_inicio, arr = semanas[idx]

                tipo = info_casas.loc[casa_id, "tipo"] if casa_id in info_casas.index else "indefinido"

                fila = {"ID": casa_id, "tipo": tipo, "ts_inicio": ts_inicio}
                fila.update({f"h{j}": arr[j] for j in range(HORAS_SEMANA)})
                filas.append(fila)

            if filas:
                df_out = pd.DataFrame(filas)
                resultado[mes] = df_out
                print(f"  {mes}: {len(df_out)} casas")

        return resultado

    # ══════════════════════════════════════════════════════════════════════════
    # MODO 3: MES COMPLETO, TODAS LAS CASAS
    # ══════════════════════════════════════════════════════════════════════════
    def muestreo_monthly(
        self,
        meses: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Para cada mes: todas las ventanas semanales de todas las casas.

        Retorno: {mes: DataFrame(columns=['ID','tipo','ts_inicio','h0'..'h167'])}
        """
        if meses is None:
            meses = [m for m in MESES_ES if self._archivos_mes(m)]

        info_casas = self.clasificar_casas()
        resultado: Dict = {}

        for i, mes in enumerate(meses):
            print(f"[MONTHLY] {mes}...")
            df_mes  = self._leer_mes(mes)
            if df_mes.empty:
                continue

            mes_next = meses[i + 1] if i + 1 < len(meses) else None
            df_next  = self._leer_mes(mes_next) if mes_next else None

            filas = []
            for casa_id in df_mes["ID"].unique():
                df_next_casa = (
                    df_next[df_next["ID"] == casa_id].copy()
                    if df_next is not None and not df_next.empty else None
                )
                semanas = self._semanas_de_mes(df_mes, casa_id, df_next_casa)
                tipo = info_casas.loc[casa_id, "tipo"] if casa_id in info_casas.index else "indefinido"
                for ts_inicio, arr in semanas:
                    fila = {"ID": casa_id, "tipo": tipo, "ts_inicio": ts_inicio}
                    fila.update({f"h{j}": arr[j] for j in range(HORAS_SEMANA)})
                    filas.append(fila)

            if filas:
                df_out = pd.DataFrame(filas)
                resultado[mes] = df_out
                print(f"  {mes}: {len(df_out)} series semanales de {df_out['ID'].nunique()} casas")

        return resultado

    # ══════════════════════════════════════════════════════════════════════════
    # MODO 4: AÑO COMPLETO, TODAS LAS CASAS
    # ══════════════════════════════════════════════════════════════════════════
    def muestreo_annual(
        self,
        meses: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Todas las ventanas semanales de todas las casas durante todo el año.

        Retorno: DataFrame con columnas ['ID','tipo','mes','ts_inicio','h0'..'h167']
        """
        monthly = self.muestreo_monthly(meses=meses)
        partes  = []
        for mes, df in monthly.items():
            df = df.copy()
            df["mes"] = mes
            partes.append(df)

        if not partes:
            return pd.DataFrame()

        df_all = pd.concat(partes, ignore_index=True)
        print(f"\n[ANNUAL] Total series: {len(df_all)} de {df_all['ID'].nunique()} casas")
        return df_all

    # ── Utilidades ────────────────────────────────────────────────────────────
    def H_COLS(self) -> List[str]:
        return [f"h{i}" for i in range(HORAS_SEMANA)]

    def extraer_matriz(self, df: pd.DataFrame) -> np.ndarray:
        """Devuelve matriz (N, 168) con las series brutas."""
        return df[self.H_COLS()].values.astype(float)

    def resumen(self, df: pd.DataFrame) -> None:
        print(f"Shape: {df.shape}")
        if "tipo" in df.columns:
            print(df["tipo"].value_counts().to_string())
        if "mes" in df.columns:
            print(df["mes"].value_counts().to_string())
