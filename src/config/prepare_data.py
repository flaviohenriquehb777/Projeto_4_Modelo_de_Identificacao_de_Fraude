import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.paths import DATA_QUALITY_REPORT_PATH, PROJECT_ROOT


def load_prepare_config() -> tuple[dict, dict]:
    params_path = PROJECT_ROOT / "params.yaml"
    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params["preprocess"], params["data_quality"]


def validate_input_dataset(df: pd.DataFrame, quality_config: dict) -> dict:
    missing_columns = [column for column in quality_config["required_columns"] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas obrigatorias ausentes no dataset bruto: {missing_columns}")

    target_column = quality_config["target_column"]
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo obrigatoria ausente: {target_column}")

    row_count = len(df)
    if row_count < quality_config["minimum_rows"]:
        raise ValueError(
            f"Dataset bruto possui {row_count} linhas, abaixo do minimo {quality_config['minimum_rows']}."
        )

    null_count = int(df.isnull().sum().sum())
    if null_count > 0 and not quality_config["allow_nulls"]:
        raise ValueError(f"Dataset bruto contem {null_count} valores nulos.")

    allowed_values = set(quality_config["target_allowed_values"])
    observed_values = set(df[target_column].dropna().unique().tolist())
    if not observed_values.issubset(allowed_values):
        raise ValueError(
            f"Coluna alvo contem valores invalidos: {sorted(observed_values - allowed_values)}"
        )

    positive_rate = float(df[target_column].mean())
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        raise ValueError("Coluna alvo precisa conter classes positivas e negativas.")

    return {
        "row_count": int(row_count),
        "column_count": int(df.shape[1]),
        "null_count": int(null_count),
        "positive_rate": positive_rate,
    }


def apply_preprocessing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    prepared = df.copy()

    if config.get("amount_transform") == "yeo-johnson":
        amount_transformer = PowerTransformer(method="yeo-johnson")
        prepared["Amount"] = amount_transformer.fit_transform(prepared[["Amount"]]).ravel()

    if config.get("time_scaler") == "minmax":
        time_scaler = MinMaxScaler()
        prepared["Time"] = time_scaler.fit_transform(prepared[["Time"]]).ravel()

    return prepared


def validate_prepared_dataset(df: pd.DataFrame, quality_config: dict, preprocess_config: dict) -> dict:
    target_column = quality_config["target_column"]
    null_count = int(df.isnull().sum().sum())
    if null_count > 0 and not quality_config["allow_nulls"]:
        raise ValueError(f"Dataset preparado contem {null_count} valores nulos.")

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError("Coluna alvo deve ser numerica apos o preparo.")

    if preprocess_config.get("time_scaler") == "minmax":
        time_min = float(df["Time"].min())
        time_max = float(df["Time"].max())
        if time_min < -1e-9 or time_max > 1.0 + 1e-9:
            raise ValueError("Coluna Time deveria estar normalizada no intervalo [0, 1].")
    else:
        time_min = float(df["Time"].min())
        time_max = float(df["Time"].max())

    if not pd.Series(df["Amount"]).map(pd.notna).all():
        raise ValueError("Coluna Amount contem valores invalidos apos o preparo.")

    if not pd.Series(df["Amount"]).map(lambda value: pd.api.types.is_number(value)).all():
        raise ValueError("Coluna Amount contem valores nao numericos apos o preparo.")

    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "null_count": int(null_count),
        "time_min": time_min,
        "time_max": time_max,
        "amount_mean": float(df["Amount"].mean()),
        "amount_std": float(df["Amount"].std()),
    }


def build_quality_report(
    raw_metrics: dict,
    prepared_metrics: dict,
    preprocess_config: dict,
    input_path: str,
    output_path: str,
) -> dict:
    return {
        "status": "passed",
        "input_path": input_path,
        "output_path": output_path,
        "preprocess": preprocess_config,
        "raw_dataset": raw_metrics,
        "prepared_dataset": prepared_metrics,
    }


def save_quality_report(report: dict) -> None:
    DATA_QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_QUALITY_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def main(input_path: str, output_path: str) -> None:
    preprocess_config, quality_config = load_prepare_config()
    df = pd.read_csv(input_path, compression="infer")

    raw_metrics = validate_input_dataset(df, quality_config)
    prepared = apply_preprocessing(df, preprocess_config)
    prepared_metrics = validate_prepared_dataset(prepared, quality_config, preprocess_config)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(output_path, index=False)

    report = build_quality_report(raw_metrics, prepared_metrics, preprocess_config, input_path, output_path)
    save_quality_report(report)

    print(f"Dados preparados e salvos em {output_path}")
    print(f"Relatorio de qualidade salvo em {DATA_QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
