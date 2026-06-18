import argparse
import json
from pathlib import Path

import pandas as pd

from src.serving.inference import load_inference_bundle, predict_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface oficial de inferencia do modelo de fraude.")
    parser.add_argument("--input", required=True, help="Arquivo CSV ou Parquet com as features de entrada.")
    parser.add_argument("--output", required=True, help="Arquivo de saida CSV ou JSON com as previsoes.")
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Formato de saida das previsoes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Opcional: limita a quantidade de linhas processadas para smoke tests.",
    )
    return parser.parse_args()


def load_input_dataframe(input_path: Path, limit: int | None) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Formato de entrada nao suportado. Use CSV ou Parquet.")

    if "Class" in df.columns:
        df = df.drop(columns="Class")

    if limit is not None:
        df = df.head(limit)

    return df


def save_predictions(predictions: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        predictions.to_csv(output_path, index=False)
        return

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(predictions.to_dict(orient="records"), file, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    bundle = load_inference_bundle()
    features = load_input_dataframe(input_path, args.limit)
    predictions = predict_dataframe(features, bundle)
    save_predictions(predictions, output_path, args.format)

    print(
        f"Inferencia concluida com {len(predictions)} linhas. "
        f"Threshold oficial aplicado: {bundle.threshold:.3f}. "
        f"Saida: {output_path}"
    )


if __name__ == "__main__":
    main()
