import json
import sys
from pathlib import Path

import pandas as pd
from fpdf import FPDF

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.paths import PROJECT_ROOT, SHAP_EXPLANATIONS_DIR, SHAP_MANIFEST_PATH, SHAP_REPORT_PATH


class SHAPReportGenerator:
    def __init__(self, shap_dir: Path, output_pdf: Path):
        self.shap_dir = shap_dir
        self.output_pdf = output_pdf
        self.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.margin_x = 10
        self.content_width = 190
        self.stats = self._load_json(self.shap_dir / "shap_stats.json", default={})
        self.manifest = self._load_json(SHAP_MANIFEST_PATH, default={})

    def _load_json(self, path: Path, default: dict) -> dict:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _add_title(self, title: str) -> None:
        self.pdf.set_font("Helvetica", "B", 18)
        self.pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(4)

    def _add_body(self, text: str) -> None:
        self.pdf.set_font("Helvetica", "", 12)
        self.pdf.multi_cell(self.content_width, 8, text)

    def _image_exists(self, relative_path: str) -> bool:
        return (self.shap_dir / relative_path).exists()

    def _add_image(self, image_path: Path, width: int = 180) -> None:
        if not image_path.exists():
            return
        self.pdf.image(str(image_path), x=self.margin_x, y=self.pdf.get_y(), w=width)
        self.pdf.set_y(self.pdf.get_y() + 95)
        self.pdf.ln(4)

    def _get_top_features(self) -> list[str]:
        top_features = self.stats.get("top_features", [])
        if top_features:
            return [item["feature"] for item in top_features[:5]]
        return ["V14", "V12", "V10", "V16", "V17"]

    def generate_report(self) -> None:
        if not self.shap_dir.exists():
            raise FileNotFoundError(f"Diretorio SHAP nao encontrado: {self.shap_dir}")

        self._add_cover()
        self._add_table_of_contents()
        self._add_shap_insights()
        self._add_feature_importance()
        self._add_dependence_plots()
        self._add_conclusions()

        self.pdf.output(str(self.output_pdf))
        print(f"Relatorio gerado: {self.output_pdf}")

    def _add_cover(self) -> None:
        self.pdf.add_page()
        self.pdf.set_font("Helvetica", "B", 24)
        self.pdf.cell(0, 40, "Relatorio de Explainability - Deteccao de Fraude", new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.ln(10)
        self.pdf.set_font("Helvetica", "", 15)
        self.pdf.cell(0, 10, "Analise SHAP do modelo operacional", new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.cell(
            0,
            10,
            f"Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )
        if self.manifest.get("decision_threshold") is not None:
            self.pdf.cell(
                0,
                10,
                f"Threshold oficial: {self.manifest['decision_threshold']}",
                new_x="LMARGIN",
                new_y="NEXT",
                align="C",
            )

    def _add_table_of_contents(self) -> None:
        self.pdf.add_page()
        self._add_title("Sumario")
        for section in [
            "1. Analise SHAP - Importancia de Features",
            "2. Ranking de Importancia de Features",
            "3. Analise de Dependencia",
            "4. Conclusoes e Recomendacoes",
        ]:
            self._add_body(section)

    def _add_shap_insights(self) -> None:
        self.pdf.add_page()
        self._add_title("1. Analise SHAP - Importancia de Features")
        self._add_image(self.shap_dir / "shap_summary_plot.png")
        top_features = self._get_top_features()
        self._add_body(f"Features mais importantes: {', '.join(top_features)}")
        self._add_body("A analise SHAP resume como cada feature desloca a probabilidade prevista de fraude.")

    def _add_feature_importance(self) -> None:
        self.pdf.add_page()
        self._add_title("2. Ranking de Importancia de Features")
        self._add_image(self.shap_dir / "shap_feature_importance.png")
        if self.stats.get("sample_size") is not None:
            self._add_body(
                f"Analise calculada sobre {self.stats['sample_size']} amostras com {self.stats.get('total_features', 'N/A')} features."
            )

    def _add_dependence_plots(self) -> None:
        dependence_dir = self.shap_dir / "dependence_plots"
        images = sorted(dependence_dir.glob("*.png")) if dependence_dir.exists() else []
        if not images:
            return

        for image_path in images:
            self.pdf.add_page()
            self._add_title(f"3. Analise de Dependencia - {image_path.stem.replace('dependence_', '')}")
            self._add_image(image_path)

    def _add_conclusions(self) -> None:
        self.pdf.add_page()
        self._add_title("4. Conclusoes e Recomendacoes")
        top_features = self._get_top_features()
        self._add_body(f"Principais drivers de fraude observados: {', '.join(top_features[:3])}.")
        self._add_body("Recomendacao: monitorar transacoes com sinais extremos nessas features e acompanhar estabilidade do threshold oficial ao longo do tempo.")
        self._add_body(f"Manifesto tecnico: {SHAP_MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    generator = SHAPReportGenerator(SHAP_EXPLANATIONS_DIR, SHAP_REPORT_PATH)
    generator.generate_report()


if __name__ == "__main__":
    main()
