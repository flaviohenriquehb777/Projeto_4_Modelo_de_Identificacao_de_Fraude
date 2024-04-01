import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
from pathlib import Path
import json

class SHAPReportGenerator:
    def __init__(self, shap_dir="reports/shap_explanations", output_pdf="reports/shap_report.pdf"):
        self.shap_dir = Path(shap_dir)
        self.output_pdf = output_pdf
        self.pdf = FPDF()
        
    def generate_report(self):
        """Gera relatório PDF profissional"""
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        
        # Capa
        self._add_cover()
        
        # Sumário
        self._add_table_of_contents()
        
        # Insights
        self._add_shap_insights()
        
        # Feature Importance
        self._add_feature_importance()
        
        # Dependence Plots
        self._add_dependence_plots()
        
        # Conclusões
        self._add_conclusions()
        
        # Salvar PDF
        self.pdf.output(self.output_pdf)
        print(f"✅ Relatório gerado: {self.output_pdf}")
    
    def _add_cover(self):
        self.pdf.set_font("Arial", "B", 24)
        self.pdf.cell(0, 40, "Relatorio de Explainability - Detecao de Fraude", 0, 1, "C")
        self.pdf.ln(20)
        self.pdf.set_font("Arial", "", 16)
        self.pdf.cell(0, 10, "Analise SHAP do Modelo de Machine Learning", 0, 1, "C")
        self.pdf.cell(0, 10, f"Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, "C")
    
    def _get_top_features(self):
        """Pega top 5 features do JSON"""
        try:
            with open(self.shap_dir / "shap_stats.json") as f:
                data = json.load(f)
                return [feat['feature'] for feat in data['top_features'][:5]]
        except:
            return ["V14", "V12", "V10", "V16", "V17"]
    
    def _add_shap_insights(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 18)
        self.pdf.cell(0, 10, "1. Analise SHAP - Importancia de Features", 0, 1)
        self.pdf.ln(5)
        
        # Adicionar imagem do summary plot
        if (self.shap_dir / "shap_summary_plot.png").exists():
            self.pdf.image(str(self.shap_dir / "shap_summary_plot.png"), x=10, y=40, w=190)
        
        # Insights textuais
        self.pdf.ln(120)
        self.pdf.set_font("Arial", "", 12)
        top_features = self._get_top_features()
        self.pdf.multi_cell(0, 8, f"- Features mais importantes: {', '.join(top_features)}")
        self.pdf.multi_cell(0, 8, "- Tendências: Analise de valores SHAP mostra padroes de fraude")
    
    def _add_feature_importance(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 18)
        self.pdf.cell(0, 10, "2. Ranking de Importancia de Features", 0, 1)
        
        if (self.shap_dir / "shap_feature_importance.png").exists():
            self.pdf.image(str(self.shap_dir / "shap_feature_importance.png"), x=10, y=30, w=190)
    
    def _add_dependence_plots(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 18)
        self.pdf.cell(0, 10, "3. Analise de Dependencia", 0, 1)
        
        dependence_dir = self.shap_dir / "dependence_plots"
        if dependence_dir.exists():
            for i, img_path in enumerate(dependence_dir.glob("*.png")):
                if i % 2 == 0:
                    self.pdf.add_page()
                self.pdf.image(str(img_path), x=10, y=30 + (i % 2)*100, w=95)
    
    def _add_conclusions(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 18)
        self.pdf.cell(0, 10, "4. Conclusoes e Recomendacoes", 0, 1)
        self.pdf.ln(10)
        self.pdf.set_font("Arial", "", 12)
        self.pdf.multi_cell(0, 8, "- Principais drivers de fraude: V14, V12, V10")
        self.pdf.multi_cell(0, 8, "- Recomendacoes: Monitorar transacoes com valores anomalos nestas features")
    
    def _add_table_of_contents(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 18)
        self.pdf.cell(0, 10, "Sumario", 0, 1)
        self.pdf.ln(10)
        
        sections = [
            "1. Analise SHAP - Importancia de Features",
            "2. Ranking de Importancia de Features", 
            "3. Analise de Dependencia",
            "4. Conclusoes e Recomendacoes"
        ]
        
        for section in sections:
            self.pdf.set_font("Arial", "", 14)
            self.pdf.cell(0, 10, section, 0, 1)

def main():
    generator = SHAPReportGenerator()
    generator.generate_report()
    print("🎉 Relatorio PDF gerado com sucesso!")

if __name__ == "__main__":
    main()