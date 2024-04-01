import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("viridis")

class FraudSHAPExplainer:
    def __init__(self, model_path, data_path, output_dir="reports/shap_explanations"):
        self.model = joblib.load(model_path)
        self.data = pd.read_parquet(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preparar dados
        self.X = self.data.drop('Class', axis=1)
        self.y = self.data['Class']
        
        print(f"✅ Dados carregados: {self.X.shape}")
        print(f"✅ Modelo carregado: {type(self.model).__name__}")
    
    def generate_shap_analysis(self):
        """Gera análise SHAP completa"""
        print("🎯 Gerando análise SHAP...")
        
        # Explainer
        self.explainer = shap.TreeExplainer(self.model)  # 👈 Salvar como atributo
        shap_values = self.explainer.shap_values(self.X)
        
        # Para modelos de classificação
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Valores para classe positiva (fraude)
        
        # 1. Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X, show=False)
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        self._plot_feature_importance(shap_values)
        
        # 3. Dependence Plots para top features
        self._generate_dependence_plots(shap_values)
        
        # 4. Force Plot (exemplo)
        self._generate_force_plot(shap_values)
        
        # 5. Salvar valores SHAP
        self._save_shap_values(shap_values)
        
        print(f"✅ Análise SHAP salva em: {self.output_dir}")
    
    def _plot_feature_importance(self, shap_values):
        """Gráfico de importância de features"""
        importance_df = pd.DataFrame({
            'feature': self.X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(15))
        plt.title('Top 15 Features por Importância SHAP')
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_feature_importance.png", dpi=300)
        plt.close()
        
        # Salvar ranking
        importance_df.to_csv(self.output_dir / "feature_importance_ranking.csv", index=False)
    
    def _generate_dependence_plots(self, shap_values):
        """Gera dependence plots para top features"""
        dependence_dir = self.output_dir / "dependence_plots"
        dependence_dir.mkdir(exist_ok=True)
        
        # Top 6 features
        top_features = pd.DataFrame({
            'feature': self.X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False).head(6)['feature'].tolist()
        
        for feature in top_features:
            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, self.X, show=False)
                plt.title(f'Dependence Plot: {feature}')
                plt.tight_layout()
                plt.savefig(dependence_dir / f"dependence_{feature}.png", dpi=300)
                plt.close()
            except Exception as e:
                print(f"⚠️  Erro no dependence plot para {feature}: {e}")
    
    def _generate_force_plot(self, shap_values):
        """Gera force plot para exemplos específicos"""
        try:
            # Exemplo de instância fraudulenta (se existir)
            fraud_indices = self.y[self.y == 1].index
            if len(fraud_indices) > 0:
                instance_idx = fraud_indices[0]
                force_plot = shap.force_plot(
                    self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                    shap_values[instance_idx],
                    self.X.iloc[instance_idx],
                    show=False
                )
                shap.save_html(str(self.output_dir / "shap_force_plot.html"), force_plot)
        except Exception as e:
            print(f"⚠️  Erro ao gerar force plot: {e}")
    
    def _save_shap_values(self, shap_values):
        """Salva valores SHAP para análise posterior - CORRIGIDO"""
        try:
            # Salva valores SHAP completos
            shap_df = pd.DataFrame(shap_values, columns=self.X.columns)
            shap_df.to_csv(self.output_dir / "shap_values.csv", index=False)
            
            # Cria estatísticas resumidas IMPORTANTES
            mean_abs_shap = np.abs(shap_values).mean(0)
            top_features_idx = mean_abs_shap.argsort()[::-1][:10]
            
            stats = {
                "mean_abs_shap": mean_abs_shap.tolist(),
                "features": self.X.columns.tolist(),
                "top_features": [
                    {
                        "feature": self.X.columns[i],
                        "importance": float(mean_abs_shap[i]),
                        "direction": "Positive" if np.mean(shap_values[:, i]) > 0 else "Negative"
                    }
                    for i in top_features_idx
                ],
                "shap_summary": {
                    "total_samples": len(shap_values),
                    "total_features": len(self.X.columns),
                    "mean_expected_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value)
                }
            }
            
            # Garante que o diretório existe
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Salva o arquivo JSON
            with open(self.output_dir / "shap_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
                
            print(f"✅ SHAP stats saved to: {self.output_dir / 'shap_stats.json'}")
            
        except Exception as e:
            print(f"❌ Error saving SHAP stats: {e}")
            import traceback
            traceback.print_exc()

def main():
    # Configurar paths
    model_path = "models/model.pkl"
    data_path = "dados/credicard_tratado.parquet"
    output_dir = "reports/shap_explanations"
    
    # Gerar insights
    explainer = FraudSHAPExplainer(model_path, data_path, output_dir)
    explainer.generate_shap_analysis()
    
    print("🎉 Análise SHAP concluída com sucesso!")
    
    # Verificação final
    import os
    stats_file = "reports/shap_explanations/shap_stats.json"
    if os.path.exists(stats_file):
        print(f"🎉 {stats_file} gerado com sucesso!")
        # Mostra preview do conteúdo
        with open(stats_file, 'r') as f:
            data = json.load(f)
            print(f"📊 Top feature: {data['top_features'][0]['feature']} "
                  f"(importance: {data['top_features'][0]['importance']:.4f})")
    else:
        print(f"❌ {stats_file} NÃO foi gerado!")

if __name__ == "__main__":
    main()