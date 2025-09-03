# conftest.py - VERSÃO PROFISSIONAL COMPLETA
import sys
import os
from pathlib import Path

def configure_project_paths():
    """Configuração profissional de paths para a estrutura src/config/"""
    
    # Diretório raiz do projeto
    root_dir = Path(__file__).parent.resolve()
    src_dir = root_dir / 'src'
    config_dir = src_dir / 'config'
    
    # Adiciona paths ao sys.path (evitando duplicatas)
    paths_to_add = [
        str(root_dir),  # Para imports absolutos: from src.config import ...
        str(src_dir),   # Para imports relativos: from config import ...
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Configura environment variables
    os.environ['PROJECT_ROOT'] = str(root_dir)
    
    # Verificação profissional
    print("🎯 CONFIGURAÇÃO PROFISSIONAL ATIVADA")
    print(f"   📁 Root: {root_dir}")
    print(f"   📁 Src: {src_dir}")
    print(f"   📁 Config: {config_dir}")
    print(f"   🐍 Python Path: {sys.path[0]}")
    
    return root_dir

# Executa a configuração
project_root = configure_project_paths()