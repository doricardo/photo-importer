# Photo Importer

Uma ferramenta desktop para Windows e macOS para importar grandes quantidades de fotos de um drive de origem para uma pasta de destino, com filtros automáticos de qualidade (detecção de desfoque, rostos) e organização em lotes.

---

## Detalhes do Projeto

O **Photo Importer** foi desenvolvido em Python usando PySide6 para a interface gráfica, Pillow para ajustes de imagem e OpenCV DNN para detecção de faces.  
Ele permite:

- Selecionar um drive de origem e uma pasta de destino.
- Configurar modo de importação: **Todas** (copia todas as fotos) ou **Lote** (divide em pastas `lote1`, `lote2`, …).
- Filtrar automaticamente fotos **desfocadas** (baseado em variância de Laplacian) ou **sem rosto** (detecção DNN).
- Ajustar parâmetros de sensibilidade: latência (desfoque), escala, vizinhança, tamanho mínimo de face, confiança, nitidez, sombras, contraste, brilho e saturação.
- Visualizar progresso em tempo real (barra de progresso, contador de fotos copiadas, tempo decorrido).
- Gerar um relatório `remove.csv` com todas as imagens descartadas e seus motivos.

---

## Objetivos

1. **Automatizar** a cópia e organização de grandes acervos de fotos.
2. **Garantir** qualidade mínima ao importar (removendo fotos fora de foco ou sem rosto).
3. **Oferecer** interface amigável e responsiva, com visual moderno.
4. **Permitir** ajustes finos de todos os parâmetros de detecção e filtragem.
5. **Facilitar** deploy: gerar bundle para Windows (.exe) e macOS (.app).

---

## Tecnologias

- **Python 3.10+**  
- **PySide6** (Qt for Python)  
- **Pillow** (PIL)  
- **OpenCV DNN**  
- **PyInstaller** / **py2app** para empacotamento  

---

## Estrutura

```md
photo-importer/
├── models/ # Modelos DNN (pesos e config)
├── icon.png # Ícone da aplicação
├── splash.png # Imagem de splash screen
├── photo_importer_app.py # Código-fonte principal
├── remove.csv # Relatório gerado após importação
├── README.md # Este arquivo
└── setup.py # Script de build (py2app/pyinstaller)
```

---

## Como Usar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/photo-importer.git
   cd photo-importer

2. Instale dependências:
   pip install -r requirements.txt

3. Execute
   python photo_importer_app.py

4. (Opcional) Crie o bundle:

macOS: python setup.py py2app
Windows: pyinstaller --onefile --windowed --add-data "models;models" --icon icon.ico photo_importer_app.py
