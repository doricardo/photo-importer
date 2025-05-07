import sys
import os
import shutil
import string
import csv
import ctypes
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QSlider, QCheckBox, QProgressBar, QFileDialog, QDialog,
    QDialogButtonBox, QGridLayout, QSizePolicy, QSpacerItem,
    QGroupBox, QMessageBox, QSplashScreen
)
from PySide6.QtCore import Qt, QObject, Signal, QThread, QTimer
from PySide6.QtGui import QPixmap, QIcon, QPalette, QColor, QFont, QPainter

__version__ = '1.0'
APP_NAME = 'Photo Importer'

# -- DNN face model setup
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
PB    = os.path.join(MODEL_DIR, 'opencv_face_detector_uint8.pb')
PBTXT = os.path.join(MODEL_DIR, 'opencv_face_detector.pbtxt')
try:
    import urllib.request
    if not os.path.exists(PB):
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb',
            PB
        )
    if not os.path.exists(PBTXT):
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
            PBTXT
        )
except Exception as e:
    raise RuntimeError(f"Falha ao baixar modelo DNN: {e}")
net = cv2.dnn.readNetFromTensorflow(PB, PBTXT)

def get_volume_label(drive_path: str) -> str:
    buf1 = ctypes.create_unicode_buffer(1024)
    buf2 = ctypes.create_unicode_buffer(1024)
    serial = ctypes.c_uint()
    max_comp = ctypes.c_uint()
    flags = ctypes.c_uint()
    rc = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive_path), buf1, ctypes.sizeof(buf1),
        ctypes.byref(serial), ctypes.byref(max_comp), ctypes.byref(flags),
        buf2, ctypes.sizeof(buf2)
    )
    return buf1.value if rc else ''

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajuda – Simulação de Parâmetros")
        self.resize(800, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        params = ['Nitidez','Sombras','Congtraste','Brilho','Cor']
        desc = {
            'Nitidez':'Nitidez percebida',
            'Sombras':'Escurece/clareia sombras',
            'Congtraste':'Ajusta contraste',
            'Brilho':'Ajusta brilho',
            'Cor':'Ajusta saturação de cor'
        }
        self.sliders = {}
        for i, name in enumerate(params):
            lbl = QLabel(f"{name}: {desc[name]}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            val_lbl = QLabel('50')
            slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            slider.valueChanged.connect(self.update_preview)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(slider, i, 1)
            grid.addWidget(val_lbl, i, 2)
            self.sliders[name.lower()] = slider
        layout.addLayout(grid)

        h = QHBoxLayout()
        self.preview_orig = QLabel("—", alignment=Qt.AlignCenter)
        self.preview_mod  = QLabel("—", alignment=Qt.AlignCenter)
        h.addWidget(self.preview_orig)
        h.addWidget(self.preview_mod)
        layout.addLayout(h)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.Close).setText("Fechar")
        layout.addWidget(btns)

    def update_preview(self):
        # Placeholder: implementar preview com PIL
        pass

class ImportWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    total_copies = Signal(int)
    batch_copies = Signal(int,int)
    finished = Signal()

    def __init__(self, src, dst, mode, batch_size, verify_face, latency,
                 scale_factor, min_neighbors, min_face_size, conf_threshold,
                 sharpness, shadows, contraste, brilho, cor):
        super().__init__()
        self.src, self.dst, self.mode = src, dst, mode
        self.batch_size, self.verify_face = batch_size, verify_face
        self.latency, self.scale_factor = latency, scale_factor
        self.min_neighbors, self.min_face_size = min_neighbors, min_face_size
        self.conf_threshold = conf_threshold
        self.sharpness, self.shadows = sharpness, shadows
        self.contraste, self.brilho, self.cor = contraste, brilho, cor

    def run(self):
        remove_dir = os.path.join(self.dst, 'remove')
        os.makedirs(remove_dir, exist_ok=True)
        csv_path = os.path.join(self.dst, 'remove.csv')
        records = []

        exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
        images = [str(p) for p in Path(self.src).rglob('*') if p.suffix.lower() in exts]
        self.status.emit(f"Iniciando importação: {len(images)} imagens")
        proc = copied = 0
        bidx = 1 if self.mode == 'Batch' else 0
        bcnt = 0

        for path in images:
            name = os.path.basename(path)
            img_bgr = cv2.imread(path)
            blur_var = 0
            gray = None
            w_full = h_full = 0
            if img_bgr is not None:
                h_full, w_full = img_bgr.shape[:2]
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            no_face = False
            if self.verify_face and img_bgr is not None:
                blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300),
                                             [104.0, 177.0, 123.0], False, False)
                net.setInput(blob)
                det = net.forward()
                faces = []
                h, w = img_bgr.shape[:2]
                for i in range(det.shape[2]):
                    conf = float(det[0,0,i,2])
                    if conf >= self.conf_threshold:
                        x1 = int(det[0,0,i,3]*w)
                        y1 = int(det[0,0,i,4]*h)
                        x2 = int(det[0,0,i,5]*w)
                        y2 = int(det[0,0,i,6]*h)
                        faces.append((x1,y1,x2-x1,y2-y1))
                no_face = len(faces) == 0

            if (gray is not None and blur_var < self.latency) or no_face:
                reason = 'desfocada' if gray is not None and blur_var < self.latency else 'sem rosto'
                dstp = os.path.join(remove_dir, name)
                shutil.copy2(path, dstp)
                records.append([
                    name, reason,
                    f"variância={blur_var:.2f}",
                    f"tamanho={w_full}x{h_full}",
                    f"latência={self.latency}",
                    f"escala={self.scale_factor}",
                    f"vizinhos={self.min_neighbors}",
                    f"minFace={self.min_face_size}",
                    f"conf={self.conf_threshold}"
                ])
                self.status.emit(f"Copiado p/ remover: {name} ({reason})")
            else:
                if self.mode == 'All':
                    shutil.copy2(path, os.path.join(self.dst, name))
                    copied += 1
                    self.total_copies.emit(copied)
                    self.status.emit(f"Copiado: {name}")
                else:
                    if bcnt and bcnt % self.batch_size == 0:
                        bidx += 1
                        bcnt = 0
                    bd = os.path.join(self.dst, f"lote{bidx}")
                    os.makedirs(bd, exist_ok=True)
                    if bcnt % self.batch_size == 0:
                        self.status.emit(f"Criando lote{bidx}")
                    shutil.copy2(path, os.path.join(bd, name))
                    copied += 1
                    bcnt += 1
                    self.total_copies.emit(copied)
                    self.batch_copies.emit(bidx, bcnt)
                    self.status.emit(f"Copiado p/ lote{bidx}: {name}")

            proc += 1
            self.progress.emit(proc)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'arquivo','motivo','variância','tamanho',
                'latência','escala','vizinhos','minFace','conf'
            ])
            w.writerows(records)

        self.status.emit("Importação concluída")
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        self.setWindowIcon(QIcon(icon_path))

        self.last_dest = os.path.expanduser("~")
        self.setWindowTitle(APP_NAME)
        self.resize(820,700)
        self.setFont(QFont('Segoe UI',11))

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_elapsed)

        # CONTROLES
        self.src_combo = QComboBox(); self.populate_drives()
        btn_refresh   = QPushButton("Atualizar"); btn_refresh.clicked.connect(self.populate_drives)
        self.dst_edit  = QLineEdit(); self.dst_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_browse    = QPushButton("Selecionar…"); btn_browse.clicked.connect(self.choose_destination)

        self.mode_combo = QComboBox(); self.mode_combo.addItems(["Todas","Lote"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        self.lbl_batch  = QLabel("Imagens por lote:")
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(1,10000); self.spin_batch.setValue(1000)

        self.spin_latency   = QSpinBox(); self.spin_latency.setRange(0,500); self.spin_latency.setValue(100)
        self.chk_v          = QCheckBox("Verificar Face"); self.chk_v.setChecked(True)
        self.chk_v.toggled.connect(self.on_verify_face_toggle)

        self.spin_scale     = QDoubleSpinBox(); self.spin_scale.setRange(0.1,3.0); self.spin_scale.setSingleStep(0.01); self.spin_scale.setValue(1.01)
        self.spin_neighbors = QSpinBox(); self.spin_neighbors.setRange(1,20); self.spin_neighbors.setValue(1)
        self.spin_min_size  = QSpinBox(); self.spin_min_size.setRange(10,500); self.spin_min_size.setValue(10)
        self.spin_conf      = QDoubleSpinBox(); self.spin_conf.setRange(0.0,1.0); self.spin_conf.setSingleStep(0.01); self.spin_conf.setValue(0.10)

        central     = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)

        grp1 = QGroupBox("Configurações de Entrada")
        gi   = QVBoxLayout(grp1)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Fonte:")); r1.addWidget(self.src_combo); r1.addWidget(btn_refresh)
        r1.addSpacing(20)
        r1.addWidget(QLabel("Destino:")); r1.addWidget(self.dst_edit,1); r1.addWidget(btn_browse)
        gi.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Modo:")); r2.addWidget(self.mode_combo)
        r2.addWidget(self.lbl_batch); r2.addWidget(self.spin_batch)
        r2.addSpacerItem(QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum))
        gi.addLayout(r2)
        main_layout.addWidget(grp1)

        grp2 = QGroupBox("Parâmetros de Detecção")
        g2   = QHBoxLayout(grp2)
        g2.addWidget(QLabel("Latência:")); g2.addWidget(self.spin_latency)
        g2.addWidget(self.chk_v)
        g2.addWidget(QLabel("Escala:")); g2.addWidget(self.spin_scale)
        g2.addWidget(QLabel("Vizinhos:")); g2.addWidget(self.spin_neighbors)
        g2.addWidget(QLabel("Min. Face:")); g2.addWidget(self.spin_min_size)
        g2.addWidget(QLabel("Confiança:")); g2.addWidget(self.spin_conf)
        main_layout.addWidget(grp2)

        grp3 = QGroupBox("Ajustes de Imagem")
        g3   = QVBoxLayout(grp3)
        self.params = {}
        ajustes = [("Nitidez",50),("Sombras",50),("Congtraste",50),("Brilho",50),("Cor",50)]
        for nome, df in ajustes:
            row = QHBoxLayout()
            row.addWidget(QLabel(nome+":"))
            sl = QSlider(Qt.Horizontal); sl.setRange(0,100); sl.setValue(df)
            sp = QSpinBox(); sp.setRange(0,100); sp.setValue(df)
            sl.valueChanged.connect(sp.setValue); sp.valueChanged.connect(sl.setValue)
            row.addWidget(sl); row.addWidget(sp)
            g3.addLayout(row)
            self.params[nome.lower()] = sp
        main_layout.addWidget(grp3)

        grp4 = QGroupBox("Mensagens")
        g4   = QVBoxLayout(grp4)
        self.status_label = QLabel("Pronto"); g4.addWidget(self.status_label)
        self.copy_info    = QLabel("Copiadas: 0"); g4.addWidget(self.copy_info)
        self.time_label   = QLabel("Tempo decorrido: 00:00:00"); g4.addWidget(self.time_label)
        self.pb           = QProgressBar(); g4.addWidget(self.pb)
        main_layout.addWidget(grp4)

        btn_start = QPushButton("Iniciar Importação")
        btn_start.clicked.connect(self.start_import)
        hbtn = QHBoxLayout()
        hbtn.addSpacerItem(QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum))
        hbtn.addWidget(btn_start)
        main_layout.addLayout(hbtn)

        self.setCentralWidget(central)
        self.on_mode_change(self.mode_combo.currentText())
        self.on_verify_face_toggle(self.chk_v.isChecked())

    def on_verify_face_toggle(self, checked):
        for w in (self.spin_scale, self.spin_neighbors, self.spin_min_size, self.spin_conf):
            w.setEnabled(checked)

    def populate_drives(self):
        self.src_combo.clear()
        if sys.platform.startswith('win'):
            sd = os.getenv('SystemDrive','C:')+os.sep
            for d in string.ascii_uppercase:
                p = f"{d}:{os.sep}"
                if os.path.exists(p) and p.lower()!=sd.lower():
                    lbl = get_volume_label(p)
                    disp = f"{p} ({lbl})" if lbl else p
                    self.src_combo.addItem(disp,p)

    def choose_destination(self):
        path = QFileDialog.getExistingDirectory(self,"Selecionar Pasta de Destino",self.last_dest)
        if path:
            self.dst_edit.setText(path)
            self.last_dest = path

    def on_mode_change(self, modo):
        batch = (modo == "Lote")
        self.lbl_batch.setVisible(batch)
        self.spin_batch.setVisible(batch)

    def start_import(self):
        src = self.src_combo.currentData()
        images = [str(p) for p in Path(src).rglob('*')
                  if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]
        total = len(images)
        if total == 0:
            self.status_label.setText(f"Nenhuma imagem em '{src}' ({total})")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Confirmar Importação")
        msg.setText(f"Confirma a importação de {total} imagens?")
        msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        msg.button(QMessageBox.Yes).setText("Sim")
        msg.button(QMessageBox.No).setText("Não")
        if msg.exec() != QMessageBox.Yes:
            return

        dst = self.dst_edit.text().strip()
        if not dst:
            self.status_label.setText("Destino deve ser informado"); return
        if dst == src:
            self.status_label.setText("Destino deve ser diferente da fonte"); return

        old = getattr(self,'worker_thread',None)
        if old and old.isRunning():
            old.quit(); old.wait()

        self.start_time = datetime.now()
        self.timer.start()

        self.pb.setMaximum(total); self.pb.setValue(0)

        self.worker = ImportWorker(
            src,dst,
            "All" if self.mode_combo.currentText()=="Todas" else "Batch",
            self.spin_batch.value(),
            self.chk_v.isChecked(),
            self.spin_latency.value(),
            self.spin_scale.value(),
            self.spin_neighbors.value(),
            self.spin_min_size.value(),
            self.spin_conf.value(),
            self.params['nitidez'].value(),
            self.params['sombras'].value(),
            self.params['congtraste'].value(),
            self.params['brilho'].value(),
            self.params['cor'].value()
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress.connect(self.pb.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.total_copies.connect(lambda x: self.copy_info.setText(f"Copiadas: {x}"))
        self.worker.batch_copies.connect(lambda b,c: self.copy_info.setText(f"Lote {b}: {c} copiadas"))
        self.worker.finished.connect(self.timer.stop)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.update_elapsed)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def update_elapsed(self):
        delta = datetime.now() - self.start_time
        h, rem = divmod(delta.seconds,3600)
        m, s  = divmod(rem,60)
        self.time_label.setText(f"Tempo decorrido: {h:02d}:{m:02d}:{s:02d}")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # --- Splash Screen com título e versão + contagem regressiva de 5s ---
    splash_path = os.path.join(os.path.dirname(__file__),'splash.png')
    if os.path.exists(splash_path):
        pix = QPixmap(splash_path)
    else:
        pix = QPixmap(600,400)
        pix.fill(QColor(50,50,50))

    # desenha título e versão centralizados, com versão descida
    painter = QPainter(pix)
    painter.setPen(Qt.white)
    title_font = QFont('Segoe UI',32,QFont.Bold)
    painter.setFont(title_font)
    painter.drawText(pix.rect(), Qt.AlignCenter, APP_NAME)
    ver_font = QFont('Segoe UI',14)
    painter.setFont(ver_font)
    # versão fica 80px abaixo do topo
    painter.drawText(pix.rect().adjusted(0,80,0,0), Qt.AlignCenter, f'Versão {__version__}')
    painter.end()

    splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    window = MainWindow()
    remaining = {'sec': 5}
    align = Qt.AlignBottom | Qt.AlignHCenter

    def tick():
        sec = remaining['sec']
        splash.showMessage(f"Iniciando em {sec}...", align, QColor(255,255,255))
        app.processEvents()
        remaining['sec'] -= 1
        if remaining['sec'] < 0:
            timer.stop()
            splash.close()
            window.show()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(1000)
    QTimer.singleShot(0, tick)

    sys.exit(app.exec())
